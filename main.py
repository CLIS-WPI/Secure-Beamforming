import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import tensorflow.keras as keras
from collections import deque
import random
import sys
import pandas as pd
import logging
import os
import time
start_time = time.time()
# Force disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging to file
logging.basicConfig(filename='simulation.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check TensorFlow and Sionna versions
def check_tf_sionna_versions():
    logging.info(f"Using TensorFlow version: {tf.__version__}")
    tf_major_minor = float('.'.join(tf.__version__.split('.')[:2]))
    if not (tf_major_minor >= 2.14):
        logging.warning(f"TensorFlow version {tf.__version__} might be older than expected for latest Sionna features.")
    try:
        import sionna
        logging.info(f"Using Sionna version: {sionna.__version__}")
    except ImportError:
        pass

check_tf_sionna_versions()

# Sionna Imports with Error Handling
try:
    from sionna.phy.channel.tr38901 import UMa
    from sionna.phy.channel.tr38901.antenna import PanelArray
    from sionna.phy.utils import log10
except ImportError as e:
    logging.error(f"Failed to import Sionna components: {e}")
    print(f"CRITICAL ERROR: Failed to import Sionna components: {e}")
    print("Please ensure Sionna is installed correctly.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Unexpected error during Sionna imports: {e}")
    print(f"An unexpected error occurred during Sionna imports: {e}")
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# mmWave ISAC Environment
class MmWaveISACEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(MmWaveISACEnv, self).__init__()

        self.carrier_frequency = 28e9
        self.bandwidth = 100e6
        self.num_bs_physical_rows = 8
        self.num_bs_physical_cols = 8

        try:
            self.bs_array = PanelArray(
                num_rows_per_panel=self.num_bs_physical_rows,
                num_cols_per_panel=self.num_bs_physical_cols,
                polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=self.carrier_frequency
            )
            logging.info("Sionna PanelArray for BS initialized successfully.")
            self.num_bs_antennas = self.bs_array.num_ant
            logging.info(f"Total BS antenna ports defined: {self.num_bs_antennas}")

            self.ut_array = PanelArray(
                num_rows_per_panel=1,
                num_cols_per_panel=1,
                polarization='single',
                polarization_type='V',
                antenna_pattern='omni',
                carrier_frequency=self.carrier_frequency
            )
            logging.info("Sionna PanelArray for UT (omni) initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Sionna PanelArray objects: {e}")
            print(f"CRITICAL ERROR: Failed to initialize Sionna PanelArray objects: {e}")
            sys.exit(1)

        self.num_user = 1
        self.num_bs = 1
        self.num_attacker = 1
        self.tx_power_dbm = 30.0
        self.tx_power_watts = 10**((self.tx_power_dbm - 30) / 10)
        self.noise_power_db_per_hz = -174.0
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth)
        self.no_lin = 10**((no_db - 30) / 10)
        self.sensing_range_max = 150.0
        self.max_steps_per_episode = 50

        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32)
        self.user_position_init = np.array([[50.0, 10.0, 1.5]], dtype=np.float32)
        self.attacker_position_init = np.array([[40.0, -15.0, 1.5]], dtype=np.float32)  # Closer attacker

        try:
            self.channel_model_core = UMa(
                carrier_frequency=self.carrier_frequency,
                o2i_model="low",
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction="downlink",
                enable_pathloss=True,
                enable_shadow_fading=True
            )
            logging.info("Sionna UMa channel model object created successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Sionna UMa channel model object: {e}")
            print(f"CRITICAL ERROR: Failed to initialize Sionna UMa channel model object: {e}")
            sys.exit(1)

        try:
            initial_ut_loc = tf.reshape(self.user_position_init, [1, self.num_user, 3])
            initial_bs_loc = tf.reshape(self.bs_position, [1, self.num_bs, 3])
            initial_ut_orientations = tf.zeros([1, self.num_user, 3], dtype=tf.float32)
            initial_bs_orientations = tf.zeros([1, self.num_bs, 3], dtype=tf.float32)
            initial_ut_velocities = tf.zeros([1, self.num_user, 3], dtype=tf.float32)
            initial_in_state = tf.zeros([1, self.num_user], dtype=tf.bool)

            self.channel_model_core.set_topology(
                ut_loc=initial_ut_loc,
                bs_loc=initial_bs_loc,
                ut_orientations=initial_ut_orientations,
                bs_orientations=initial_bs_orientations,
                ut_velocities=initial_ut_velocities,
                in_state=initial_in_state
            )
            logging.info("Sionna UMa channel model topology set successfully.")
        except Exception as e:
            logging.error(f"Failed to set topology for Sionna UMa channel model: {e}")
            print(f"CRITICAL ERROR: Failed to set topology for Sionna UMa channel model: {e}")
            sys.exit(1)

        # Expanded state space
        low_obs = np.array([-30.0, -np.pi, -np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)
        high_obs = np.array([30.0, np.pi, np.pi, self.sensing_range_max, 1.0, self.sensing_range_max, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.num_discrete_actions = 5
        self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.beam_angle_delta_rad = np.deg2rad(5)

        self.user_position = tf.Variable(self.user_position_init, dtype=tf.float32)
        self.attacker_position = tf.Variable(self.attacker_position_init, dtype=tf.float32)
        self.current_beam_angles_tf = tf.Variable([0.0], dtype=tf.float32)
        self.current_isac_effort = 0.7
        self.current_step = 0

    def _get_steering_vector(self, angles_rad_tf):
        logging.debug("Entering _get_steering_vector")
        try:
            azimuth_rad = tf.reshape(angles_rad_tf[0], [])
            zenith_rad = tf.constant(np.pi/2, dtype=tf.float32)

            try:
                field_output = self.bs_array.ant_pol1.field(theta=zenith_rad, phi=azimuth_rad)
                if not (isinstance(field_output, tuple) and len(field_output) == 2):
                    logging.warning(f"Unexpected field output: {type(field_output)}")
                    raise ValueError("Invalid field output")
                
                f_theta, f_phi = field_output
                f_theta = tf.reduce_mean(tf.squeeze(f_theta))
                f_phi = tf.reduce_mean(tf.squeeze(f_phi))
                logging.debug(f"f_theta: {f_theta.numpy()}, f_phi: {f_phi.numpy()}")
                f_theta = tf.cast(f_theta, tf.complex64)
            except Exception as e:
                logging.warning(f"Field computation failed: {e}. Using default gain.")
                f_theta = tf.constant(1.0, dtype=tf.complex64)

            num_elements = tf.cast(self.num_bs_physical_rows, dtype=tf.float32)
            num_cols = tf.cast(self.num_bs_physical_cols, dtype=tf.float32)

            row_indices = tf.range(self.num_bs_physical_rows, dtype=tf.float32) - (num_rows - 1.0) / 2.0
            col_indices = tf.range(self.num_bs_physical_cols, dtype=tf.float32) - (num_cols - 1.0) / 2.0

            d_v = d_h = 0.5
            wavelength = 3e8 / self.carrier_frequency
            k = 2 * np.pi / wavelength

            row_grid, col_grid = tf.meshgrid(row_indices, col_indices, indexing='ij')
            positions_y = row_grid * d_v * wavelength
            positions_x = col_grid * d_h * wavelength
            positions_z = tf.zeros_like(positions_x)
            positions = tf.stack([positions_x, positions_y, positions_z], axis=-1)
            positions = tf.reshape(positions, [-1, 3])

            sin_theta = tf.sin(zenith_rad)
            cos_theta = tf.cos(zenith_rad)
            sin_phi = tf.sin(azimuth_rad)
            cos_phi = tf.cos(azimuth_rad)

            direction = tf.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)
            direction = tf.expand_dims(direction, axis=0)

            phase_shifts = k * tf.reduce_sum(positions * direction, axis=-1)
            array_response = tf.exp(tf.complex(0.0, phase_shifts))

            sv = array_response * f_theta
            if sv.shape[0] != num_elements:
                logging.error(f"sv has {sv.shape[0]} elements, expected {num_elements}")
                sv = tf.ones([num_elements], dtype=tf.complex64)

            sv = tf.cast(sv, tf.complex64)
            sv = tf.ensure_shape(sv, [self.num_bs_antennas])
            logging.debug(f"Steering vector norm: {tf.norm(sv).numpy()}")
            return sv
        except Exception as e:
            logging.error(f"Error in _get_steering_vector: {e}")
            return tf.ones([self.num_bs_antennas], dtype=tf.complex64)

    def _get_channel_and_powers(self, current_beam_angles_tf_in, user_pos_in, attacker_pos_in):
        logging.debug("Entering _get_channel_and_powers")
        try:
            current_batch_size = tf.shape(user_pos_in)[0]
            bs_loc_reshaped = tf.reshape(self.bs_position, [current_batch_size, self.num_bs, 3])
            user_loc_reshaped = tf.reshape(user_pos_in, [current_batch_size, self.num_user, 3])
            attacker_loc_reshaped = tf.reshape(attacker_pos_in, [current_batch_size, self.num_attacker, 3])

            bs_orientation = tf.zeros([current_batch_size, self.num_bs, 3], dtype=tf.float32)
            bs_velocity = tf.zeros([current_batch_size, self.num_bs, 3], dtype=tf.float32)
            ut_orientation = tf.zeros([current_batch_size, self.num_user, 3], dtype=tf.float32)
            ut_velocity = tf.zeros([current_batch_size, self.num_user, 3], dtype=tf.float32)
            in_state = tf.zeros([current_batch_size, self.num_user], dtype=tf.bool)

            num_time_samples_val = 10
            sampling_frequency_val = tf.cast(self.bandwidth, dtype=tf.float32)

            self.channel_model_core.set_topology(
                ut_loc=user_loc_reshaped,
                bs_loc=bs_loc_reshaped,
                ut_orientations=ut_orientation,
                bs_orientations=bs_orientation,
                ut_velocities=ut_velocity,
                in_state=in_state
            )
            h_user_time_all_paths, _ = self.channel_model_core(num_time_samples_val, sampling_frequency_val)
            h_user = tf.reduce_mean(h_user_time_all_paths[:, 0, 0, 0, :, 0, 0], axis=0)
            tf.ensure_shape(h_user, [self.num_bs_antennas])
            logging.debug(f"h_user norm: {tf.norm(h_user).numpy()}")

            self.channel_model_core.set_topology(
                ut_loc=attacker_loc_reshaped,
                bs_loc=bs_loc_reshaped,
                ut_orientations=ut_orientation,
                bs_orientations=bs_orientation,
                ut_velocities=ut_velocity,
                in_state=in_state
            )
            h_attacker_time_all_paths, _ = self.channel_model_core(num_time_samples_val, sampling_frequency_val)
            h_attacker = tf.reduce_mean(h_attacker_time_all_paths[:, 0, 0, 0, :, 0, 0], axis=0)
            tf.ensure_shape(h_attacker, [self.num_bs_antennas])
            logging.debug(f"h_attacker norm: {tf.norm(h_attacker).numpy()}")

            steering_vec = self._get_steering_vector(current_beam_angles_tf_in)
            steering_vec = tf.reshape(steering_vec, [-1])
            precoder_w = tf.math.conj(steering_vec) / (tf.norm(steering_vec) + 1e-9)
            precoder_w = tf.reshape(precoder_w, [-1, 1])

            h_user_row_vec = tf.cast(h_user, tf.complex64)
            h_attacker_row_vec = tf.cast(h_attacker, tf.complex64)

            y_user_eff_scalar = tf.reduce_sum(h_user_row_vec * tf.squeeze(precoder_w))
            y_attacker_eff_scalar = tf.reduce_sum(h_attacker_row_vec * tf.squeeze(precoder_w))

            signal_power_user = tf.abs(y_user_eff_scalar)**2 * self.tx_power_watts
            signal_power_attacker = tf.abs(y_attacker_eff_scalar)**2 * self.tx_power_watts
            logging.debug(f"Signal power user: {signal_power_user.numpy()}, attacker: {signal_power_attacker.numpy()}")

            # بررسی مقادیر غیرواقعی
            if signal_power_user < 1e-12:
                logging.warning("Signal power user too low, setting default SINR")
                signal_power_user = tf.constant(1e-12, dtype=tf.float32)
            if signal_power_attacker < 1e-12:
                logging.warning("Signal power attacker too low, setting default")
                signal_power_attacker = tf.constant(1e-12, dtype=tf.float32)

            return signal_power_user, signal_power_attacker
        except Exception as e:
            logging.error(f"Error in _get_channel_and_powers: {e}")
            return tf.constant(1e-12, dtype=tf.float32), tf.constant(1e-12, dtype=tf.float32)

    def _get_state(self):
        logging.debug("Entering _get_state")
        try:
            signal_power_user_tf, signal_power_attacker_tf = self._get_channel_and_powers(
                self.current_beam_angles_tf, self.user_position, self.attacker_position
            )
            signal_power_user = signal_power_user_tf.numpy()
            sinr_user_val = 10 * log10(tf.cast(signal_power_user / self.no_lin, tf.float32)).numpy() if signal_power_user > 1e-12 else -30.0
            sinr_user_clipped = np.clip(sinr_user_val, -30.0, 30.0)

            sensing_noise_std_factor = 1.0 - (self.current_isac_effort * 0.75)
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]

            true_attacker_vector = attacker_pos_np - bs_pos_np
            true_attacker_range = np.linalg.norm(true_attacker_vector)
            true_attacker_azimuth = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])

            detected_az, detected_range, confidence = -np.pi, -1.0, 0.0

            if true_attacker_range <= self.sensing_range_max and true_attacker_range > 0:
                prob_detection = self.current_isac_effort * np.exp(-0.01 * true_attacker_range)
                prob_detection = np.minimum(1.0, prob_detection * 1.75)
                logging.debug(f"prob_detection: {prob_detection}, true_attacker_range: {true_attacker_range}, isac_effort: {self.current_isac_effort}")
                if self.np_random.random() < prob_detection:
                    confidence = prob_detection + self.np_random.normal(0, 0.05)
                    confidence = np.clip(confidence, 0.0, 1.0)
                    noise_az = self.np_random.normal(0, np.deg2rad(5) * sensing_noise_std_factor)
                    noise_range = self.np_random.normal(0, 2.0 * sensing_noise_std_factor)
                    detected_az = true_attacker_azimuth + noise_az
                    detected_range = max(0.1, true_attacker_range + noise_range)
                    logging.debug(f"Attacker detected: confidence={confidence}, detected_range={detected_range}, detected_az={detected_az}")

            state_array = np.array([
                sinr_user_clipped,
                self.current_beam_angles_tf[0].numpy(),
                np.clip(detected_az, -np.pi, np.pi),
                np.clip(detected_range, 0, self.sensing_range_max),
                confidence,
                np.clip(true_attacker_range, 0, self.sensing_range_max),
                np.clip(true_attacker_azimuth, -np.pi, np.pi)
            ], dtype=np.float32)
            logging.debug(f"Returning state_array: {state_array}")
            return state_array
        except Exception as e:
            logging.error(f"Exception in _get_state: {e}")
            return np.array([-30.0, 0.0, -np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)

    def reset(self, seed=None, options=None):
        logging.debug("Entering MmWaveISACEnv.reset")
        super().reset(seed=seed)
        self.current_beam_angles_tf.assign([0.0])
        self.current_isac_effort = 0.7

        if not hasattr(self, 'np_random') or self.np_random is None:
            logging.warning("self.np_random not initialized. Initializing manually.")
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        user_pos_offset = self.np_random.normal(0, 1.0, 3).reshape(1,3).astype(np.float32)
        attacker_pos_offset = self.np_random.normal(0, 2.0, 3).reshape(1,3).astype(np.float32)
        self.user_position.assign(self.user_position_init + user_pos_offset)
        self.attacker_position.assign(self.attacker_position_init + attacker_pos_offset)
        self.current_step = 0

        obs = self._get_state()
        if obs is None:
            logging.error("reset: _get_state returned None. Returning default state.")
            obs = np.array([-30.0, 0.0, -np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)
        info = {}
        logging.debug(f"reset returning obs shape: {obs.shape}")
        return obs, info

    def step(self, action_idx):
        logging.debug(f"Entering step with action_idx: {action_idx}")
        current_az_val = self.current_beam_angles_tf.numpy()[0]

        if action_idx == 0: current_az_val -= self.beam_angle_delta_rad
        elif action_idx == 1: current_az_val += self.beam_angle_delta_rad
        elif action_idx == 2: pass
        elif action_idx == 3: self.current_isac_effort = min(1.0, self.current_isac_effort + 0.2)
        elif action_idx == 4: self.current_isac_effort = max(0.3, self.current_isac_effort - 0.2)

        self.current_isac_effort = np.clip(self.current_isac_effort, 0.3, 1.0)
        current_az_val = np.clip(current_az_val, -np.pi, np.pi)
        self.current_beam_angles_tf.assign([current_az_val])

        user_pos_offset = self.np_random.normal(0, 0.05, 3).astype(np.float32).reshape(1,3)
        attacker_pos_offset = self.np_random.normal(0, 0.2, 3).astype(np.float32).reshape(1,3)
        self.user_position.assign_add(tf.constant(user_pos_offset, dtype=tf.float32))
        self.attacker_position.assign_add(tf.constant(attacker_pos_offset, dtype=tf.float32))

        next_state = self._get_state()
        if next_state is None:
            logging.error("step: _get_state returned None. Using default state.")
            next_state = np.array([-30.0, 0.0, -np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)

        reward = self._compute_reward(next_state)

        self.current_step += 1
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps_per_episode:
            truncated = True

        if self._is_beam_stolen(next_state):
            terminated = True
            reward -= 20

        if next_state[0] < -5.0:
            if not terminated: reward -= 10
            terminated = True

        return next_state, reward, terminated, truncated, {}

    def _is_beam_stolen(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_az_rad = current_obs_state[2]
        detected_attacker_range_m = current_obs_state[3]
        detection_confidence = current_obs_state[4]

        if detection_confidence > 0.5 and 0 < detected_attacker_range_m < 100:
            angle_diff_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_az = min(angle_diff_az, 2*np.pi - angle_diff_az)
            if angle_diff_az < np.deg2rad(20) and sinr_user < -5.0:
                logging.info(f"BEAM STOLEN: AngleDiff={np.rad2deg(angle_diff_az):.1f}deg, UserSINR={sinr_user:.1f}dB, AttRange={detected_attacker_range_m:.1f}m")
                return True
        return False

    def _compute_reward(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_rad = current_obs_state[2]
        detected_attacker_range_m = current_obs_state[3]
        detection_conf = current_obs_state[4]

        reward = np.clip(sinr_user / 2.0, -6.0, 6.0)  # افزایش حساسیت
        if sinr_user > 16: reward += 10.0  # پاداش قوی
        elif sinr_user > 12: reward += 5.0
        elif sinr_user > 8.0: reward += 2.0
        if detection_conf > 0.6 and self.current_step % 5 == 0:  # سخت‌تر
            reward += 1.5

        if detection_conf > 0.3 and detected_attacker_range_m > 0:
            true_attacker_rad = current_obs_state[6]
            doa_error_rad = abs(detected_attacker_rad - true_attacker_rad)
            doa_error_rad = min(doa_error_rad, 2*np.pi - doa_error_rad)
            reward += (1.0 - doa_error_rad / np.pi) * 2.0 * detection_conf
            angle_diff_beam = abs(beam_rad - detected_attacker_rad)
            angle_diff_beam = min(angle_diff_beam_rad, 2*np.pi - angle_diff_rad)
            reward -= (1.0 - angle_diff_beam_rad / np.pi) * 1.5 * detection_conf
        else:
            true_attacker_m = current_obs_state[5]
            if true_attacker_m < self.sensing_range_m * 0.5:
                reward -= 0.15  # جریمه کمتر

        if self.current_is > 0.85: reward -= 0.15
        elif self.current_is < 0.35: reward -= 0.1
        return float(reward)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, ISAC Effort: {self.current_isac_effort:.2f}")
            current_render_state = self._get_state()
            print(f"Beam Azimuth: {np.rad2deg(current_render_state[1]):.1f}°")
            print(f"User SINR: {current_render_state[0]:.2f} dB")
            if current_render_state[4] > 0.1:
                print(f"Detected Attacker: Az={np.rad2deg(current_render_state[2]):.1f}°, "
                      f"Range={current_render_state[3]:.1f}m (Conf: {current_render_state[4]:.2f})")
            else:
                print("Attacker Not Detected or Low Confidence.")

    def close(self):
        pass

# DRL Agent (Double DQN)
class DoubleDQNAgent:
    def __init__(self, state_dim, action_n, learning_rate=0.00002, gamma=0.995,
             epsilon_start=1.0, epsilon_end=0.15, epsilon_decay_steps=12000):
        self.state_dim = int(state_dim)
        self.action_n = action_n
        self.memory = deque(maxlen=50000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.current_learning_steps = 0
        self.learning_rate = learning_rate
        self.batch_size = 24

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(64, activation='relu'),  # تغییر از 256
            keras.layers.Dense(32, activation='relu'),  # تغییر از 128
            keras.layers.Dense(self.action_n, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        self.current_learning_steps += 1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_n)
        if state.ndim == 1: state_for_pred = np.expand_dims(state, axis=0)
        else: state_for_pred = state
        act_values = self.model.predict(state_for_pred, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        minibatch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Double DQN: Select actions using online model, evaluate using target model
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        q_values_next_state_target = self.target_model.predict(next_states, verbose=0)
        targets_for_taken_actions = rewards + self.gamma * q_values_next_state_target[np.arange(self.batch_size), next_actions] * (1 - dones)
        current_q_values_all_actions = self.model.predict(states, verbose=0)

        for i in range(self.batch_size):
            current_q_values_all_actions[i, actions[i]] = targets_for_taken_actions[i]

        history = self.model.fit(states, current_q_values_all_actions, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_value
            self.epsilon = max(self.epsilon_end, self.epsilon)

        return history.history['loss'][0]

    def save(self, name):
        try:
            self.model.save_weights(name)
            logging.info(f"Model weights saved to {name}")
        except Exception as e:
            logging.error(f"Error saving model weights to {name}: {e}")

# Main simulation function
def run_simulation():
    print("Starting DRL Simulation for Secure mmWave ISAC Beamforming...")
    env = MmWaveISACEnv()
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DoubleDQNAgent(state_dim, action_n, epsilon_decay_steps=15000)

    episodes = 500
    target_update_freq_steps = 1000
    print_freq_episodes = 10
    save_freq_episodes = 50
    episode_data = []
    attack_detections = []
    total_training_steps = 0
    early_stop_count = 0
    early_stop_threshold = 50

    for e in range(episodes):
        current_state, _ = env.reset()
        total_episode_reward = 0
        episode_sinr = []

        for step in range(env.max_steps_per_episode):
            action_idx = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            episode_sinr.append(next_state[0])
            detection_confidence = next_state[4]
            detected_range = next_state[3]
            attack_detected = 1 if (detection_confidence > 0.5 and 0 < detected_range < 100) else 0
            attack_detections.append(attack_detected)
            agent.remember(current_state, action_idx, reward, next_state, terminated)
            current_state = next_state
            total_episode_reward += reward

            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                total_training_steps += 1

            if total_training_steps % target_update_freq_steps == 0:
                agent.update_target_model()

            if terminated or truncated:
                break

        episode_data.append({
            'Episode': e + 1,
            'Total_Reward': total_episode_reward,
            'Avg_SINR': np.mean(episode_sinr) if episode_sinr else 0,
            'Steps': step + 1
        })

        if (e + 1) % print_freq_episodes == 0:
            avg_r = np.mean([d['Total_Reward'] for d in episode_data[-print_freq_episodes:]])
            avg_s = np.mean([d['Avg_SINR'] for d in episode_data[-print_freq_episodes:]])
            detection_rate = np.mean(attack_detections[-print_freq_episodes*env.max_steps_per_episode:]) * 100
            print(f"Ep: {e+1}/{episodes} | Avg Reward: {avg_r:.4f} | Avg SINR: {avg_s:.2f} dB | Detection Rate: {detection_rate:.2f}%")
            if avg_s > 15.0 and detection_rate > 75.0:
                early_stop_count += 1
                if early_stop_count >= early_stop_threshold:
                    print(f"Early stopping at episode {e+1} due to sustained high performance.")
                    break
            else:
                early_stop_count = 0

        if (e + 1) % save_freq_episodes == 0:
            agent.save(f"drl_beam_simplified_ep{e+1}.weights.h5")

    df = pd.DataFrame(episode_data)
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    df.to_csv('episode_results.csv', index=False)
    print("Episode results saved to 'episode_results.csv'")
    
    evaluation_data = []
    for _ in range(8):  
        state, _ = env.reset()
        prev_isac_effort = env.current_isac_effort
        env.current_beam_angles_tf.assign([0.])
        baseline_state = env._get_state()
        baseline_sinr = baseline_state[0]
        baseline_detected = 1 if (baseline_state[4] > 0.5 and 0 < baseline_state[3] < 64) else 0
        env.current_isac_effort = prev_isac_effort

        state, _ = env.reset()
        drl_sines = []
        for _ in range(6):  
            action_idx = agent.act(state)
            next_state, _, _, _, _ = env.step(action_idx)
            drl_sines.append(next_state[0])
            state = next_state
        drl_sinr = np.mean(drl_sines)
        drl_detected = 1 if (next_state[4] > 0.5 and 0 < next_state[3] < 64) else 0

        evaluation_data.append({
            'Test': len(evaluation_data) + 1,
            'Baseline_SINR': baseline_sinr,
            'DRL_SINR': drl_sinr,
            'Baseline_Detected': baseline_detected,
            'DRL_Detected': drl_detected
        })

    df_eval = pd.DataFrame(evaluation_data)
    df_eval.to_csv('evaluation_results.csv', index=False)
    print("Evaluation results saved to 'evaluation_results.csv'")
    print(f"Average Baseline SINR: {np.mean([d['Baseline_SINR'] for d in evaluation_data]):.2f} dB")
    print(f"Average DRL SINR: {np.mean([d['DRL_SINR'] for d in evaluation_data]):.2f} dB")
    print(f"Baseline Detection Rate: {np.mean([d['Baseline_Detected'] for d in evaluation_data]) * 100:.2f}%")
    print(f"DRL Detection Rate: {np.mean([d['DRL_Detected'] for d in evaluation_data]) * 100:.2f}%")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot([d['Total_Reward'] for d in episode_data])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(1, 3, 2)
        plt.plot([d['Avg_SINR'] for d in episode_data])
        plt.title('Average SINR per Episode')
        plt.xlabel('Episode')
        plt.ylabel('SINR (dB)')

        plt.subplot(1, 3, 3)
        detection_rates = [np.mean(attack_detections[i:i+print_freq_episodes*env.max_steps_per_episode]) * 100 
                          for i in range(0, len(attack_detections), print_freq_episodes*env.max_steps_per_episode)]
        plt.plot(detection_rates)
        plt.title('Attack Detection Rate')
        plt.xlabel(f'Episode Block (x{print_freq_episodes})')
        plt.ylabel('Detection Rate (%)')

        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("Saved training_progress.png")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        try:
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  GPU {i}: {gpu.name} - Memory growth enabled.")
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow sees {len(logical_gpus)} Logical GPU(s).")
        except RuntimeError as e:
            print(f"RuntimeError during GPU setup: {e}")
    else:
        print("No GPU detected. Running on CPU.")
    run_simulation()