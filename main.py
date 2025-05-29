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
        self.max_steps_per_episode = 50 # Keep this as 50 for now for fair comparison with previous run

        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32)
        self.user_position_init = np.array([[50.0, 10.0, 1.5]], dtype=np.float32)
        self.attacker_position_init = np.array([[40.0, -15.0, 1.5]], dtype=np.float32)

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
                # f_phi = tf.reduce_mean(tf.squeeze(f_phi)) # f_phi is not used later, can be commented
                logging.debug(f"f_theta: {f_theta.numpy()}, f_phi: {f_phi.numpy()}") # f_phi might be 0 or near 0 for V polarization
                f_theta = tf.cast(f_theta, tf.complex64)
            except Exception as e:
                logging.warning(f"Field computation failed: {e}. Using default gain.")
                f_theta = tf.constant(1.0, dtype=tf.complex64)

            num_bs_physical_rows_float = tf.cast(self.num_bs_physical_rows, dtype=tf.float32)
            num_bs_physical_cols_float = tf.cast(self.num_bs_physical_cols, dtype=tf.float32)

            row_indices = tf.range(self.num_bs_physical_rows, dtype=tf.float32) - (num_bs_physical_rows_float - 1.0) / 2.0
            col_indices = tf.range(self.num_bs_physical_cols, dtype=tf.float32) - (num_bs_physical_cols_float - 1.0) / 2.0

            d_v = d_h = 0.5
            wavelength = 3e8 / self.carrier_frequency
            k = 2 * np.pi / wavelength

            row_grid, col_grid = tf.meshgrid(row_indices, col_indices, indexing='ij')
            positions_y = row_grid * d_v * wavelength
            positions_x = col_grid * d_h * wavelength
            positions_z = tf.zeros_like(positions_x)
            positions = tf.stack([positions_x, positions_y, positions_z], axis=-1)
            positions = tf.reshape(positions, [-1, 3])

            sin_theta_val = tf.sin(zenith_rad) # Renamed for clarity
            cos_theta_val = tf.cos(zenith_rad) # Renamed for clarity
            sin_phi_val = tf.sin(azimuth_rad)  # Renamed for clarity
            cos_phi_val = tf.cos(azimuth_rad)  # Renamed for clarity

            direction = tf.stack([sin_theta_val * cos_phi_val, sin_theta_val * sin_phi_val, cos_theta_val], axis=0)
            direction = tf.expand_dims(direction, axis=0)

            phase_shifts = k * tf.reduce_sum(positions * direction, axis=-1)
            array_response = tf.exp(tf.complex(0.0, phase_shifts))

            sv = array_response * f_theta
            if sv.shape[0] != self.num_bs_antennas:
                logging.error(f"Steering vector calculation resulted in {sv.shape[0]} elements, expected {self.num_bs_antennas}. Falling back to tf.ones.")
                sv = tf.ones([self.num_bs_antennas], dtype=tf.complex64)

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
            in_state = tf.zeros([current_batch_size, self.num_user], dtype=tf.bool) # Assuming this in_state is for the UMa model's topology setting

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

            # For attacker channel, we need to use a similar in_state for the UMa model topology if num_attacker is >0
            # Assuming num_attacker is 1, so in_state_attacker would be tf.zeros([current_batch_size, self.num_attacker], dtype=tf.bool)
            # However, the UMa model in Sionna typically expects in_state per UT, so if we model attacker as another UT type,
            # it should be handled correctly if the model is configured for num_ut = num_user + num_attacker, or called sequentially.
            # Current sequential call with re-set_topology is fine.
            self.channel_model_core.set_topology(
                ut_loc=attacker_loc_reshaped,
                bs_loc=bs_loc_reshaped,
                ut_orientations=ut_orientation, # Assuming same orientation model for attacker as for user for simplicity
                bs_orientations=bs_orientation,
                ut_velocities=ut_velocity,    # Assuming same velocity model
                in_state=in_state # This should ideally be in_state_attacker if shape differs. Given num_user=1, num_attacker=1, current in_state should be fine.
            )
            h_attacker_time_all_paths, _ = self.channel_model_core(num_time_samples_val, sampling_frequency_val)
            h_attacker = tf.reduce_mean(h_attacker_time_all_paths[:, 0, 0, 0, :, 0, 0], axis=0)
            tf.ensure_shape(h_attacker, [self.num_bs_antennas])
            logging.debug(f"h_attacker norm: {tf.norm(h_attacker).numpy()}")

            steering_vec = self._get_steering_vector(current_beam_angles_tf_in)
            steering_vec = tf.reshape(steering_vec, [-1])
            precoder_w = tf.math.conj(steering_vec) / (tf.norm(steering_vec) + 1e-9) # Adding epsilon for numerical stability
            precoder_w = tf.reshape(precoder_w, [-1, 1])

            h_user_row_vec = tf.cast(h_user, tf.complex64)
            h_attacker_row_vec = tf.cast(h_attacker, tf.complex64)

            y_user_eff_scalar = tf.reduce_sum(h_user_row_vec * tf.squeeze(precoder_w))
            y_attacker_eff_scalar = tf.reduce_sum(h_attacker_row_vec * tf.squeeze(precoder_w))

            signal_power_user = tf.abs(y_user_eff_scalar)**2 * self.tx_power_watts
            signal_power_attacker = tf.abs(y_attacker_eff_scalar)**2 * self.tx_power_watts
            logging.debug(f"Signal power user: {signal_power_user.numpy()}, attacker: {signal_power_attacker.numpy()}")

            if signal_power_user < 1e-15: # Slightly lower threshold
                logging.warning("Signal power user too low, setting default SINR adjustment power")
                signal_power_user = tf.constant(1e-15, dtype=tf.float32)
            if signal_power_attacker < 1e-15: # Slightly lower threshold
                logging.warning("Signal power attacker too low, setting default power")
                signal_power_attacker = tf.constant(1e-15, dtype=tf.float32)

            return signal_power_user, signal_power_attacker
        except Exception as e:
            logging.error(f"Error in _get_channel_and_powers: {e}")
            return tf.constant(1e-15, dtype=tf.float32), tf.constant(1e-15, dtype=tf.float32)

    def _get_state(self):
        logging.debug("Entering _get_state")
        try:
            signal_power_user_tf, signal_power_attacker_tf = self._get_channel_and_powers(
                self.current_beam_angles_tf, self.user_position, self.attacker_position
            )
            signal_power_user = signal_power_user_tf.numpy()
            # Ensure no_lin is not zero or extremely small to prevent overflow/very large SINR
            sinr_user_val = 10 * log10(tf.cast(signal_power_user / (self.no_lin + 1e-20), tf.float32)).numpy() if signal_power_user > 1e-15 else -30.0
            sinr_user_clipped = np.clip(sinr_user_val, -30.0, 30.0)

            sensing_noise_std_factor = 1.0 - (self.current_isac_effort * 0.75) # Lower effort, higher noise
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]

            true_attacker_vector = attacker_pos_np - bs_pos_np
            true_attacker_range = np.linalg.norm(true_attacker_vector)
            true_attacker_azimuth = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])

            detected_az, detected_range, confidence = -np.pi, 0.0, 0.0 # Default to 0 range if not detected

            if true_attacker_range <= self.sensing_range_max and true_attacker_range > 0:
                prob_detection = self.current_isac_effort * np.exp(-0.01 * true_attacker_range)
                prob_detection = np.minimum(1.0, prob_detection * 1.75) # Max prob_detection is 1.0
                logging.debug(f"prob_detection: {prob_detection}, true_attacker_range: {true_attacker_range}, isac_effort: {self.current_isac_effort}")
                if self.np_random.random() < prob_detection:
                    confidence = prob_detection + self.np_random.normal(0, 0.05) # Add some noise to confidence
                    confidence = np.clip(confidence, 0.0, 1.0)
                    noise_az = self.np_random.normal(0, np.deg2rad(10) * sensing_noise_std_factor) # Increased noise slightly
                    noise_range = self.np_random.normal(0, 5.0 * sensing_noise_std_factor) # Increased noise slightly
                    detected_az = true_attacker_azimuth + noise_az
                    detected_range = max(0.1, true_attacker_range + noise_range) # Ensure range is positive
                    logging.debug(f"Attacker detected: confidence={confidence}, detected_range={detected_range}, detected_az={detected_az}")
            # Ensure detected values are within observation space if not detected or out of range
            detected_az = np.clip(detected_az, -np.pi, np.pi)
            detected_range = np.clip(detected_range, 0, self.sensing_range_max)


            state_array = np.array([
                sinr_user_clipped,
                self.current_beam_angles_tf[0].numpy(),
                detected_az, # Already clipped
                detected_range, # Already clipped
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
        self.current_isac_effort = 0.7 # Initial ISAC effort

        if not hasattr(self, 'np_random') or self.np_random is None:
            logging.warning("self.np_random not initialized. Initializing manually.")
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # More controlled randomization for initial positions for consistency in testing
        user_pos_offset_x = self.np_random.uniform(-5.0, 5.0)
        user_pos_offset_y = self.np_random.uniform(-5.0, 5.0)
        self.user_position.assign(self.user_position_init + np.array([[user_pos_offset_x, user_pos_offset_y, 0.0]], dtype=np.float32))

        attacker_pos_offset_x = self.np_random.uniform(-10.0, 10.0)
        attacker_pos_offset_y = self.np_random.uniform(-10.0, 10.0)
        self.attacker_position.assign(self.attacker_position_init + np.array([[attacker_pos_offset_x, attacker_pos_offset_y, 0.0]], dtype=np.float32))

        self.current_step = 0
        obs = self._get_state()
        if obs is None: # Should not happen with the try-except in _get_state
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
        elif action_idx == 2: pass # Hold beam
        elif action_idx == 3: self.current_isac_effort += 0.2 # Increase ISAC effort
        elif action_idx == 4: self.current_isac_effort -= 0.2 # Decrease ISAC effort

        self.current_isac_effort = np.clip(self.current_isac_effort, 0.3, 1.0) # Ensure ISAC effort is within bounds
        current_az_val = np.clip(current_az_val, -np.pi, np.pi) # Clip beam angle
        self.current_beam_angles_tf.assign([current_az_val])

        # User and attacker movement
        user_move_dist = self.np_random.uniform(0, 0.1) # Small movement
        user_move_angle = self.np_random.uniform(-np.pi, np.pi)
        user_pos_offset = np.array([[user_move_dist * np.cos(user_move_angle), user_move_dist * np.sin(user_move_angle), 0.0]], dtype=np.float32)
        self.user_position.assign_add(tf.constant(user_pos_offset, dtype=tf.float32))

        attacker_move_dist = self.np_random.uniform(0, 0.5) # Slightly larger movement
        attacker_move_angle = self.np_random.uniform(-np.pi, np.pi)
        attacker_pos_offset = np.array([[attacker_move_dist * np.cos(attacker_move_angle), attacker_move_dist * np.sin(attacker_move_angle), 0.0]], dtype=np.float32)
        self.attacker_position.assign_add(tf.constant(attacker_pos_offset, dtype=tf.float32))


        next_state = self._get_state()
        reward = self._compute_reward(next_state)
        self.current_step += 1

        terminated = False
        truncated = False # For Gymnasium compatibility, set truncated if episode ends due to time limit

        if self.current_step >= self.max_steps_per_episode:
            truncated = True # Gymnasium uses truncated for time limits

        if self._is_beam_stolen(next_state):
            terminated = True
            reward -= 50 # Stronger penalty for being stolen

        # If SINR is very low for a sustained period, could also terminate
        if next_state[0] < -10.0: # More aggressive SINR threshold for termination
            if not terminated : reward -= 15 # Penalty if not already terminated by beam stealing
            terminated = True

        # If attacker gets too close to user or BS and is not detected, penalize heavily and terminate
        bs_pos_np = self.bs_position.numpy()[0,:2] # x,y of BS
        attacker_pos_np = self.attacker_position.numpy()[0,:2] # x,y of attacker
        user_pos_np = self.user_position.numpy()[0,:2] # x,y of user

        dist_attacker_bs = np.linalg.norm(attacker_pos_np - bs_pos_np)
        dist_attacker_user = np.linalg.norm(attacker_pos_np - user_pos_np)

        # If attacker is very close to BS or User, and confidence is low
        if (dist_attacker_bs < 10.0 or dist_attacker_user < 10.0) and next_state[4] < 0.5 : # next_state[4] is confidence
            if not terminated: reward -= 25 # Heavy penalty
            terminated = True
            logging.info(f"TERMINATED: Attacker too close ({dist_attacker_bs:.1f}m from BS, {dist_attacker_user:.1f}m from User) and not detected (conf={next_state[4]:.2f}).")


        return next_state, reward, terminated, truncated, {}

    def _is_beam_stolen(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_az_rad = current_obs_state[2]
        detected_attacker_range_m = current_obs_state[3]
        detection_confidence = current_obs_state[4]

        # Stricter conditions for beam stealing: higher confidence, closer range, larger SINR drop
        if detection_confidence > 0.7 and 0 < detected_attacker_range_m < 75: # Attacker must be reasonably close and confidently detected
            angle_diff_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_az = min(angle_diff_az, 2*np.pi - angle_diff_az) # Normalize
            if angle_diff_az < np.deg2rad(15) and sinr_user < -10.0: # Beam aligned with attacker & very low SINR
                logging.info(f"BEAM STOLEN: AngleDiff={np.rad2deg(angle_diff_az):.1f}deg, UserSINR={sinr_user:.1f}dB, AttRange={detected_attacker_range_m:.1f}m")
                return True
        return False

    def _compute_reward(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_az_rad = current_obs_state[2]
        detected_attacker_range_m = current_obs_state[3]
        detection_conf = current_obs_state[4]
        true_attacker_range = current_obs_state[5]
        true_attacker_azimuth = current_obs_state[6]

        # --- SINR-based Reward ---
        if sinr_user > 15: # Target SINR from paper
            reward = 15.0 + (sinr_user - 15.0) * 0.5 # Strong reward for exceeding target
        elif sinr_user > 5:
            reward = sinr_user * 0.5 # Moderate reward for acceptable SINR
        else:
            reward = sinr_user # Penalize low SINR, clip at -30

        # --- Detection-based Reward/Penalty ---
        if detection_conf > 0.7 and detected_attacker_range_m > 0: # Confident detection
            reward += 10.0 * detection_conf # Strong reward for confident detection

            # Penalty for beam misalignment with *USER* if attacker is detected (encourage focusing on user)
            # Calculate vector to user
            user_vector = self.user_position.numpy()[0] - self.bs_position.numpy()[0]
            true_user_azimuth = np.arctan2(user_vector[1], user_vector[0])
            angle_diff_beam_user = abs(beam_az_rad - true_user_azimuth)
            angle_diff_beam_user = min(angle_diff_beam_user, 2 * np.pi - angle_diff_beam_user)
            # Penalize if beam is not pointing to user when an attacker is detected
            reward -= (angle_diff_beam_user / np.pi) * 5.0 # Penalty scaled by misalignment

            # Accuracy of detection (bonus for low DOA error)
            doa_error_rad = abs(detected_attacker_az_rad - true_attacker_azimuth)
            doa_error_rad = min(doa_error_rad, 2*np.pi - doa_error_rad)
            reward += (1.0 - doa_error_rad / np.pi) * 5.0 * detection_conf # Bonus for accurate DOA

        elif true_attacker_range < self.sensing_range_max * 0.75 : # Attacker is present and relatively close but NOT detected confidently
            reward -= 10.0 # Strong penalty for missing a nearby attacker

        # --- ISAC Effort Penalty ---
        # Penalize very high or very low ISAC effort to encourage a balance
        if self.current_isac_effort > 0.9:
            reward -= (self.current_isac_effort - 0.9) * 10.0 # Penalize excessive effort
        elif self.current_isac_effort < 0.4 and (true_attacker_range < self.sensing_range_max * 0.75 and detection_conf < 0.5) : # Low effort when nearby attacker is missed
             reward -= 5.0


        # --- Penalty for beam pointing towards actual attacker if attacker is close ---
        if true_attacker_range < self.sensing_range_max * 0.5: # If attacker is close
            angle_diff_beam_true_attacker = abs(beam_az_rad - true_attacker_azimuth)
            angle_diff_beam_true_attacker = min(angle_diff_beam_true_attacker, 2 * np.pi - angle_diff_beam_true_attacker)
            if angle_diff_beam_true_attacker < np.deg2rad(30): # If beam is somewhat towards the true attacker
                 reward -= (1.0 - angle_diff_beam_true_attacker / np.pi) * 5.0 * (1 - detection_conf) # Penalize more if not detected

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
                print(f"True Attacker: Az={np.rad2deg(current_render_state[6]):.1f}°, Range={current_render_state[5]:.1f}m")

            else:
                print("Attacker Not Detected or Low Confidence.")
                print(f"True Attacker: Az={np.rad2deg(current_render_state[6]):.1f}°, Range={current_render_state[5]:.1f}m")


    def close(self):
        pass

# DRL Agent (Double DQN)
class DoubleDQNAgent:
    def __init__(self, state_dim, action_n, learning_rate=0.0001, gamma=0.99, # Adjusted gamma slightly
             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=15000): # Adjusted epsilon_end
        self.state_dim = int(state_dim)
        self.action_n = action_n
        self.memory = deque(maxlen=50000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.current_learning_steps = 0
        self.learning_rate = learning_rate
        self.batch_size = 32 # Changed to 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'), # Increased units
            keras.layers.Dense(64, activation='relu'),  # Increased units
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
        # TensorFlow operations should be used within tf.function for performance if possible,
        # but for direct model.predict, it's generally fine.
        act_values = self.model.predict(state_for_pred, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0 # Or some other indicator that replay didn't happen

        minibatch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Double DQN: Q(s', argmax_a' Q_online(s', a'))
        # Predict actions for next_states using the online model
        next_actions_online = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        # Evaluate Q-values of these actions using the target model
        q_values_next_target = self.target_model.predict(next_states, verbose=0)
        # Select Q-values corresponding to the actions chosen by the online model
        q_values_next_state_selected = q_values_next_target[np.arange(self.batch_size), next_actions_online]

        targets_for_taken_actions = rewards + self.gamma * q_values_next_state_selected * (1 - dones)

        # Get current Q-values for the states in the minibatch
        current_q_values_all_actions = self.model.predict(states, verbose=0)
        # Update the Q-values for the actions that were actually taken
        for i in range(self.batch_size):
            current_q_values_all_actions[i, actions[i]] = targets_for_taken_actions[i]

        # Train the online model
        history = self.model.fit(states, current_q_values_all_actions, epochs=1, verbose=0, batch_size=self.batch_size) # Added batch_size here

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon) # Ensure epsilon doesn't go below end value

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
    # Parameters from the latest suggestions/article
    agent = DoubleDQNAgent(state_dim, action_n,
                           learning_rate=0.0001, # Increased LR
                           gamma=0.99,
                           epsilon_end=0.05,
                           epsilon_decay_steps=15000)

    episodes = 500
    target_update_freq_steps = 1000 # As per article/common practice
    print_freq_episodes = 10
    save_freq_episodes = 50
    episode_data = []
    attack_detections_in_episode = [] # To store detection outcomes per step
    total_training_steps = 0

    # Early stopping parameters from article
    early_stop_avg_sinr_threshold = 15.0
    early_stop_detection_rate_threshold = 75.0 # Percentage
    early_stop_consecutive_blocks = 50 # Article mentions 50 consecutive episode blocks
    early_stop_block_size = 10 # Each block spans 10 episodes
    consecutive_good_blocks_count = 0


    for e in range(episodes):
        current_state, _ = env.reset()
        total_episode_reward = 0
        episode_sinr_values = []
        episode_detections = [] # For this episode's detections

        for step in range(env.max_steps_per_episode):
            action_idx = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action_idx)

            episode_sinr_values.append(next_state[0])
            detection_confidence = next_state[4]
            detected_range = next_state[3] # state[3] is detected range
            # Store detection outcome for this step
            attack_detected_this_step = 1 if (detection_confidence > 0.5 and 0 < detected_range < 100) else 0
            episode_detections.append(attack_detected_this_step)


            agent.remember(current_state, action_idx, reward, next_state, terminated or truncated) # Pass terminated or truncated for 'done'
            current_state = next_state
            total_episode_reward += reward

            if len(agent.memory) >= agent.batch_size: # Start replay once buffer has enough samples
                loss = agent.replay() # replay() returns loss
                total_training_steps += 1
                if total_training_steps % 100 == 0: # Log loss occasionally
                    logging.debug(f"Episode {e+1}, Step {step+1}, Training Step {total_training_steps}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")


            if total_training_steps > 0 and total_training_steps % target_update_freq_steps == 0:
                agent.update_target_model()
                logging.info(f"Target model updated at training step {total_training_steps}")


            if terminated or truncated:
                break
        
        # After episode ends, store aggregated detection rate for this episode
        # This attack_detections_in_episode is for the overall average, not per-episode average for printout
        attack_detections_in_episode.extend(episode_detections)


        avg_episode_sinr = np.mean(episode_sinr_values) if episode_sinr_values else -30.0 # Default if no steps
        avg_episode_detection_rate = np.mean(episode_detections) * 100 if episode_detections else 0.0

        episode_data.append({
            'Episode': e + 1,
            'Total_Reward': total_episode_reward,
            'Avg_SINR': avg_episode_sinr,
            'Avg_Detection_Rate_Episode': avg_episode_detection_rate, # New metric
            'Steps': step + 1,
            'Epsilon': agent.epsilon # Log epsilon at end of episode
        })

        if (e + 1) % print_freq_episodes == 0:
            # Calculate averages over the last 'print_freq_episodes'
            avg_r_block = np.mean([d['Total_Reward'] for d in episode_data[-print_freq_episodes:]])
            avg_s_block = np.mean([d['Avg_SINR'] for d in episode_data[-print_freq_episodes:]])
            # For overall detection rate, use `attack_detections_in_episode` which accumulates all step-wise detections
            # For block-wise detection rate, average the per-episode average detection rates
            avg_dr_block = np.mean([d['Avg_Detection_Rate_Episode'] for d in episode_data[-print_freq_episodes:]])

            print(f"Ep: {e+1}/{episodes} | Avg Reward (last {print_freq_episodes}): {avg_r_block:.2f} | Avg SINR (last {print_freq_episodes}): {avg_s_block:.2f} dB | Avg Det.Rate (last {print_freq_episodes} ep.): {avg_dr_block:.2f}% | Epsilon: {agent.epsilon:.3f}")

            # Early stopping check (based on block averages)
            if avg_s_block > early_stop_avg_sinr_threshold and avg_dr_block > early_stop_detection_rate_threshold:
                consecutive_good_blocks_count += 1
                logging.info(f"Episode {e+1}: Good block performance. Consecutive count: {consecutive_good_blocks_count}")
                if consecutive_good_blocks_count >= (early_stop_consecutive_blocks): #Block is 10 episodes, so 50 blocks * 10 = 500 episodes. But paper meant 50 * (10 episode blocks)
                    print(f"Early stopping at episode {e+1} after {consecutive_good_blocks_count} consecutive high-performance blocks.")
                    logging.info(f"Early stopping at episode {e+1}.")
                    break
            else:
                if consecutive_good_blocks_count > 0:
                     logging.info(f"Episode {e+1}: Resetting consecutive good blocks count from {consecutive_good_blocks_count}. SINR: {avg_s_block:.2f}, Det.Rate: {avg_dr_block:.2f}")
                consecutive_good_blocks_count = 0


        if (e + 1) % save_freq_episodes == 0:
            agent.save(f"drl_isac_secure_beam_ep{e+1}.weights.h5") # Changed filename slightly

    df = pd.DataFrame(episode_data)
    print(f"Total training time: {(time.time() - start_time)/60:.2f} minutes")
    df.to_csv('episode_results_v2.csv', index=False) # Changed filename
    print("Episode results saved to 'episode_results_v2.csv'")

    # --- Evaluation Phase ---
    print("\n--- Starting Evaluation Phase ---")
    evaluation_episodes = 8 # Number of test scenarios
    evaluation_steps = env.max_steps_per_episode # Or a fixed number like 50 for consistency if max_steps_per_episode changes for training

    # Save current agent epsilon and set to 0 for greedy evaluation
    eval_epsilon_backup = agent.epsilon
    agent.epsilon = 0.0
    print(f"Agent epsilon set to {agent.epsilon} for evaluation.")


    evaluation_data = []
    for i_eval in range(evaluation_episodes):
        # --- Baseline Evaluation ---
        state_b, _ = env.reset() # Resets user/attacker pos, beam angle to 0, ISAC effort to 0.7
        # For baseline, we assume a fixed beam (e.g., 0 deg) and fixed ISAC effort (e.g., 0.7, which is env's reset default)
        # Let's run the baseline for a few steps to get an average
        baseline_sinrs = []
        baseline_detections_conf = [] # Store confidence for more nuanced detection check
        baseline_detected_attacker_ranges = []

        for _ in range(evaluation_steps): # Run for same number of steps as DRL eval
            # In baseline, no action is taken by DRL; assume fixed beam/ISAC effort as per reset
            # We need to manually step the environment if we want to simulate movement
            # For simplicity, let's take one _get_state() reading after reset for this simple baseline.
            # Or, better, simulate random small movements for a few steps without DRL actions
            # For now, let's use the first state after reset for baseline metrics
            # For a more robust baseline, you might want to average over a few steps with fixed actions
            # or pre-defined heuristic actions.
            # Let's stick to initial state after reset for simplicity of "fixed beam" baseline
            pass # No DRL action for baseline, just use the initial state after reset.

        # To make baseline more comparable, let's do one get_state for it.
        # The env.reset() already calls _get_state(), so state_b is the initial state.
        # Re-ensure beam is at 0 for baseline (env.reset does this, but good to be explicit if changing logic)
        env.current_beam_angles_tf.assign([0.0])
        env.current_isac_effort = 0.7 # Fixed ISAC effort for baseline
        baseline_current_state = env._get_state()

        baseline_sinr = baseline_current_state[0]
        baseline_detected_conf = baseline_current_state[4]
        baseline_detected_range = baseline_current_state[3]
        baseline_detected = 1 if (baseline_detected_conf > 0.5 and 0 < baseline_detected_range < 100) else 0


        # --- DRL Agent Evaluation ---
        state_drl, _ = env.reset() # Fresh reset for DRL agent for this evaluation episode
        drl_sinrs_episode = []
        drl_detections_episode_conf = []
        drl_detected_attacker_ranges_episode = []

        for _ in range(evaluation_steps):
            action_idx_drl = agent.act(state_drl) # Agent acts greedily (epsilon=0)
            next_state_drl, _, terminated_drl, truncated_drl, _ = env.step(action_idx_drl)
            drl_sinrs_episode.append(next_state_drl[0])
            drl_detections_episode_conf.append(next_state_drl[4])
            drl_detected_attacker_ranges_episode.append(next_state_drl[3])
            state_drl = next_state_drl
            if terminated_drl or truncated_drl:
                break
        
        avg_drl_sinr_eval = np.mean(drl_sinrs_episode) if drl_sinrs_episode else -30.0
        # DRL detection: check if attacker was detected (conf > 0.5 and valid range) in *any* step of the eval episode
        drl_detected_in_eval_episode = 0
        for conf, drange in zip(drl_detections_episode_conf, drl_detected_attacker_ranges_episode):
            if conf > 0.5 and 0 < drange < 100:
                drl_detected_in_eval_episode = 1
                break


        evaluation_data.append({
            'Test_Scenario': i_eval + 1,
            'Baseline_SINR': baseline_sinr,
            'DRL_SINR_Avg_Eval': avg_drl_sinr_eval,
            'Baseline_Detected_Binary': baseline_detected,
            'DRL_Detected_In_Episode_Binary': drl_detected_in_eval_episode
        })
        print(f"Eval Scenario {i_eval+1}: Baseline SINR={baseline_sinr:.2f}, DRL SINR={avg_drl_sinr_eval:.2f}, Baseline Det={baseline_detected}, DRL Det={drl_detected_in_eval_episode}")


    # Restore agent epsilon
    agent.epsilon = eval_epsilon_backup
    print(f"Agent epsilon restored to {agent.epsilon}.")


    df_eval = pd.DataFrame(evaluation_data)
    df_eval.to_csv('evaluation_results_v2.csv', index=False) # Changed filename
    print("Evaluation results saved to 'evaluation_results_v2.csv'")

    # Calculate and print overall average evaluation metrics
    avg_baseline_sinr_overall = np.mean([d['Baseline_SINR'] for d in evaluation_data])
    avg_drl_sinr_overall = np.mean([d['DRL_SINR_Avg_Eval'] for d in evaluation_data])
    avg_baseline_detection_overall = np.mean([d['Baseline_Detected_Binary'] for d in evaluation_data]) * 100
    avg_drl_detection_overall = np.mean([d['DRL_Detected_In_Episode_Binary'] for d in evaluation_data]) * 100

    print(f"\n--- Overall Evaluation Averages ---")
    print(f"Average Baseline SINR: {avg_baseline_sinr_overall:.2f} dB")
    print(f"Average DRL SINR (greedy): {avg_drl_sinr_overall:.2f} dB")
    print(f"Baseline Detection Rate: {avg_baseline_detection_overall:.2f}%")
    print(f"DRL Detection Rate (greedy, detected in episode): {avg_drl_detection_overall:.2f}%")


    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 6)) # Adjusted figure size

        plt.subplot(1, 3, 1)
        plt.plot(df['Episode'], df['Total_Reward'], label='Total Reward')
        # Moving average for rewards
        if len(df['Total_Reward']) >= print_freq_episodes:
            plt.plot(df['Episode'], df['Total_Reward'].rolling(window=print_freq_episodes, center=True, min_periods=1).mean(), label=f'Moving Avg (window {print_freq_episodes})', linestyle='--')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)


        plt.subplot(1, 3, 2)
        plt.plot(df['Episode'], df['Avg_SINR'], label='Avg SINR')
        if len(df['Avg_SINR']) >= print_freq_episodes:
            plt.plot(df['Episode'], df['Avg_SINR'].rolling(window=print_freq_episodes, center=True, min_periods=1).mean(), label=f'Moving Avg (window {print_freq_episodes})', linestyle='--')
        plt.title('Average SINR per Episode')
        plt.xlabel('Episode')
        plt.ylabel('SINR (dB)')
        plt.legend()
        plt.grid(True)


        plt.subplot(1, 3, 3)
        plt.plot(df['Episode'], df['Avg_Detection_Rate_Episode'], label='Avg Detection Rate per Episode')
        if len(df['Avg_Detection_Rate_Episode']) >= print_freq_episodes:
            plt.plot(df['Episode'], df['Avg_Detection_Rate_Episode'].rolling(window=print_freq_episodes, center=True, min_periods=1).mean(), label=f'Moving Avg (window {print_freq_episodes})', linestyle='--')
        plt.title('Average Attack Detection Rate per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Detection Rate (%)')
        plt.legend()
        plt.grid(True)


        plt.tight_layout()
        plt.savefig('training_progress_v2.png') # Changed filename
        print("Saved training_progress_v2.png")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"Error during plot generation: {e}")


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