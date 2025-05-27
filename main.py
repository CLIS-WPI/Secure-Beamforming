import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import tensorflow.keras as keras
from collections import deque
import random
import sys # For sys.exit

# --- Helper function to check TensorFlow version ---
def check_tf_sionna_versions():
    """Checks if the TensorFlow version is compatible with Sionna (typically 2.14-2.19)."""
    print(f"Using TensorFlow version: {tf.__version__}")
    tf_major_minor = float('.'.join(tf.__version__.split('.')[:2]))
    # Sionna's typical TF compatibility range (adjust if specific Sionna version has different reqs)
    if not (2.14 <= tf_major_minor <= 2.19): # Common range, check Sionna docs for your version
        print(f"Warning: TensorFlow version {tf.__version__} might not be fully compatible. "
              f"Sionna often requires TensorFlow versions between 2.14 and 2.19.")
    
    # Check Sionna version (optional, but good practice)
    try:
        import sionna
        print(f"Using Sionna version: {sionna.__version__}")
    except ImportError:
        pass # Sionna import error will be caught later

# Call the version check early
check_tf_sionna_versions()

# --- Sionna Imports with Error Handling ---
try:
    import sionna # General import to get version if needed
    from sionna.phy.channel.tr38901 import UMa
    from sionna.phy.antenna import PanelArray # Using PanelArray for BS
    # PatchArray was in the original plan for UE, but UMa handles UT antenna implicitly.
    # If direct PatchArray instantiation is needed, it would be here.
    # from sionna.phy.mimo import StreamManagement, lmmse_equalizer # Not used in current simplified beamforming
    from sionna.phy.utils import log10 # Only log10 is used from this list in the current code
    # from sionna.phy.utils import compute_gain, ebnodb_to_no, expand_to_rank, complex_normal # If needed later
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import Sionna components: {e}")
    print("Please ensure Sionna is installed correctly and accessible in your Python environment.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during Sionna imports: {e}")
    sys.exit(1)


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Define the mmWave ISAC Environment as a Gym environment ---
class MmWaveISACEnv(gym.Env):
    """
    Custom Gym environment for mmWave ISAC secure beamforming.

    State Space:
    [SINR_User (dB),
     Current_Beam_Steering_Angle_Azimuth (radians), Current_Beam_Steering_Angle_Elevation (radians),
     ISAC_Detected_Attacker_DoA_Azimuth (radians, or special value if none),
     ISAC_Detected_Attacker_DoA_Elevation (radians, or special value if none),
     ISAC_Detected_Attacker_Range (meters, or special value if none),
     ISAC_Detection_Confidence (0-1)]

    Action Space (Discrete):
     0: Steer beam azimuth Left
     1: Steer beam azimuth Right
     2: Steer beam elevation Up
     3: Steer beam elevation Down
     4: Maintain current beam
     5: ISAC High Effort Probing
     6: ISAC Low Effort Probing
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(MmWaveISACEnv, self).__init__()

        # Simulation parameters
        self.carrier_frequency = 28e9  # 28 GHz mmWave
        self.bandwidth = 100e6 # 100 MHz bandwidth
        self.num_bs_antennas_sqrt_rows = 8
        self.num_bs_antennas_sqrt_cols = 8
        self.num_bs_antennas = self.num_bs_antennas_sqrt_rows * self.num_bs_antennas_sqrt_cols
        
        # BS Antenna Array Configuration using sionna.phy.antenna
        try:
            self.bs_array = PanelArray(
                num_rows_per_panel=self.num_bs_antennas_sqrt_rows,
                num_cols_per_panel=self.num_bs_antennas_sqrt_cols,
                polarization="dual",
                polarization_type="cross",
                antenna_pattern="38.901", # 3GPP TR38.901 antenna pattern
                carrier_frequency=self.carrier_frequency
            )
            # For UE, UMa model can use simpler omni or patch antenna models implicitly.
            # If you need to define it explicitly for other purposes:
            # self.ue_array = PatchArray(
            #     polarization="dual",
            #     antenna_pattern="omni",
            #     carrier_frequency=self.carrier_frequency
            # )
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Sionna antenna arrays: {e}")
            sys.exit(1)

        self.num_user = 1
        self.num_attacker = 1
        
        self.tx_power_dbm = 20.0
        self.tx_power_watts = 10**((self.tx_power_dbm - 30) / 10)
        self.noise_power_db_per_hz = -174.0
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth)
        self.no_lin = 10**((no_db - 30) / 10)

        self.sensing_range_max = 150.0
        self.max_steps_per_episode = 100

        # Sionna channel model (3GPP UMa from TR38.901)
        try:
            self.channel_model = UMa(
                carrier_frequency=self.carrier_frequency,
                bandwidth=self.bandwidth,
                bs_array_config=self.bs_array, # Pass the instantiated PanelArray
                ut_array_config="omni", # Use a predefined omni for UE simplicity in UMa
                domain="time",
                los_probability_model="R1-20_LoS",
                delay_spread_model="R1-20_RMS_DS",
                enable_pathloss=True, # Ensure pathloss is part of the model
                enable_shadow_fading=True
            )
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Sionna UMa channel model: {e}")
            sys.exit(1)
        
        # State space definition (tensor shape: (7,))
        low_obs = np.array([-30.0, -np.pi, -np.pi/2, -np.pi-0.1, -np.pi/2-0.1, -1.0, 0.0], dtype=np.float32)
        high_obs = np.array([30.0, np.pi, np.pi/2, np.pi, np.pi/2, self.sensing_range_max, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Action space definition
        self.num_discrete_actions = 7
        self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.beam_angle_delta_rad = np.deg2rad(5) # 5 degrees adjustment step

        # Positions [batch_size, num_devices, 3] or [num_devices, 3] then expand_dims
        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32) # BS at 10m height
        self.user_position_init = np.array([[50.0, 10.0, 1.5]], dtype=np.float32)
        self.attacker_position_init = np.array([[60.0, -15.0, 1.5]], dtype=np.float32)

        self.user_position = tf.Variable(self.user_position_init, dtype=tf.float32)
        self.attacker_position = tf.Variable(self.attacker_position_init, dtype=tf.float32)

        self.current_beam_angles_tf = tf.Variable([0.0, 0.0], dtype=tf.float32) # [Azimuth, Elevation] in radians
        self.current_isac_effort = 0.5 # Normalized 0 (low) to 1 (high)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_beam_angles_tf.assign([0.0, 0.0])
        self.current_isac_effort = 0.5
        self.user_position.assign(self.user_position_init + tf.random.normal(shape=[1,3], mean=0.0, stddev=1.0, seed=self.np_random.integers(0,1000)))
        self.attacker_position.assign(self.attacker_position_init + tf.random.normal(shape=[1,3], mean=0.0, stddev=2.0, seed=self.np_random.integers(0,1000)))
        self.current_step = 0
        return self._get_state(), {}

    def _get_steering_vector(self, angles_rad_tf):
        """Computes the steering vector for the BS array."""
        # angles_rad_tf: TensorFlow tensor [azimuth_rad, elevation_rad]
        azimuth_rad = tf.constant(angles_rad_tf[0], shape=(1,)) # Shape [1] for batch_size=1
        zenith_rad = tf.constant(np.pi/2 - angles_rad_tf[1], shape=(1,)) # Shape [1]

        # Output shape of steering_vector: [batch_size, num_ant, num_streams_per_obj]
        # Assuming num_streams_per_obj = 1 for this BS to UE/Attacker link
        sv = self.bs_array.steering_vector(azimuth_rad, zenith_rad) # sv shape [1, num_bs_antennas, 1]
        return tf.cast(tf.squeeze(sv, axis=0), tf.complex64) # Squeeze batch, result shape [num_bs_antennas, 1]

    @tf.function(reduce_retracing=True) # Decorate for performance
    def _get_channel_and_precoder_outputs(self, current_beam_angles_tf_in, user_pos_in, attacker_pos_in):
        # This function can be compiled by tf.function for speed if inputs are tensors
        # For UMa model, positions are (location, orientation, velocity) tuples
        # Using simplified static orientations and zero velocities for now
        bs_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        bs_velocity = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        ut_orientation = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        ut_velocity = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)

        bs_config = (self.bs_position, bs_orientation, bs_velocity)
        ut_config_user = (user_pos_in, ut_orientation, ut_velocity)
        ut_config_attacker = (attacker_pos_in, ut_orientation, ut_velocity)

        h_user_time, _ = self.channel_model(bs_config, ut_config_user)
        # h_user_time shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # Taking the channel from the first path, first time step for simplicity.
        # A more realistic model would use OFDM and frequency-domain channels.
        h_user = h_user_time[0, 0, :, 0, :, 0, 0] # Shape: [num_ue_antennas, num_bs_antennas]
        
        h_attacker_time, _ = self.channel_model(bs_config, ut_config_attacker)
        h_attacker = h_attacker_time[0, 0, :, 0, :, 0, 0] # Shape: [num_ue_antennas, num_bs_antennas]

        steering_vec = self._get_steering_vector(current_beam_angles_tf_in) # Shape [num_bs_antennas, 1]
        # Normalized precoder (conjugate for MRT)
        precoder_w = tf.math.conj(steering_vec) / (tf.norm(steering_vec) + 1e-9) # Add epsilon for stability
        precoder_w_expanded = tf.expand_dims(precoder_w, axis=0) # Shape [1, num_bs_antennas, 1]

        # Effective channel after precoding (y = H * w)
        # H shapes: [num_ue_ant, num_bs_ant], w shape: [num_bs_ant, 1]
        y_user_eff = tf.matmul(tf.cast(h_user, tf.complex64), precoder_w) # Shape [num_ue_ant, 1]
        y_attacker_eff = tf.matmul(tf.cast(h_attacker, tf.complex64), precoder_w) # Shape [num_ue_ant, 1]
        
        signal_power_user = tf.reduce_sum(tf.abs(y_user_eff)**2) * self.tx_power_watts
        signal_power_attacker = tf.reduce_sum(tf.abs(y_attacker_eff)**2) * self.tx_power_watts
        
        return signal_power_user, signal_power_attacker

    def _get_state(self):
        signal_power_user_tf, signal_power_attacker_tf = self._get_channel_and_precoder_outputs(
            self.current_beam_angles_tf, self.user_position, self.attacker_position
        )
        signal_power_user = signal_power_user_tf.numpy()
        # signal_power_attacker = signal_power_attacker_tf.numpy() # Not directly used in state for now

        sinr_user = 10 * log10(signal_power_user / self.no_lin) if signal_power_user > 1e-12 else -30.0 # Use Sionna's log10
        sinr_user = tf.clip_by_value(sinr_user, -30.0, 30.0).numpy()


        # --- ISAC Sensing Part ---
        sensing_noise_std_factor = 1.0 - (self.current_isac_effort * 0.75) # Max 75% noise reduction
        
        bs_pos_np = self.bs_position.numpy()[0] # Shape (3,)
        attacker_pos_np = self.attacker_position.numpy()[0] # Shape (3,)

        true_attacker_vector = attacker_pos_np - bs_pos_np
        true_attacker_range = np.linalg.norm(true_attacker_vector)
        true_attacker_azimuth = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])
        true_attacker_elevation = np.arctan2(true_attacker_vector[2], np.linalg.norm(true_attacker_vector[:2])) # Elevation from xy-plane

        detected_az, detected_el, detected_range, confidence = -np.pi-0.1, -np.pi/2-0.1, -1.0, 0.0
        
        if true_attacker_range <= self.sensing_range_max:
            prob_detection = self.current_isac_effort * np.exp(-0.01 * true_attacker_range) # Exponential decay by range
            if self.np_random.random() < prob_detection:
                confidence = prob_detection + self.np_random.normal(0, 0.1) # Add some noise to confidence too
                confidence = np.clip(confidence, 0.0, 1.0)

                noise_az = self.np_random.normal(0, np.deg2rad(15) * sensing_noise_std_factor)
                noise_el = self.np_random.normal(0, np.deg2rad(15) * sensing_noise_std_factor)
                noise_range = self.np_random.normal(0, 7.0 * sensing_noise_std_factor)

                detected_az = true_attacker_azimuth + noise_az
                detected_el = true_attacker_elevation + noise_el
                detected_range = max(0, true_attacker_range + noise_range)
        
        state_array = np.array([
            sinr_user,
            self.current_beam_angles_tf[0].numpy(), self.current_beam_angles_tf[1].numpy(),
            np.clip(detected_az, -np.pi, np.pi), 
            np.clip(detected_el, -np.pi/2, np.pi/2),
            np.clip(detected_range, 0, self.sensing_range_max),
            confidence
        ], dtype=np.float32)
        return state_array

    def step(self, action_idx):
        """Applies an action and returns the new state, reward, done, truncated, info."""
        current_az_val, current_el_val = self.current_beam_angles_tf.numpy()

        if action_idx == 0: current_az_val -= self.beam_angle_delta_rad
        elif action_idx == 1: current_az_val += self.beam_angle_delta_rad
        elif action_idx == 2: current_el_val += self.beam_angle_delta_rad
        elif action_idx == 3: current_el_val -= self.beam_angle_delta_rad
        elif action_idx == 4: pass # Maintain beam
        elif action_idx == 5: self.current_isac_effort = 1.0 # High effort
        elif action_idx == 6: self.current_isac_effort = 0.2 # Low effort
        
        current_az_val = np.clip(current_az_val, -np.pi, np.pi)
        current_el_val = np.clip(current_el_val, -np.pi/2 * 0.95, np.pi/2 * 0.95)
        self.current_beam_angles_tf.assign([current_az_val, current_el_val])

        self.user_position.assign_add(tf.random.normal(shape=[1,3], mean=0.0, stddev=0.05, dtype=tf.float32, seed=self.np_random.integers(0,1000))) # Less user movement
        self.attacker_position.assign_add(tf.random.normal(shape=[1,3], mean=0.0, stddev=0.2, dtype=tf.float32, seed=self.np_random.integers(0,1000)))

        next_state_obs = self._get_state()
        reward = self._compute_reward(next_state_obs)
        
        self.current_step += 1
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps_per_episode:
            truncated = True
        
        if self._is_beam_stolen(next_state_obs):
            terminated = True
            reward -= 200 # Large penalty for being stolen

        if next_state_obs[0] < -20.0: # Very low SINR might also terminate
            terminated = True
            reward -= 50
            
        return next_state_obs, reward, terminated, truncated, {}

    def _is_beam_stolen(self, current_obs_state):
        """Checks if the beam is considered stolen based on current observation."""
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        # beam_el_rad = current_obs_state[2] # Not using elevation for this simple check
        detected_attacker_az_rad = current_obs_state[3]
        # detected_attacker_el_rad = current_obs_state[4]
        detected_attacker_range_m = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        if detection_confidence > 0.6 and 0 < detected_attacker_range_m < 30: # Attacker confidently detected and very close
            angle_diff_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_az = min(angle_diff_az, 2*np.pi - angle_diff_az) # Normalize to [0, pi]
            
            # If beam is within ~15 degrees of detected close attacker AND user SINR is bad
            if angle_diff_az < np.deg2rad(20) and sinr_user < -5.0:
                print(f"--- BEAM STOLEN condition met: AngleDiff={np.rad2deg(angle_diff_az):.1f}deg, UserSINR={sinr_user:.1f}dB, AttRange={detected_attacker_range_m:.1f}m ---")
                return True
        return False

    def _compute_reward(self, current_obs_state):
        """Computes the reward based on the current state."""
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        # beam_el_rad = current_obs_state[2]
        detected_attacker_az_rad = current_obs_state[3]
        # detected_attacker_el_rad = current_obs_state[4]
        detected_attacker_range_m = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        # 1. Reward for User SINR (Primary objective)
        reward = tf.clip_by_value(sinr_user / 5.0, -3.0, 3.0).numpy() # Scaled and clipped

        # 2. ISAC-related rewards/penalties
        if detection_confidence > 0.3 and detected_attacker_range_m > 0: # If attacker is somewhat detected
            # Reward for accurate DoA estimation (if detected)
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_vector = attacker_pos_np - bs_pos_np
            true_attacker_az_rad = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])
            
            doa_error_az = abs(detected_attacker_az_rad - true_attacker_az_rad)
            doa_error_az = min(doa_error_az, 2*np.pi - doa_error_az)
            reward += (1.0 - doa_error_az / np.pi) * 1.0 * detection_confidence # Max +1 for perfect DoA

            # Penalty for beam pointing towards a confidently detected attacker
            angle_diff_beam_to_detected_attacker_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_beam_to_detected_attacker_az = min(angle_diff_beam_to_detected_attacker_az, 2*np.pi - angle_diff_beam_to_detected_attacker_az)
            reward -= (1.0 - angle_diff_beam_to_detected_attacker_az / np.pi) * 3.0 * detection_confidence # Max penalty -3
        else: # Attacker not confidently detected
            # If attacker is close but not detected, penalize (encourages better ISAC effort)
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_range_val = np.linalg.norm(attacker_pos_np - bs_pos_np)
            if true_attacker_range_val < self.sensing_range_max * 0.5: # If attacker is actually close
                reward -= 0.5 # Penalty for not detecting a present, close attacker

        # 3. Cost for ISAC effort (to prevent constant high effort if not needed)
        if self.current_isac_effort > 0.8: reward -= 0.2
        elif self.current_isac_effort < 0.3: reward -= 0.05 # Small penalty for very low effort if it leads to missed detections

        return float(reward)


    def render(self, mode='human'):
        if mode == 'human':
            print(f"  Step: {self.current_step}, ISAC Effort: {self.current_isac_effort:.2f}")
            state = self._get_state() # Get latest state for rendering
            print(f"  Beam Angles (Az,El): ({np.rad2deg(state[1]):.1f}°, {np.rad2deg(state[2]):.1f}°)")
            print(f"  User SINR: {state[0]:.2f} dB")
            if state[6] > 0.1:
                print(f"  Detected Attacker: Az={np.rad2deg(state[3]):.1f}°, El={np.rad2deg(state[4]):.1f}°, "
                      f"Range={state[5]:.1f}m (Conf: {state[6]:.2f})")
            else:
                print("  Attacker Not Detected or Low Confidence.")
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_range_val = np.linalg.norm(attacker_pos_np - bs_pos_np)
            print(f"  True Attacker Range: {true_attacker_range_val:.1f}m")


    def close(self):
        print("Environment closing.")
        pass

# --- DRL Agent (DQN) ---
class DQNAgent:
    def __init__(self, state_dim, action_n, learning_rate=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        self.state_dim = state_dim
        self.action_n = action_n
        self.memory = deque(maxlen=20000) # Replay buffer
        self.gamma = gamma    # Discount factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_value = (epsilon_start - epsilon_end) / epsilon_decay_steps # Linear decay for steps
        self.current_learning_steps = 0

        self.learning_rate = learning_rate
        self.batch_size = 64

        self.model = self._build_model()
        self.target_model = self. _build_model()
        self.update_target_model() # Initialize target model

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(256, activation='relu'), # Increased layer size
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_n, activation='linear')
        ])
        model.compile(loss='huber_loss',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        self.current_learning_steps +=1
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_n)
        
        if state.ndim == 1: state_for_pred = np.expand_dims(state, axis=0)
        else: state_for_pred = state
        
        act_values = self.model.predict(state_for_pred, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0 # No training done, return 0 loss

        minibatch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        q_values_next_target = self.target_model.predict(next_states, verbose=0)
        
        targets = rewards + self.gamma * np.amax(q_values_next_target, axis=1) * (1 - dones)
        
        # Create target Q-values for training: only update Q for action taken
        target_q_values = self.model.predict(states, verbose=0) # Get current Q-values
        for i in range(self.batch_size):
            target_q_values[i, actions[i]] = targets[i] # Set the new Q-value for the action taken

        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        # Linear epsilon decay based on steps
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_value
        
        return history.history['loss'][0]


    def load(self, name):
        try:
            self.model.load_weights(name)
            self.update_target_model()
            print(f"Model weights loaded successfully from {name}")
        except Exception as e:
            print(f"Error loading model weights from {name}: {e}")


    def save(self, name):
        try:
            self.model.save_weights(name)
            print(f"Model weights saved successfully to {name}")
        except Exception as e:
            print(f"Error saving model weights to {name}: {e}")


# --- Main simulation function ---
def run_simulation():
    print("Starting DRL Simulation for Secure mmWave ISAC Beamforming...")
    env = MmWaveISACEnv()
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DQNAgent(state_dim, action_n, epsilon_decay_steps=20000) # Decay epsilon over more steps

    episodes = 500
    target_update_freq_steps = 1000 # Update target model every X training steps (replays)
    print_freq_episodes = 10
    save_freq_episodes = 100
    total_steps_for_target_update = 0

    episode_rewards = []
    avg_losses = []

    for e in range(episodes):
        current_state, _ = env.reset()
        total_episode_reward = 0
        episode_loss_sum = 0
        num_replays_in_episode = 0

        for step in range(env.max_steps_per_episode):
            action_idx = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            agent.remember(current_state, action_idx, reward, next_state, terminated)
            current_state = next_state
            total_episode_reward += reward

            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                episode_loss_sum += loss
                num_replays_in_episode +=1
                total_steps_for_target_update +=1

            if total_steps_for_target_update >= target_update_freq_steps:
                agent.update_target_model()
                total_steps_for_target_update = 0
                print(f"Target model updated at episode {e+1}, step {step+1}.")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_episode_reward)
        if num_replays_in_episode > 0:
            avg_losses.append(episode_loss_sum / num_replays_in_episode)
        else:
            avg_losses.append(0)


        if (e + 1) % print_freq_episodes == 0:
            avg_r = np.mean(episode_rewards[-(print_freq_episodes):])
            avg_l = np.mean(avg_losses[-(print_freq_episodes):])
            print(f"Ep: {e+1}/{episodes}| Avg Reward (last {print_freq_episodes}): {avg_r:.2f}| Last Ep Reward: {total_episode_reward:.2f}| Avg Loss: {avg_l:.4f}| Epsilon: {agent.epsilon:.3f}| Steps: {step+1}")
            # env.render() # Render last state of an episode if needed

        if (e + 1) % save_freq_episodes == 0:
             agent.save(f"drl_beamsecure_sionna_ep{e+1}.weights.h5")
    
    env.close()
    print("Training finished.")

    # Basic plotting of rewards
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward per Episode')
        plt.savefig('episode_rewards.png')
        print("Saved episode_rewards.png")
        # plt.show() # Uncomment to display plot
    except ImportError:
        print("Matplotlib not found. Skipping reward plot generation.")


if __name__ == "__main__":
    # GPU Configuration: Enable memory growth for GPUs
    # This attempts to use only one GPU if multiple are available by default behavior of TF,
    # but sets memory growth for all visible ones.
    # To force a specific GPU, use CUDA_VISIBLE_DEVICES environment variable (e.g., "0" or "1").
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} Physical GPU(s):")
        try:
            # Set memory growth for each GPU
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  GPU {i}: {gpu.name} - Memory growth enabled.")
            
            # If you want to strictly use only the first GPU:
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            # print(f"Strictly using GPU 0: {gpus[0].name}")
            # However, usually setting CUDA_VISIBLE_DEVICES is preferred for this.
            # The default TF behavior with Keras usually picks one GPU for model.fit
            # if no distribution strategy is defined.

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow sees {len(logical_gpus)} Logical GPU(s).")

        except RuntimeError as e:
            print(f"RuntimeError during GPU setup: {e}")
    else:
        print("No GPU detected by TensorFlow. Model will run on CPU.")

    run_simulation()
