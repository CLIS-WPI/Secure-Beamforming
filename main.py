import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import tensorflow.keras as keras # Using tf.keras
from collections import deque
import random
import sys # For sys.exit

# --- Helper function to check TensorFlow version ---
def check_tf_sionna_versions():
    """Checks if the TensorFlow version is compatible with Sionna."""
    print(f"Using TensorFlow version: {tf.__version__}")
    tf_major_minor = float('.'.join(tf.__version__.split('.')[:2]))
    if not (tf_major_minor >= 2.14): # Sionna generally requires TF 2.14+
        print(f"Warning: TensorFlow version {tf.__version__} might be older than typically expected for latest Sionna features.")
    try:
        import sionna
        print(f"Using Sionna version: {sionna.__version__}")
    except ImportError:
        pass

check_tf_sionna_versions()

# --- Sionna Imports with Error Handling ---
try:
    # Corrected import path for UMa and PanelArray
    from sionna.phy.channel.tr38901 import UMa
    from sionna.phy.channel.tr38901.antenna import PanelArray
    from sionna.phy.utils import log10
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
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    # Inside class MmWaveISACEnv:

    def __init__(self):
        super(MmWaveISACEnv, self).__init__()

        self.carrier_frequency = 28e9
        self.bandwidth = 100e6 
        self.num_bs_antennas_sqrt_rows = 8
        self.num_bs_antennas_sqrt_cols = 8
        self.num_bs_antennas = self.num_bs_antennas_sqrt_rows * self.num_bs_antennas_sqrt_cols
        
        # 1. Initialize Antenna Arrays
        try:
            self.bs_array = PanelArray( 
                num_rows_per_panel=self.num_bs_antennas_sqrt_rows,
                num_cols_per_panel=self.num_bs_antennas_sqrt_cols,
                polarization="dual",
                polarization_type="cross",
                antenna_pattern="38.901",
                carrier_frequency=self.carrier_frequency
            )
            print("Sionna PanelArray for BS initialized successfully.") 
            self.ut_array = PanelArray(
                num_rows_per_panel=1,
                num_cols_per_panel=1,
                polarization='single',
                polarization_type='V',
                antenna_pattern='omni',
                carrier_frequency=self.carrier_frequency
            )
            print("Sionna PanelArray for UT (omni) initialized successfully.")
        except Exception as e: 
            print(f"CRITICAL ERROR: Failed to initialize Sionna PanelArray objects: {e}")
            sys.exit(1)

        # 2. Define other necessary parameters, including initial positions
        self.num_user = 1 
        self.num_bs = 1   
        self.tx_power_dbm = 20.0
        self.tx_power_watts = 10**((self.tx_power_dbm - 30) / 10)
        self.noise_power_db_per_hz = -174.0
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth) 
        self.no_lin = 10**((no_db - 30) / 10)
        self.sensing_range_max = 150.0
        self.max_steps_per_episode = 100
        
        # Define initial positions BEFORE they are used in set_topology or UMa
        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32)
        self.user_position_init = np.array([[50.0, 10.0, 1.5]], dtype=np.float32) # Defined here
        self.attacker_position_init = np.array([[60.0, -15.0, 1.5]], dtype=np.float32) # Defined here

        # 3. Initialize UMa channel model
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
            print("Sionna UMa channel model object created successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Sionna UMa channel model object: {e}")
            sys.exit(1)

        # 4. Set initial topology
        try:
            initial_ut_loc = tf.reshape(self.user_position_init, [1, self.num_user, 3]) # Now self.user_position_init exists
            
            if self.bs_position.shape == (1,3) and self.num_bs == 1:
                initial_bs_loc_for_set_topology = tf.expand_dims(self.bs_position, axis=1) 
            else:
                # Assuming self.bs_position might already be [1,1,3] or other logic needed
                initial_bs_loc_for_set_topology = tf.reshape(self.bs_position, [1, self.num_bs, 3])


            initial_ut_orientations = tf.zeros([1, self.num_user, 3], dtype=tf.float32)
            initial_bs_orientations = tf.zeros([1, self.num_bs, 3], dtype=tf.float32)
            initial_ut_velocities = tf.zeros([1, self.num_user, 3], dtype=tf.float32)
            initial_in_state = tf.zeros([1, self.num_user], dtype=tf.bool)

            self.channel_model_core.set_topology(
                ut_loc=initial_ut_loc,
                bs_loc=initial_bs_loc_for_set_topology,
                ut_orientations=initial_ut_orientations,
                bs_orientations=initial_bs_orientations,
                ut_velocities=initial_ut_velocities,
                in_state=initial_in_state
            )
            print("Sionna UMa channel model topology set successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to set topology for Sionna UMa channel model: {e}")
            sys.exit(1)
        
        # 5. Define State and Action Spaces and other instance variables
        low_obs = np.array([-30.0, -np.pi, -np.pi/2, -np.pi-0.1, -np.pi/2-0.1, -1.0, 0.0], dtype=np.float32)
        high_obs = np.array([30.0, np.pi, np.pi/2, np.pi, np.pi/2, self.sensing_range_max, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.num_discrete_actions = 7
        self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.beam_angle_delta_rad = np.deg2rad(5)

        # These are now tf.Variables, their initial value is set from *_init arrays
        self.user_position = tf.Variable(self.user_position_init, dtype=tf.float32)
        self.attacker_position = tf.Variable(self.attacker_position_init, dtype=tf.float32)

        self.current_beam_angles_tf = tf.Variable([0.0, 0.0], dtype=tf.float32)
        self.current_isac_effort = 0.5
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_beam_angles_tf.assign([0.0, 0.0])
        self.current_isac_effort = 0.5
        user_pos_offset = self.np_random.normal(0, 1.0, 3).reshape(1,3).astype(np.float32)
        attacker_pos_offset = self.np_random.normal(0, 2.0, 3).reshape(1,3).astype(np.float32)
        self.user_position.assign(self.user_position_init + user_pos_offset)
        self.attacker_position.assign(self.attacker_position_init + attacker_pos_offset)
        self.current_step = 0
        return self._get_state(), {}

    def _get_steering_vector(self, angles_rad_tf):
        azimuth_rad = tf.constant(angles_rad_tf[0], shape=(1,))
        zenith_rad = tf.constant(np.pi/2 - angles_rad_tf[1], shape=(1,))
        sv = self.bs_array.steering_vector(azimuth_rad, zenith_rad)
        return tf.cast(tf.squeeze(sv, axis=[0,2]), tf.complex64) # Squeeze batch and stream dim

    #@tf.function(reduce_retracing=True) # Keep commented out for now to ensure Python-level correctness first
    def _get_channel_and_powers(self, current_beam_angles_tf_in, user_pos_in, attacker_pos_in):
        current_batch_size = tf.shape(user_pos_in)[0] 

        bs_loc_reshaped = tf.reshape(self.bs_position, [current_batch_size, 1, 3])
        user_loc_reshaped = tf.reshape(user_pos_in, [current_batch_size, 1, 3])
        attacker_loc_reshaped = tf.reshape(attacker_pos_in, [current_batch_size, 1, 3])

        bs_orientation = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)
        bs_velocity = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)
        
        ut_orientation_user = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)
        ut_velocity_user = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)
        
        ut_orientation_attacker = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)
        ut_velocity_attacker = tf.zeros([current_batch_size, 1, 3], dtype=tf.float32)

        # These are the variable names already defined in your function scope
        bs_config = (bs_loc_reshaped, bs_orientation, bs_velocity)
        ut_config_user = (user_loc_reshaped, ut_orientation_user, ut_velocity_user)
        ut_config_attacker = (attacker_loc_reshaped, ut_orientation_attacker, ut_velocity_attacker)

        num_time_samples_val = 1 
        sampling_frequency_val = tf.cast(self.bandwidth, dtype=tf.float32) 

        # --- CORRECTED CALLS USING YOUR DEFINED VARIABLE NAMES ---
        # Pass bs_config and ut_config POSITIALLY, then OTHERS BY KEYWORD
        h_user_time_all_paths, _ = self.channel_model_core(
            bs_config,            # 1st Positional argument 
            ut_config_user,       # 2nd Positional argument
            num_time_samples=num_time_samples_val,       # Keyword argument
            sampling_frequency=sampling_frequency_val      # Keyword argument
        )
        
        h_user = h_user_time_all_paths[0, 0, :, 0, :, 0, 0]

        h_attacker_time_all_paths, _ = self.channel_model_core(
            bs_config,            # 1st Positional argument
            ut_config_attacker,   # 2nd Positional argument
            num_time_samples=num_time_samples_val,       # Keyword argument
            sampling_frequency=sampling_frequency_val      # Keyword argument
        )
        h_attacker = h_attacker_time_all_paths[0, 0, :, 0, :, 0, 0]
        # --- END OF CORRECTED CALLS ---

        steering_vec = self._get_steering_vector(current_beam_angles_tf_in)
        precoder_w = tf.math.conj(steering_vec) / (tf.norm(steering_vec) + 1e-9)
        precoder_w = tf.expand_dims(precoder_w, axis=1)

        h_user_effective_row = tf.cast(h_user[0,:], tf.complex64) 
        h_attacker_effective_row = tf.cast(h_attacker[0,:], tf.complex64)

        y_user_eff_scalar = tf.reduce_sum(h_user_effective_row * tf.squeeze(precoder_w))
        y_attacker_eff_scalar = tf.reduce_sum(h_attacker_effective_row * tf.squeeze(precoder_w))
        
        signal_power_user = tf.abs(y_user_eff_scalar)**2 * self.tx_power_watts
        signal_power_attacker = tf.abs(y_attacker_eff_scalar)**2 * self.tx_power_watts
        
        return signal_power_user, signal_power_attacker

    def _get_state(self):
        signal_power_user_tf, _ = self._get_channel_and_powers(
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
        true_attacker_elevation = np.arctan2(true_attacker_vector[2], np.linalg.norm(true_attacker_vector[0:2]))

        detected_az, detected_el, detected_range, confidence = -np.pi-0.1, -np.pi/2-0.1, -1.0, 0.0
        
        if true_attacker_range <= self.sensing_range_max and true_attacker_range > 0:
            prob_detection = self.current_isac_effort * np.exp(-0.02 * true_attacker_range)
            if self.np_random.random() < prob_detection:
                confidence = prob_detection + self.np_random.normal(0, 0.1)
                confidence = np.clip(confidence, 0.0, 1.0)
                noise_az = self.np_random.normal(0, np.deg2rad(20) * sensing_noise_std_factor)
                noise_el = self.np_random.normal(0, np.deg2rad(20) * sensing_noise_std_factor)
                noise_range = self.np_random.normal(0, 10.0 * sensing_noise_std_factor)
                detected_az = true_attacker_azimuth + noise_az
                detected_el = true_attacker_elevation + noise_el
                detected_range = max(0.1, true_attacker_range + noise_range)
        
        state_array = np.array([
            sinr_user_clipped,
            self.current_beam_angles_tf[0].numpy(), self.current_beam_angles_tf[1].numpy(),
            np.clip(detected_az, -np.pi, np.pi), 
            np.clip(detected_el, -np.pi/2, np.pi/2),
            np.clip(detected_range, 0, self.sensing_range_max),
            confidence
        ], dtype=np.float32)
        return state_array

    def step(self, action_idx):
        current_az_val, current_el_val = self.current_beam_angles_tf.numpy()

        if action_idx == 0: current_az_val -= self.beam_angle_delta_rad
        elif action_idx == 1: current_az_val += self.beam_angle_delta_rad
        elif action_idx == 2: current_el_val += self.beam_angle_delta_rad
        elif action_idx == 3: current_el_val -= self.beam_angle_delta_rad
        elif action_idx == 4: pass
        elif action_idx == 5: self.current_isac_effort = min(1.0, self.current_isac_effort + 0.4)
        elif action_idx == 6: self.current_isac_effort = max(0.1, self.current_isac_effort - 0.4)
        
        self.current_isac_effort = np.clip(self.current_isac_effort, 0.1, 1.0)
        current_az_val = np.clip(current_az_val, -np.pi, np.pi)
        current_el_val = np.clip(current_el_val, -np.pi/2 * 0.95, np.pi/2 * 0.95)
        self.current_beam_angles_tf.assign([current_az_val, current_el_val])

        user_pos_offset = self.np_random.normal(0, 0.05, 3).astype(np.float32).reshape(1,3)
        attacker_pos_offset = self.np_random.normal(0, 0.2, 3).astype(np.float32).reshape(1,3)
        self.user_position.assign_add(tf.constant(user_pos_offset, dtype=tf.float32))
        self.attacker_position.assign_add(tf.constant(attacker_pos_offset, dtype=tf.float32))

        next_state_obs = self._get_state()
        reward = self._compute_reward(next_state_obs)
        
        self.current_step += 1
        terminated = False
        truncated = False

        if self.current_step >= self.max_steps_per_episode:
            truncated = True
        
        if self._is_beam_stolen(next_state_obs):
            terminated = True
            reward -= 200 

        if next_state_obs[0] < -20.0: # Persistently very low SINR
            if not terminated : reward -= 50 # Add penalty only if not already terminated by beam stealing
            terminated = True
            
        return next_state_obs, reward, terminated, truncated, {}

    def _is_beam_stolen(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_az_rad = current_obs_state[3]
        detected_attacker_range_m = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        if detection_confidence > 0.6 and 0 < detected_attacker_range_m < 30:
            angle_diff_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_az = min(angle_diff_az, 2*np.pi - angle_diff_az)
            if angle_diff_az < np.deg2rad(20) and sinr_user < -5.0:
                print(f"--- BEAM STOLEN condition met: AngleDiff={np.rad2deg(angle_diff_az):.1f}deg, UserSINR={sinr_user:.1f}dB, AttRange={detected_attacker_range_m:.1f}m ---")
                return True
        return False

    def _compute_reward(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az_rad = current_obs_state[1]
        detected_attacker_az_rad = current_obs_state[3]
        detected_attacker_range_m = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        reward = tf.clip_by_value(sinr_user / 5.0, -3.0, 3.0).numpy()

        if detection_confidence > 0.3 and detected_attacker_range_m > 0:
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_vector = attacker_pos_np - bs_pos_np
            true_attacker_az_rad = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])
            doa_error_az = abs(detected_attacker_az_rad - true_attacker_az_rad)
            doa_error_az = min(doa_error_az, 2*np.pi - doa_error_az)
            reward += (1.0 - doa_error_az / np.pi) * 1.0 * detection_confidence
            angle_diff_beam_to_detected_attacker_az = abs(beam_az_rad - detected_attacker_az_rad)
            angle_diff_beam_to_detected_attacker_az = min(angle_diff_beam_to_detected_attacker_az, 2*np.pi - angle_diff_beam_to_detected_attacker_az)
            reward -= (1.0 - angle_diff_beam_to_detected_attacker_az / np.pi) * 3.0 * detection_confidence
        else:
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_range_val = np.linalg.norm(attacker_pos_np - bs_pos_np)
            if true_attacker_range_val < self.sensing_range_max * 0.5:
                reward -= 0.5

        if self.current_isac_effort > 0.8: reward -= 0.2
        elif self.current_isac_effort < 0.3: reward -= 0.05
        return float(reward)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"  Step: {self.current_step}, ISAC Effort: {self.current_isac_effort:.2f}")
            # state = self._get_state() # _get_state is already called before render implicitly if needed
            # For direct state from last step if available through an internal var or re-call:
            current_render_state = self._get_state() # Or pass state to render if it's available
            print(f"  Beam Angles (Az,El): ({np.rad2deg(current_render_state[1]):.1f}°, {np.rad2deg(current_render_state[2]):.1f}°)")
            print(f"  User SINR: {current_render_state[0]:.2f} dB")
            if current_render_state[6] > 0.1:
                print(f"  Detected Attacker: Az={np.rad2deg(current_render_state[3]):.1f}°, El={np.rad2deg(current_render_state[4]):.1f}°, "
                      f"Range={current_render_state[5]:.1f}m (Conf: {current_render_state[6]:.2f})")
            else:
                print("  Attacker Not Detected or Low Confidence.")
            bs_pos_np = self.bs_position.numpy()[0]
            attacker_pos_np = self.attacker_position.numpy()[0]
            true_attacker_range_val = np.linalg.norm(attacker_pos_np - bs_pos_np)
            print(f"  True Attacker Range: {true_attacker_range_val:.1f}m")

    def close(self):
        # print("Environment closing.") # Optional: can be removed if not adding specific cleanup
        pass

# --- DRL Agent (DQN) ---
class DQNAgent:
    def __init__(self, state_dim, action_n, learning_rate=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000):
        self.state_dim = state_dim
        self.action_n = action_n
        self.memory = deque(maxlen=20000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_value = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.current_learning_steps = 0
        self.learning_rate = learning_rate
        self.batch_size = 64

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(256, activation='relu'),
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
        self.current_learning_steps +=1 # Ensure this is called for epsilon decay logic
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_n)
        
        if state.ndim == 1: state_for_pred = np.expand_dims(state, axis=0)
        else: state_for_pred = state # Should already be [batch_size, state_dim] if coming from batch
        
        act_values = self.model.predict(state_for_pred, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        minibatch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch]) # These are action indices
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        q_values_next_state_target_model = self.target_model.predict(next_states, verbose=0)
        
        # Bellman equation: target = r + gamma * max_a'(Q_target(s', a'))
        targets_for_taken_actions = rewards + self.gamma * np.amax(q_values_next_state_target_model, axis=1) * (1 - dones)
        
        # Get current Q-values from the main model for all actions
        current_q_values_all_actions = self.model.predict(states, verbose=0)
        
        # Update only the Q-value for the action that was actually taken
        for i in range(self.batch_size):
            current_q_values_all_actions[i, actions[i]] = targets_for_taken_actions[i]
        
        # Train the main model
        history = self.model.fit(states, current_q_values_all_actions, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_value # Linear decay per replay step
            self.epsilon = max(self.epsilon_end, self.epsilon) # Ensure it doesn't go below min
        
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
    # Adjusted epsilon_decay_steps for potentially longer training
    agent = DQNAgent(state_dim, action_n, epsilon_decay_steps=50000) 

    episodes = 500
    target_update_freq_steps = 1000 
    print_freq_episodes = 10
    save_freq_episodes = 100
    
    total_training_steps = 0 # Renamed for clarity
    episode_rewards = []
    avg_losses_per_print_freq = []


    for e in range(episodes):
        current_state, _ = env.reset()
        total_episode_reward = 0
        episode_losses = [] # Collect losses within an episode

        for step in range(env.max_steps_per_episode):
            action_idx = agent.act(current_state)
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            agent.remember(current_state, action_idx, reward, next_state, terminated)
            current_state = next_state
            total_episode_reward += reward

            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                episode_losses.append(loss)
                total_training_steps +=1 # Count each replay as a training step

            if total_training_steps > 0 and total_training_steps % target_update_freq_steps == 0:
                agent.update_target_model()
                # print(f"Target model updated at episode {e+1}, total training step {total_training_steps}.") # Less frequent printing
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_episode_reward)
        if episode_losses: # Avoid division by zero if no replays happened
            avg_losses_per_print_freq.append(np.mean(episode_losses))


        if (e + 1) % print_freq_episodes == 0:
            avg_r = np.mean(episode_rewards[-(print_freq_episodes):]) if episode_rewards else 0
            avg_l = np.mean(avg_losses_per_print_freq[-(print_freq_episodes):]) if avg_losses_per_print_freq else 0
            print(f"Ep: {e+1}/{episodes}| Avg Reward (last {print_freq_episodes}): {avg_r:.2f}| Last Ep Reward: {total_episode_reward:.2f}| Avg Loss: {avg_l:.4f}| Epsilon: {agent.epsilon:.3f}| Steps in ep: {step+1}")
            # env.render()

        if (e + 1) % save_freq_episodes == 0:
             agent.save(f"drl_beamsecure_sionna_ep{e+1}.weights.h5")
    
    env.close()
    print("Training finished.")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward per Episode')

        # Smoothing rewards for better trend visibility
        if len(episode_rewards) >= print_freq_episodes:
            smoothed_rewards = np.convolve(episode_rewards, np.ones(print_freq_episodes)/print_freq_episodes, mode='valid')
            plt.plot(np.arange(print_freq_episodes-1, len(episode_rewards)), smoothed_rewards, label=f'Smoothed (window {print_freq_episodes})')
            plt.legend()
        
        plt.subplot(1, 2, 2)
        if avg_losses_per_print_freq: # Plot if there's loss data
            plt.plot(avg_losses_per_print_freq)
            plt.title('Average Loss per Print Frequency')
            plt.xlabel(f'Episode Block (x{print_freq_episodes})')
            plt.ylabel('Average Loss')

        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("Saved training_progress.png")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} Physical GPU(s):")
        try:
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  GPU {i}: {gpu.name} - Memory growth enabled.")
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow sees {len(logical_gpus)} Logical GPU(s).")
        except RuntimeError as e:
            print(f"RuntimeError during GPU setup: {e}")
    else:
        print("No GPU detected by TensorFlow. Model will run on CPU.")
    run_simulation()