import sionna
import tensorflow as tf
import numpy as np
from sionna.channel.tr38901 import UMa # Using UMa from tr38901 for more typical channel modeling
from sionna.utils import compute_gain, ebnodb_to_no, log10, expand_to_rank, complex_normal
from sionna.mimo import StreamManagement, lmmse_equalizer
from sionna.mapping import Mapper, Demapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import gym
from gym import spaces
import tensorflow.keras as keras
from collections import deque
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Define the mmWave ISAC Environment as a Gym environment ---
class MmWaveISACEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(MmWaveISACEnv, self).__init__()

        # Simulation parameters
        self.carrier_frequency = 28e9  # 28 GHz mmWave
        self.bandwidth = 100e6 # 100 MHz bandwidth
        self.num_bs_antennas_sqrt = 8 # BS has num_bs_antennas_sqrt^2 antennas
        self.num_bs_antennas = self.num_bs_antennas_sqrt**2
        self.num_ue_antennas = 4
        self.num_user = 1
        self.num_attacker = 1
        self.bs_array = sionna.channel.tr38901.PanelArray(
            num_rows_per_panel=self.num_bs_antennas_sqrt,
            num_cols_per_panel=self.num_bs_antennas_sqrt,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.carrier_frequency
        )
        self.ue_array = sionna.channel.tr38901.PatchArray(
            polarization="dual",
            antenna_pattern="omni", # Simpler UE antenna
            carrier_frequency=self.carrier_frequency
        )


        self.tx_power_dbm = 20.0  # BS transmit power in dBm
        self.tx_power_watts = 10**((self.tx_power_dbm - 30) / 10)
        self.noise_power_db_per_hz = -174.0 # Noise power spectral density in dBm/Hz
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth)
        self.no_lin = 10**((no_db - 30) / 10) # Linear noise power

        self.sensing_range_max = 150.0  # Max ISAC sensing range in meters
        self.max_steps_per_episode = 100 # Max steps before episode termination

        # Sionna channel model (3GPP UMa from TR38.901)
        self.channel_model = UMa(
            carrier_frequency=self.carrier_frequency,
            bandwidth=self.bandwidth,
            bs_array=self.bs_array,
            ut_array=self.ue_array,
            domain="time", # Time domain simulation for easier PDP-like features if needed later
            los_probability="R1-20_LoS", # LoS probability model
            delay_spread="R1-20_RMS_DS"
        )

        # --- State Space ---
        # [SINR_User (dB),
        #  Current_Beam_Steering_Angle_Azimuth (radians), Current_Beam_Steering_Angle_Elevation (radians),
        #  ISAC_Detected_Attacker_DoA_Azimuth (radians, or special value if none),
        #  ISAC_Detected_Attacker_DoA_Elevation (radians, or special value if none),
        #  ISAC_Detected_Attacker_Range (meters, or special value if none),
        #  ISAC_Detection_Confidence (0-1)
        # ]
        # Using -np.pi-1, -1, -1 as special values for no detection
        low_obs = np.array([-30.0, -np.pi, -np.pi/2, -np.pi-1, -np.pi/2-1, -1.0, 0.0], dtype=np.float32)
        high_obs = np.array([30.0, np.pi, np.pi/2, np.pi, np.pi/2, self.sensing_range_max, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # --- Action Space (Discrete) ---
        # Action 0: Steer beam slightly left
        # Action 1: Steer beam slightly right
        # Action 2: Steer beam slightly up
        # Action 3: Steer beam slightly down
        # Action 4: Maintain current beam
        # Action 5: ISAC High Effort Probing (more accurate sensing, higher cost)
        # Action 6: ISAC Low Effort Probing (less accurate sensing, lower cost)
        self.num_discrete_actions = 7
        self.action_space = spaces.Discrete(self.num_discrete_actions)
        self.beam_angle_delta = np.deg2rad(5) # 5 degrees adjustment

        # Initial positions (x, y, z) - BS at origin
        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32) # BS at 10m height
        self.user_position_init = np.array([50.0, 10.0, 1.5]) # User at (50m, 10m) on x-y plane, 1.5m height
        self.attacker_position_init = np.array([60.0, -15.0, 1.5]) # Attacker

        self.user_position = tf.Variable(self.user_position_init.reshape(1,3), dtype=tf.float32)
        self.attacker_position = tf.Variable(self.attacker_position_init.reshape(1,3), dtype=tf.float32)

        # Beam steering angles [azimuth, elevation] in radians
        self.current_beam_angles_tf = tf.Variable([0.0, 0.0], dtype=tf.float32) # Azimuth, Elevation
        self.current_isac_effort = 0.5 # Normalized 0 (low) to 1 (high)

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles a new seed if provided
        self.current_beam_angles_tf.assign([0.0, 0.0])
        self.current_isac_effort = 0.5

        # Slightly randomize initial positions for varied episodes
        self.user_position.assign(self.user_position_init + np.random.normal(0, 1.0, 3).reshape(1,3))
        self.attacker_position.assign(self.attacker_position_init + np.random.normal(0, 2.0, 3).reshape(1,3))
        
        self.current_step = 0
        #print("Environment Reset.")
        return self._get_state(), {}

    def _get_steering_vector(self, angles_rad_tf):
        # angles_rad_tf: [azimuth, elevation]
        # This is a simplified way to get a steering vector.
        # Sionna's bs_array.steering_vector expects azimuth and zenith.
        # Elevation = 90 - zenith. So, zenith = 90 - elevation.
        azimuth_rad = angles_rad_tf[0]
        elevation_rad = angles_rad_tf[1]
        zenith_rad = np.pi/2 - elevation_rad
        
        # Sionna expects shape [num_tx, num_tx_ant, num_streams_per_tx]
        # For a single stream from the BS:
        sv = self.bs_array.steering_vector(tf.expand_dims(azimuth_rad, axis=0),
                                           tf.expand_dims(zenith_rad, axis=0))
        # sv shape is [1, num_bs_antennas, 1]. We need [num_bs_antennas, 1] for precoder
        return tf.cast(tf.squeeze(sv, axis=0), tf.complex64)


    def _get_state(self):
        # --- Communication Part ---
        # Configure transmitter and receiver positions for Sionna channel
        # BS is tx_array_orientation [1, 3], ue_array_orientation [1, 3], attacker_array_orientation [1,3]
        # Assuming BS looks along x-axis, UEs are omni so orientation less critical
        # For simplicity, we'll assume fixed orientations or that they are ideally pointed for max gain apart from beamsteering.
        # Sionna's UMa positions are [batch_size, num_rx, num_tx, 3]
        # Here, batch_size=1, num_tx=1 (BS), num_rx=2 (User, Attacker)
        
        # Concatenate user and attacker positions for channel computation
        # Shape for UMa: [batch_size, num_links, 3] where num_links is num_rx * num_tx
        # Here, one BS transmitting to User (link 0) and to Attacker (link 1)
        ue_locations_for_channel = tf.concat([self.user_position, self.attacker_position], axis=0) # Shape [2, 3]
        # UMa model needs specific inputs for bs_orientations, ut_orientations, velocities if used
        # For static scenario for now:
        bs_config = (self.bs_position,          # location
                     [0.0, 0.0, 0.0],           # orientation [yaw, pitch, roll]
                     [0.0, 0.0, 0.0])           # velocity [vx, vy, vz]
        
        ut_configs_user = (self.user_position, [0.0,0.0,0.0], [0.0,0.0,0.0])
        ut_configs_attacker = (self.attacker_position, [0.0,0.0,0.0], [0.0,0.0,0.0])

        # Get channel to user
        h_user_time, _ = self.channel_model(bs_config, ut_configs_user) # Time-domain channel
        h_user = h_user_time[0,0,0,:,:,0] # Extract [num_rx_ant, num_tx_ant] for the first path (simplification)
                                          # True channel would be CIR; for simplicity taking first tap as channel matrix
                                          # For OFDM, you'd use h_user_freq from channel_model.cir_to_ofdm_channel
        
        # Get channel to attacker
        h_attacker_time, _ = self.channel_model(bs_config, ut_configs_attacker)
        h_attacker = h_attacker_time[0,0,0,:,:,0]

        # Beamforming towards current_beam_angles_tf
        # This is a simplification: in reality, precoder might be more complex (e.g., ZF, MMSE based on CSI)
        # Here, we use a steering vector as the precoder (Max-Ratio Transmission towards steered angle)
        steering_vec = self._get_steering_vector(self.current_beam_angles_tf) # [num_bs_antennas, 1]
        precoder_w = tf.math.conj(steering_vec) / tf.norm(steering_vec) # Normalize
        precoder_w = tf.expand_dims(precoder_w, axis=0) # Shape [1, num_bs_antennas, 1] for Sionna functions

        # Signal power at User
        y_user_vec = tf.matmul(tf.cast(tf.expand_dims(h_user, axis=0), tf.complex64), precoder_w) # Effective channel after precoding
        signal_power_user = tf.reduce_sum(tf.abs(y_user_vec)**2).numpy() * self.tx_power_watts

        # Signal power at Attacker (potential interference to user if attacker relays, or eavesdropped power)
        y_attacker_vec = tf.matmul(tf.cast(tf.expand_dims(h_attacker, axis=0), tf.complex64), precoder_w)
        signal_power_attacker = tf.reduce_sum(tf.abs(y_attacker_vec)**2).numpy() * self.tx_power_watts
        
        # Simplified SINR for the user (assuming attacker might cause interference, not full beam-stealing model yet)
        # For beam-stealing, the "interference" is that the attacker *becomes* the receiver
        # For this state, let's focus on legitimate user's SINR
        # A more elaborate model would consider if the beam is *pointed at* the attacker
        sinr_user = 10 * np.log10(signal_power_user / self.no_lin) if signal_power_user > 0 else -100.0

        # --- ISAC Sensing Part ---
        # Simulate ISAC: estimate DoA and range to attacker based on self.current_isac_effort
        # Higher effort = lower noise in sensing
        sensing_accuracy_factor = 1.0 - (self.current_isac_effort * 0.8) # Max 80% noise reduction
        
        true_attacker_vector = self.attacker_position[0].numpy() - self.bs_position[0].numpy()
        true_attacker_range = np.linalg.norm(true_attacker_vector)
        true_attacker_azimuth = np.arctan2(true_attacker_vector[1], true_attacker_vector[0])
        true_attacker_elevation = np.arctan2(true_attacker_vector[2], np.linalg.norm(true_attacker_vector[:2]))

        detected_attacker_doa_az, detected_attacker_doa_el, detected_attacker_range, detection_confidence = -np.pi-1, -np.pi/2-1, -1.0, 0.0
        
        # Simple detection probability based on range and effort
        # This is a placeholder for a real ISAC sensing model
        if true_attacker_range <= self.sensing_range_max:
            prob_detection = self.current_isac_effort * (1 - true_attacker_range / (self.sensing_range_max * 1.5))
            if np.random.rand() < prob_detection:
                detection_confidence = prob_detection # Use prob as confidence
                noise_az = np.random.normal(0, np.deg2rad(10) * sensing_accuracy_factor) # Angle noise
                noise_el = np.random.normal(0, np.deg2rad(10) * sensing_accuracy_factor)
                noise_range = np.random.normal(0, 5.0 * sensing_accuracy_factor)  # Range noise in meters

                detected_attacker_doa_az = true_attacker_azimuth + noise_az
                detected_attacker_doa_el = true_attacker_elevation + noise_el
                detected_attacker_range = max(0, true_attacker_range + noise_range)
        
        state_array = np.array([
            sinr_user,
            self.current_beam_angles_tf[0].numpy(), self.current_beam_angles_tf[1].numpy(),
            detected_attacker_doa_az, detected_attacker_doa_el, detected_attacker_range,
            detection_confidence
        ], dtype=np.float32)
        return state_array

    def step(self, action_idx):
        # Apply discrete action
        # Action 0-3: Steer beam azimuth/elevation
        # Action 4: Maintain current beam
        # Action 5: ISAC High Effort
        # Action 6: ISAC Low Effort
        current_az, current_el = self.current_beam_angles_tf.numpy()

        if action_idx == 0: # Steer Azimuth Left
            current_az -= self.beam_angle_delta
        elif action_idx == 1: # Steer Azimuth Right
            current_az += self.beam_angle_delta
        elif action_idx == 2: # Steer Elevation Up
            current_el += self.beam_angle_delta
        elif action_idx == 3: # Steer Elevation Down
            current_el -= self.beam_angle_delta
        elif action_idx == 4: # Maintain beam
            pass # No change to beam angles
        elif action_idx == 5: # ISAC High Effort
            self.current_isac_effort = 1.0
        elif action_idx == 6: # ISAC Low Effort
            self.current_isac_effort = 0.2
        
        # Clip beam angles
        current_az = np.clip(current_az, -np.pi, np.pi)
        current_el = np.clip(current_el, -np.pi/2 * 0.9, np.pi/2 * 0.9) # Limit elevation to avoid gimbal lock issues conceptually
        self.current_beam_angles_tf.assign([current_az, current_el])

        # Simulate mobility (simple random walk)
        self.user_position.assign_add(tf.random.normal(shape=[1,3], mean=0.0, stddev=0.1, dtype=tf.float32))
        self.attacker_position.assign_add(tf.random.normal(shape=[1,3], mean=0.0, stddev=0.5, dtype=tf.float32)) # Attacker might move more unpredictably

        next_state_obs = self._get_state()
        reward = self._compute_reward(next_state_obs)
        
        self.current_step += 1
        done = False
        if self.current_step >= self.max_steps_per_episode:
            done = True
        # Additional done conditions based on security breach can be added
        # For example, if beam points directly at attacker for too long with low user SINR
        if self._is_beam_stolen(next_state_obs):
            done = True
            reward -= 100 # Large penalty for being stolen

        return next_state_obs, reward, done, False, {} # Added truncated flag for Gym new API

    def _is_beam_stolen(self, current_obs_state):
        # Placeholder: Define what constitutes a "beam stolen" state
        # Example: If beam is close to attacker AND user SINR is very low AND attacker is detected close
        sinr_user = current_obs_state[0]
        beam_az = current_obs_state[1]
        beam_el = current_obs_state[2]
        attacker_doa_az = current_obs_state[3]
        attacker_range = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        if detection_confidence > 0.5 and attacker_range > 0 and attacker_range < 20: # Attacker detected close
            angle_diff_az = abs(beam_az - attacker_doa_az)
            # Normalize angle difference
            if angle_diff_az > np.pi: angle_diff_az = 2*np.pi - angle_diff_az
            
            if angle_diff_az < np.deg2rad(15) and sinr_user < -5.0: # Beam close to detected attacker & bad user SINR
                print("!!! Beam considered stolen !!!")
                return True
        return False

    def _compute_reward(self, current_obs_state):
        sinr_user = current_obs_state[0]
        beam_az = current_obs_state[1]
        beam_el = current_obs_state[2]
        detected_attacker_doa_az = current_obs_state[3]
        detected_attacker_doa_el = current_obs_state[4]
        detected_attacker_range = current_obs_state[5]
        detection_confidence = current_obs_state[6]

        # 1. Reward for User SINR
        reward = tf.clip_by_value(sinr_user / 10.0, -2.0, 2.0).numpy() # Normalize and clip

        # 2. Penalty if beam is directed towards a detected attacker
        #    And reward if beam is directed away from detected attacker
        if detection_confidence > 0.5 and detected_attacker_range > 0: # If attacker is confidently detected
            # Calculate true direction to user for optimal pointing
            user_vec = self.user_position[0].numpy() - self.bs_position[0].numpy()
            true_user_az = np.arctan2(user_vec[1], user_vec[0])
            
            angle_diff_beam_to_attacker_az = abs(beam_az - detected_attacker_doa_az)
            if angle_diff_beam_to_attacker_az > np.pi: angle_diff_beam_to_attacker_az = 2*np.pi - angle_diff_beam_to_attacker_az

            # Penalize pointing towards attacker
            reward -= (1.0 - (angle_diff_beam_to_attacker_az / np.pi)) * 2.0 * detection_confidence # Max penalty -2

            # Reward for ISAC accuracy (if it helped detect)
            reward += detection_confidence * 0.5

            # Small penalty for high ISAC effort to encourage efficiency
            if self.current_isac_effort > 0.8:
                 reward -= 0.1
        else:
            # If attacker not detected, incentivize pointing towards user
            user_vec = self.user_position[0].numpy() - self.bs_position[0].numpy()
            true_user_az = np.arctan2(user_vec[1], user_vec[0])
            angle_diff_beam_to_user_az = abs(beam_az - true_user_az)
            if angle_diff_beam_to_user_az > np.pi: angle_diff_beam_to_user_az = 2*np.pi - angle_diff_beam_to_user_az
            reward += (1.0 - (angle_diff_beam_to_user_az / np.pi)) * 1.0 # Max reward +1 for perfect alignment

        # Penalty for ISAC not detecting a close attacker
        true_attacker_range_val = np.linalg.norm(self.attacker_position[0].numpy() - self.bs_position[0].numpy())
        if true_attacker_range_val < self.sensing_range_max / 2 and detection_confidence < 0.3:
            reward -= 1.0

        return reward

    def render(self, mode='human'):
        # For simplicity, printing key state info.
        # A real render would use matplotlib or a more complex visualizer.
        if mode == 'human':
            print(f"  Step: {self.current_step}")
            print(f"  Beam Angles (Az,El): ({np.rad2deg(self.current_beam_angles_tf[0].numpy()):.1f}°, "
                  f"{np.rad2deg(self.current_beam_angles_tf[1].numpy()):.1f}°)")
            state = self._get_state()
            print(f"  User SINR: {state[0]:.2f} dB")
            if state[6] > 0.1: # If detection confidence is high enough
                print(f"  Detected Attacker: Az={np.rad2deg(state[3]):.1f}°, El={np.rad2deg(state[4]):.1f}°, Range={state[5]:.1f}m (Conf: {state[6]:.2f})")
            else:
                print("  Attacker Not Detected or Low Confidence.")
            print(f"  ISAC Effort: {self.current_isac_effort:.2f}")


    def close(self):
        pass

# --- DRL Agent (DQN) ---
class DQNAgent:
    def __init__(self, state_dim, action_n):
        self.state_dim = state_dim
        self.action_n = action_n
        self.memory = deque(maxlen=10000) # Increased memory
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05 # Minimum exploration rate
        self.epsilon_decay = 0.999 # Slower decay
        self.learning_rate = 0.0005 # Adjusted learning rate
        self.batch_size = 64    # Batch size for training

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep Q-learning Model
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_n, activation='linear') # Q-values for each discrete action
        ])
        model.compile(loss='huber_loss', # Huber loss can be more robust
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        # print("Target model updated.")

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_n) # Explore action space
        
        # state might be (7,) need to be (1,7)
        if state.ndim == 1:
            state_for_pred = np.expand_dims(state, axis=0)
        else:
            state_for_pred = state

        act_values = self.model.predict(state_for_pred, verbose=0)
        return np.argmax(act_values[0])  # Exploit learned values

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Predict Q-values for current states and next states
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        targets = np.copy(q_values_current)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(q_values_next_target[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()
        print(f"Model weights loaded from {name}")

    def save(self, name):
        self.model.save_weights(name)
        print(f"Model weights saved to {name}")

# --- Main simulation function ---
def run_simulation():
    print("Starting DRL Simulation for Secure mmWave ISAC Beamforming...")
    env = MmWaveISACEnv()
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    agent = DQNAgent(state_dim, action_n)

    episodes = 500 # Increased episodes for some learning
    max_steps = env.max_steps_per_episode
    target_update_freq = 10 # Update target model every 10 episodes
    print_freq = 20
    save_freq = 100

    all_rewards = []

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action_idx = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action_idx)
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            if done or truncated:
                break
        
        all_rewards.append(total_reward)

        if (e + 1) % target_update_freq == 0:
            agent.update_target_model()
            print(f"Target model updated at episode {e+1}.")

        if (e + 1) % print_freq == 0:
            avg_reward = np.mean(all_rewards[-(print_freq):])
            print(f"Episode: {e+1}/{episodes}, Avg Reward (last {print_freq}): {avg_reward:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {step+1}")
            # env.render() # Optional: render last step of an episode

        if (e + 1) % save_freq == 0:
             agent.save(f"drl_beamsecure_sionna_ep{e+1}.weights.h5")


    print("Training finished.")
    # TODO: Add an evaluation phase here, where epsilon is set to 0 (or very low)
    # and the agent's performance on specific scenarios is measured.
    # The baseline comparison from the original code can be adapted here.

if __name__ == "__main__":
    # Check if GPU is available for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be set before GPUs have been initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs available.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU detected. TensorFlow will run on CPU.")

    run_simulation()
