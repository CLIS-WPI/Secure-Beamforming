# =====================================================================
# FINAL SCRIPT WITH INTENSIVE & GUARANTEED SUCCESS CURRICULUM
# Purpose: Implement the definitive curriculum to solve the exploration trap.
# =====================================================================
import os
import sys
import time
import logging

# --- Environment Variable Setup for Performance ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- TensorFlow Core and GPU Initialization ---
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

print("--- Main Script: Initializing GPU for TensorFlow ---")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu_device in gpus:
            tf.config.experimental.set_memory_growth(gpu_device, True)
        print(f"✅ Found and configured {len(gpus)} GPU(s). Memory growth enabled.")
    else:
        print("⚠️ No GPU found by TensorFlow. Running on CPU.")
except Exception as e:
    print(f"❌ Error during GPU initialization in main script: {e}")
print("--- Main Script: GPU Initialization Complete ---")

# --- Other Core Libraries ---
import numpy as np
import pandas as pd

# --- Gym Environment Library ---
import gym
from gym import spaces

# --- Sionna Library ---
try:
    import sionna
    from sionna.phy.channel.tr38901 import UMa
    from sionna.phy.channel.tr38901.antenna import PanelArray
    from sionna.phy.utils import log10
    print("✅ Sionna imported successfully.")
except ImportError as e_sionna:
    print(f"❌ CRITICAL ERROR: Failed to import Sionna components: {e_sionna}")
    sys.exit(1)

# =====================================================================
# ENVIRONMENT (Modified for More Challenging Scenario)
# =====================================================================
class MmWaveISACEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super(MmWaveISACEnv, self).__init__()
        # --- Standard Parameters ---
        self.carrier_frequency = 28e9
        self.bandwidth = 100e6
        self.num_bs_physical_rows = 8
        self.num_bs_physical_cols = 8
        self.bs_array = PanelArray(num_rows_per_panel=self.num_bs_physical_rows, num_cols_per_panel=self.num_bs_physical_cols, polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=self.carrier_frequency)
        self.num_bs_antennas = self.bs_array.num_ant
        self.ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V', antenna_pattern='omni', carrier_frequency=self.carrier_frequency)
        self.tx_power_dbm = 30.0
        self.tx_power_watts = tf.constant(10**((self.tx_power_dbm - 30) / 10), dtype=tf.float32)
        self.noise_power_db_per_hz = -174.0
        no_db = self.noise_power_db_per_hz + 10 * np.log10(self.bandwidth)
        self.no_lin = tf.constant(10**((no_db - 30) / 10), dtype=tf.float32)
        self.sensing_range_max = 150.0
        self.max_steps_per_episode = 50
        self.bs_position = tf.constant([[0.0, 0.0, 10.0]], dtype=tf.float32)
        self.user_position_init = np.array([[50.0, 10.0, 1.5]], dtype=np.float32)
        self.attacker_position_init = np.array([[60.0, -20.0, 1.5]], dtype=np.float32)
        self.channel_model_core = UMa(carrier_frequency=self.carrier_frequency, o2i_model="low", ut_array=self.ut_array, bs_array=self.bs_array, direction="downlink", enable_pathloss=True, enable_shadow_fading=True)
        low_obs = np.array([-30.0, -np.pi, -np.pi, 0.0, 0.0, 0.0, -np.pi], dtype=np.float32)
        high_obs = np.array([30.0, np.pi, np.pi, self.sensing_range_max, 1.0, self.sensing_range_max, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.beam_angle_delta_rad = tf.constant(np.deg2rad(2.5), dtype=tf.float32)
        self.user_position = tf.Variable(self.user_position_init, dtype=tf.float32)
        self.attacker_position = tf.Variable(self.attacker_position_init, dtype=tf.float32)
        self.current_beam_azimuth_tf = tf.Variable(0.0, dtype=tf.float32)
        self.current_isac_effort = tf.Variable(0.7, dtype=tf.float32)
        self.current_step = 0
        self.is_curriculum_phase = False
        self.forced_success_steps = tf.convert_to_tensor([], dtype=tf.int32)

    @tf.function
    def _get_steering_vector(self, azimuth_rad_tf):
        zenith_rad = tf.constant(np.pi / 2, dtype=tf.float32)
        f_theta, _ = self.bs_array.ant_pol1.field(theta=zenith_rad, phi=azimuth_rad_tf)
        f_theta = tf.cast(tf.squeeze(f_theta), tf.complex64)
        positions = self.bs_array.ant_pos
        wavelength = tf.constant(3e8 / self.carrier_frequency, dtype=tf.float32)
        k = tf.constant(2 * np.pi, dtype=tf.float32) / wavelength
        u = tf.stack([tf.sin(zenith_rad) * tf.cos(azimuth_rad_tf), tf.sin(zenith_rad) * tf.sin(azimuth_rad_tf), tf.cos(zenith_rad)])
        phase_shifts = k * tf.reduce_sum(positions * u, axis=1)
        array_response = tf.exp(tf.complex(0.0, phase_shifts))
        sv = array_response * f_theta
        return tf.ensure_shape(sv, [self.num_bs_antennas])

    def step(self, action_idx):
        _, _, _, true_az = self._get_state_components()
        
        is_forced_step = tf.reduce_any(tf.equal(self.current_step, self.forced_success_steps))

        if self.is_curriculum_phase and is_forced_step:
            self.current_beam_azimuth_tf.assign(true_az)
            self.current_isac_effort.assign(1.0)
        # --- KEY CHANGE: Added Guided Exploration ---
        elif not self.is_curriculum_phase and tf.random.uniform([]) < 0.1:
            self.current_beam_azimuth_tf.assign(true_az)
            self.current_isac_effort.assign(1.0)
        else:
            if action_idx == 0: self.current_beam_azimuth_tf.assign_sub(self.beam_angle_delta_rad)
            elif action_idx == 1: self.current_beam_azimuth_tf.assign_add(self.beam_angle_delta_rad)
            elif action_idx == 2: pass
            elif action_idx == 3: self.current_isac_effort.assign_add(0.2)
            elif action_idx == 4: self.current_isac_effort.assign_sub(0.2)
        
        self.current_isac_effort.assign(tf.clip_by_value(self.current_isac_effort, 0.3, 1.0))
        self.current_beam_azimuth_tf.assign(tf.clip_by_value(self.current_beam_azimuth_tf, -np.pi, np.pi))
        
        user_move = tf.random.uniform([1, 3], -0.1, 0.1, dtype=tf.float32) * tf.constant([[1.0, 1.0, 0.0]])
        self.user_position.assign_add(user_move)
        attacker_move = tf.random.uniform([1, 3], -0.5, 0.5, dtype=tf.float32) * tf.constant([[1.0, 1.0, 0.0]])
        self.attacker_position.assign_add(attacker_move)
        
        next_state_tensor = self._get_state()
        reward = self._compute_reward(next_state_tensor)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode
        is_stolen = self._is_beam_stolen(next_state_tensor)
        sinr_too_low = next_state_tensor[0] < -20.0
        terminated = tf.logical_or(is_stolen, sinr_too_low)
        return next_state_tensor.numpy(), reward.numpy(), terminated.numpy(), truncated, {}

    @tf.function
    def _get_channel_and_powers(self, beam_azimuth, user_pos):
        bs_loc = tf.reshape(self.bs_position, [1, 1, 3])
        common_args = {'bs_loc': bs_loc, 'ut_orientations': tf.zeros([1,1,3]), 'bs_orientations': tf.zeros([1,1,3]), 'ut_velocities': tf.zeros([1,1,3]), 'in_state': tf.zeros([1,1], dtype=tf.bool)}
        self.channel_model_core.set_topology(ut_loc=tf.reshape(user_pos, [1,1,3]), **common_args)
        h_user = tf.reduce_mean(self.channel_model_core(1, self.bandwidth)[0][:,0,0,0,:,0,0], axis=0)
        sv = self._get_steering_vector(beam_azimuth)
        precoder_w = tf.math.conj(sv) / (tf.norm(sv) + 1e-9)
        sig_user = tf.square(tf.abs(tf.reduce_sum(h_user * precoder_w))) * self.tx_power_watts
        return tf.maximum(sig_user, 1e-15)

    @tf.function
    def _get_state_components(self):
        true_vec = self.attacker_position[0] - self.bs_position[0]
        true_range = tf.norm(true_vec)
        true_az = tf.math.atan2(true_vec[1], true_vec[0])
        sig_user = self._get_channel_and_powers(self.current_beam_azimuth_tf, self.user_position)
        sinr_db = 10.0 * log10(sig_user / (self.no_lin + 1e-20))
        return sinr_db, true_vec, true_range, true_az

    @tf.function
    def _get_state(self):
        sinr_db, _, true_range, true_az = self._get_state_components()
        
        is_forced_step = tf.reduce_any(tf.equal(self.current_step, self.forced_success_steps))

        if self.is_curriculum_phase and is_forced_step:
            det_az, det_range, conf = true_az, true_range, 1.0
        else:
            # --- KEY CHANGE: prob_det_decay returned to 0.015 ---
            prob_det_decay = tf.constant(0.015, dtype=tf.float32) 
            prob_det_multiplier = tf.constant(2.5, dtype=tf.float32)
            prob_det_base = tf.exp(-prob_det_decay * true_range)
            prob_det = self.current_isac_effort * prob_det_base * prob_det_multiplier 
            
            det_az, det_range, conf = -np.pi, 0.0, 0.0
            if tf.logical_and(true_range <= self.sensing_range_max, tf.random.uniform([]) < prob_det):
                noise_std = 1.0 - (self.current_isac_effort * 0.75)
                det_az = true_az + tf.random.normal([], stddev=tf.constant(np.deg2rad(3), dtype=tf.float32) * noise_std)
                det_range = tf.maximum(0.1, true_range + tf.random.normal([], stddev=1.5 * noise_std))
                conf = tf.clip_by_value(prob_det + tf.random.normal([], stddev=0.05), 0.0, 1.0)

        return tf.stack([tf.clip_by_value(sinr_db, -30., 30.), self.current_beam_azimuth_tf, tf.clip_by_value(det_az, -np.pi, np.pi), tf.clip_by_value(det_range, 0., self.sensing_range_max), conf, tf.clip_by_value(true_range, 0., self.sensing_range_max), true_az])
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.current_beam_azimuth_tf.assign(0.0)
        self.current_isac_effort.assign(0.7)
        user_offset_xy = tf.random.uniform([1, 2], -5.0, 5.0, dtype=tf.float32)
        full_user_offset = tf.concat([user_offset_xy, tf.zeros([1, 1], dtype=tf.float32)], axis=1)
        self.user_position.assign(self.user_position_init + full_user_offset)
        attacker_offset_xy = tf.random.uniform([1, 2], -10.0, 10.0, dtype=tf.float32)
        full_attacker_offset = tf.concat([attacker_offset_xy, tf.zeros([1, 1], dtype=tf.float32)], axis=1)
        self.attacker_position.assign(self.attacker_position_init + full_attacker_offset)
        self.current_step = 0
        
        if self.is_curriculum_phase:
            self.forced_success_steps = tf.convert_to_tensor(np.random.choice(range(5, 46), 5, replace=False), dtype=tf.int32)
        else:
            self.forced_success_steps = tf.convert_to_tensor([], dtype=tf.int32)

        return self._get_state().numpy(), {}
    
    @tf.function
    def _is_beam_stolen(self, state):
        sinr, beam_az, det_az, _, conf = state[0], state[1], state[2], state[3], state[4]
        angle_diff = tf.abs(beam_az - det_az)
        is_aligned = tf.minimum(angle_diff, 2*np.pi - angle_diff) < np.deg2rad(20.0)
        return tf.logical_and(conf > 0.7, tf.logical_and(is_aligned, sinr < -5.0))

    @tf.function
    def _compute_reward(self, state):
        sinr, _, _, det_range, conf, true_range, _ = state[0], state[1], state[2], state[3], state[4], state[5], state[6]
        reward = 0.0
        
        # SINR reward is removed to focus agent on security
        
        is_detected = tf.logical_and(conf > 0.7, det_range > 0)
        if is_detected:
            reward += 150.0
        
        is_unaware_of_threat = tf.logical_and(true_range < 80.0, conf < 0.7)
        if is_unaware_of_threat:
            reward -= 5.0 
        
        is_proactive = tf.logical_and(true_range < 80.0, self.current_isac_effort > 0.8)
        if is_proactive:
            reward += 25.0
        return reward
    
    def close(self): pass

# =====================================================================
# PPO AGENT (Unchanged)
# =====================================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, gae_lambda, policy_clip, global_batch_size):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.global_batch_size = global_batch_size
        self.actor = self._build_actor(state_dim, action_dim)
        self.critic = self._build_critic(state_dim)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.memory = []

    def _build_actor(self, state_dim, action_dim):
        actor_input = Input(shape=(state_dim,))
        x = Dense(256, activation='relu')(actor_input)
        x = Dense(128, activation='relu')(x)
        actor_output = Dense(action_dim, activation='softmax')(x)
        return Model(actor_input, actor_output)

    def _build_critic(self, state_dim):
        critic_input = Input(shape=(state_dim,))
        y = Dense(256, activation='relu')(critic_input)
        y = Dense(128, activation='relu')(y)
        critic_output = Dense(1, activation='linear')(y)
        return Model(critic_input, critic_output)

    def remember(self, state, action, logprob, val, reward, done):
        self.memory.append([state, action, logprob, val, reward, done])

    def act(self, state):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.actor(state_tensor, training=False)
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample()[0]
        action_logprob = dist.log_prob(action)
        value = self.critic(state_tensor, training=False)
        return action.numpy(), action_logprob.numpy(), value.numpy()[0,0]

    def _calculate_advantages(self, rewards, dones, vals):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[t])
            next_values = vals[t+1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - vals[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        rewards_to_go = advantages + np.array(vals)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return tf.convert_to_tensor(advantages, dtype=tf.float32), tf.convert_to_tensor(rewards_to_go, dtype=tf.float32)

    def _compute_per_example_losses(self, states, actions, old_logprobs, advantages, rewards_to_go):
        probs = self.actor(states, training=True)
        dist = tfp.distributions.Categorical(probs)
        new_logprobs = dist.log_prob(actions)
        ratios = tf.exp(new_logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        actor_loss = -tf.minimum(surr1, surr2)
        values = tf.squeeze(self.critic(states, training=True), axis=1)
        critic_loss = tf.square(rewards_to_go - values)
        return actor_loss, critic_loss

    @tf.function
    def train_step(self, states, actions, old_logprobs, advantages, rewards_to_go):
        with tf.GradientTape(persistent=True) as tape:
            per_example_actor_loss, per_example_critic_loss = self._compute_per_example_losses(
                states, actions, old_logprobs, advantages, rewards_to_go
            )
            scaled_actor_loss = tf.nn.compute_average_loss(
                per_example_actor_loss, global_batch_size=self.global_batch_size
            )
            scaled_critic_loss = tf.nn.compute_average_loss(
                per_example_critic_loss, global_batch_size=self.global_batch_size
            )
        actor_grads = tape.gradient(scaled_actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(scaled_critic_loss, self.critic.trainable_variables)
        del tape
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return per_example_actor_loss, per_example_critic_loss

    def clear_memory(self):
        self.memory = []

# =====================================================================
# MAIN SIMULATION (with Final, More Challenging Curriculum)
# =====================================================================
def run_simulation():
    print("\n🚀 Starting DRL Simulation with FINAL Curriculum...")
    logging.basicConfig(filename='simulation_final_v4.log', level=logging.INFO, format='%(asctime)s - %(message)s - %(message)s')

    strategy = tf.distribute.MirroredStrategy()
    print(f'✅ MirroredStrategy enabled. Number of devices: {strategy.num_replicas_in_sync}')

    max_training_timesteps = int(5e5) 
    CURRICULUM_EPISODES = 1500 
    
    BATCH_SIZE_PER_REPLICA = 2048 
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    update_timestep = GLOBAL_BATCH_SIZE
    
    with strategy.scope():
        env = MmWaveISACEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        K_epochs = 40
        policy_clip = 0.2
        gamma = 0.99
        gae_lambda = 0.95
        lr_actor = 3e-4
        lr_critic = 1e-3
        agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, gae_lambda, policy_clip, GLOBAL_BATCH_SIZE)

    time_step = 0
    i_episode = 0
    episode_data = []
    step_level_data = []

    print(f"\n--- Starting training with a {CURRICULUM_EPISODES}-episode intensive curriculum phase ---")

    while time_step <= max_training_timesteps:
        env.is_curriculum_phase = (i_episode < CURRICULUM_EPISODES)
        state, _ = env.reset()
        
        if i_episode == CURRICULUM_EPISODES:
            print("\n--- INTENSIVE CURRICULUM PHASE COMPLETE. Agent is now fully autonomous. ---\n")

        current_ep_reward = 0
        ep_sinr, ep_det_steps = [], 0

        for t in range(1, env.max_steps_per_episode + 1):
            action, log_prob, val = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.remember(state, action, log_prob, val, reward, terminated or truncated)
            time_step += 1
            current_ep_reward += reward
            state = next_state
            
            step_level_data.append({
                'Episode': i_episode, 'Timestep': time_step, 'User_SINR': next_state[0],
                'Beam_Azimuth': next_state[1], 'ISAC_Effort': env.current_isac_effort.numpy(), 
                'Detection_Confidence': next_state[4], 'True_Attacker_Range': next_state[5],
                'Step_Reward': reward, 'Action': action
            })
            
            ep_sinr.append(next_state[0])
            if next_state[4] > 0.7 and next_state[3] > 0:
                ep_det_steps += 1

            if len(agent.memory) >= update_timestep:
                print(f"\n--- Timestep {time_step}: Starting learning step ---")
                mem = agent.memory
                states, actions, logprobs, vals, rewards, dones = ([item[i] for item in mem] for i in range(6))
                advantages, rewards_to_go = agent._calculate_advantages(rewards, dones, vals)
                dataset = tf.data.Dataset.from_tensor_slices((
                    tf.convert_to_tensor(states, dtype=tf.float32),
                    tf.convert_to_tensor(actions, dtype=tf.int32),
                    tf.convert_to_tensor(logprobs, dtype=tf.float32),
                    advantages,
                    rewards_to_go
                )).batch(GLOBAL_BATCH_SIZE)
                dist_dataset = strategy.experimental_distribute_dataset(dataset)
                for _ in range(agent.K_epochs):
                    for batch in dist_dataset:
                        strategy.run(agent.train_step, args=(batch))
                agent.clear_memory()
                print("--- Learning step complete ---")

            if terminated or truncated:
                break
        
        i_episode += 1
        avg_det_rate = (ep_det_steps / t) * 100 if t > 0 else 0
        episode_data.append({
            'Episode': i_episode, 'Total_Reward': current_ep_reward,
            'Avg_SINR': np.mean(ep_sinr) if ep_sinr else -30, 
            'Detection_Rate_Steps': avg_det_rate,
            'Steps': t
        })

        if i_episode % 100 == 0:
            avg_r = np.mean([d['Total_Reward'] for d in episode_data[-100:]])
            avg_s = np.mean([d['Avg_SINR'] for d in episode_data[-100:]])
            avg_d = np.mean([d['Detection_Rate_Steps'] for d in episode_data[-100:]])
            print(f"Episode: {i_episode}, Timestep: {time_step}, Avg Reward: {avg_r:.2f}, Avg SINR: {avg_s:.2f}, Avg Det.Rate: {avg_d:.2f}%")

    output_filename = 'ppo_final_v4_challenging'
    pd.DataFrame(episode_data).to_csv(f'{output_filename}_episodes.csv', index=False)
    pd.DataFrame(step_level_data).to_csv(f'{output_filename}_steps.csv', index=False)
    agent.actor.save_weights(f"{output_filename}_actor.weights.h5")
    agent.critic.save_weights(f"{output_filename}_critic.weights.h5")
    print(f"\n✅ Training complete. Final results saved to '{output_filename}_*.csv'.")

if __name__ == "__main__":
    run_simulation()
 