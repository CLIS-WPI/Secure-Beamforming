import numpy as np
import tensorflow as tf
import pandas as pd
import os

# We import the environment directly from your main script
# Make sure this file is in the same directory as your main script
from main import MmWaveISACEnv 

def run_baseline_simulation(num_episodes=100):
    """
    Runs a simulation for a simple, non-learning baseline agent.
    This agent always keeps its beam pointed at the user's initial location.
    """
    print("\n🚀 Starting Baseline Simulation...")
    
    # Initialize the environment
    env = MmWaveISACEnv()
    
    all_sinr_values = []
    
    # --- Main Simulation Loop ---
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        
        # Calculate the fixed direction towards the user's initial position
        user_initial_pos = env.user_position_init[0]
        bs_pos = env.bs_position.numpy()[0]
        direction_vector = user_initial_pos - bs_pos
        # Calculate the fixed azimuth angle in radians
        fixed_azimuth = np.arctan2(direction_vector[1], direction_vector[0])
        
        # In each step, the baseline agent performs a fixed action
        # It holds its beam towards the user and uses a constant ISAC effort.
        env.current_beam_azimuth_tf.assign(fixed_azimuth)
        env.current_isac_effort.assign(0.5) # A constant, medium effort
        
        ep_sinr = []
        
        for t in range(env.max_steps_per_episode):
            # The baseline agent's action is always "2" (Hold Beam)
            # The beam direction and ISAC effort are already set above and won't change.
            action = 2 
            
            # We don't need the full step logic, just the SINR calculation
            # For simplicity, we can directly get the state which includes SINR
            state, _, _, _, _ = env.step(action)
            
            sinr = state[0]
            ep_sinr.append(sinr)

        avg_ep_sinr = np.mean(ep_sinr)
        all_sinr_values.append(avg_ep_sinr)
        
        if (i_episode + 1) % 10 == 0:
            print(f"Episode {i_episode + 1}/{num_episodes} complete. Average SINR so far: {np.mean(all_sinr_values):.2f} dB")

    # --- Final Results ---
    final_mean_sinr = np.mean(all_sinr_values)
    final_std_sinr = np.std(all_sinr_values)
    final_median_sinr = np.median(all_sinr_values)
    final_min_sinr = np.min(all_sinr_values)
    final_max_sinr = np.max(all_sinr_values)

    print("\n" + "="*40)
    print("✅ Baseline Simulation Complete.")
    print("="*40)
    print("\n--- Baseline Performance Metrics ---")
    print(f"  - Mean SINR: {final_mean_sinr:.2f} dB")
    print(f"  - Std. Dev. of SINR: {final_std_sinr:.2f} dB")
    print(f"  - Median SINR: {final_median_sinr:.2f} dB")
    print(f"  - Min / Max SINR: {final_min_sinr:.2f} / {final_max_sinr:.2f} dB")
    print(f"  - Detection Rate: 0.00% (by definition)")
    print("\nUse these values to fill the 'Baseline' column in your paper's table.")
    print("="*40)


if __name__ == "__main__":
    # Ensure TensorFlow uses GPUs if available, but it's less critical for this simple script
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu_device in gpus:
                tf.config.experimental.set_memory_growth(gpu_device, True)
            print(f"Found and configured {len(gpus)} GPU(s).")
        else:
            print("No GPU found. Running on CPU.")
    except Exception as e:
        print(f"Error during GPU initialization: {e}")
        
    run_baseline_simulation()
