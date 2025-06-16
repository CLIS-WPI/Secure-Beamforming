import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_publication_training_curves(df_episode, window_size=100):
    """
    Plots the final learning curves for reward, SINR, detection rate, and policy stability,
    formatted for publication in a 2x2 grid.
    """
    if df_episode.empty:
        print("Error: Episode data is empty. Skipping training curve plot.")
        return

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16)) # Changed to 2x2 layout
    fig.suptitle('PPO Agent Training Performance Analysis', fontsize=28, fontweight='bold')

    # Define subplot labels and flatten axes for easy iteration
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    flat_axes = axes.flatten()

    # --- Plot (a): Cumulative Reward ---
    ax = flat_axes[0]
    reward_moving_avg = df_episode['Total_Reward'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(df_episode['Episode'], reward_moving_avg, color='royalblue', linewidth=3, label=f'Moving Avg (w={window_size})')
    ax.scatter(df_episode['Episode'], df_episode['Total_Reward'], alpha=0.1, s=20, color='lightsteelblue')
    ax.set_ylabel('Cumulative Reward', fontsize=20)
    ax.set_title('Reward Progression', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16)
    
    # --- Plot (b): Detection Rate ---
    ax = flat_axes[1]
    detection_moving_avg = df_episode['Detection_Rate_Steps'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(df_episode['Episode'], detection_moving_avg, color='crimson', linewidth=3, label=f'Moving Avg (w={window_size})')
    ax.scatter(df_episode['Episode'], df_episode['Detection_Rate_Steps'], alpha=0.1, s=20, color='lightcoral')
    ax.set_ylabel('Successful Detection Rate (%)', fontsize=20)
    ax.set_title('Security Performance', fontsize=22, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=16)

    # --- Plot (c): Communication Quality (SINR) ---
    ax = flat_axes[2]
    sinr_moving_avg = df_episode['Avg_SINR'].rolling(window=window_size, min_periods=1).mean()
    ax.plot(df_episode['Episode'], sinr_moving_avg, color='forestgreen', linewidth=3, label=f'Moving Avg (w={window_size})')
    ax.scatter(df_episode['Episode'], df_episode['Avg_SINR'], alpha=0.1, s=20, color='lightgreen')
    ax.set_ylabel('Average SINR (dB)', fontsize=20)
    ax.set_title('Communication Performance', fontsize=22, fontweight='bold')
    ax.axhline(y=5.0, color='darkorange', linestyle='--', linewidth=2.5, label='Adequate SINR Threshold (5 dB)')
    ax.legend(fontsize=16)

    # --- Plot (d): Policy Stability ---
    ax = flat_axes[3]
    reward_std_moving_avg = df_episode['Total_Reward'].rolling(window=window_size, min_periods=1).std()
    ax.plot(df_episode['Episode'], reward_std_moving_avg, color='purple', linewidth=3, label=f'Moving Avg (w={window_size})')
    ax.set_ylabel('Std. Dev. of Reward', fontsize=20)
    ax.set_title('Learned Policy Stability', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16)

    # General formatting for all subplots
    for i, ax in enumerate(flat_axes):
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('Episode', fontsize=20)
        # Add subplot labels (a), (b), etc.
        ax.text(-0.1, 1.05, subplot_labels[i], transform=ax.transAxes, 
                fontsize=26, fontweight='bold', va='top', ha='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
    plt.savefig('figure_publication_1_training_curves_updated.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_publication_1_training_curves_updated.png', format='png', dpi=300, bbox_inches='tight')
    print("✅ Figure 1 (Updated Publication Training Curves) saved.")

def plot_publication_tradeoff(df_episode):
    """
    Creates a clearer scatter plot to visualize the final trade-off between SINR and Detection Rate.
    """
    if df_episode.empty:
        print("Warning: Episode data is empty. Skipping trade-off plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9)) # Increased figure size

    # Use the latter half of training data to represent the converged policy
    converged_df = df_episode.iloc[int(len(df_episode) * 0.5):]

    scatter = ax.scatter(
        converged_df['Avg_SINR'],
        converged_df['Detection_Rate_Steps'],
        c=converged_df['Total_Reward'],
        cmap='viridis',
        alpha=0.7, # Increased alpha for better visibility
        s=80,      # Increased size for better visibility
        edgecolor='k',
        linewidth=0.8
    )
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cumulative Reward', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    ax.set_title('Final Learned Policy: SINR vs. Detection Trade-off', fontsize=22, fontweight='bold')
    ax.set_xlabel('Average SINR per Episode (dB)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Detection Rate per Episode (%)', fontsize=18, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig('figure_publication_2_tradeoff_plot_updated.pdf', format='pdf', dpi=300)
    plt.savefig('figure_publication_2_tradeoff_plot_updated.png', format='png', dpi=300)
    print("✅ Figure 2 (Updated SINR/Detection Rate Trade-off) saved.")

def plot_publication_isac_strategy(df_steps):
    """
    Creates a clearer Kernel Density Estimate plot to show the agent's ISAC effort strategy.
    """
    if df_steps.empty:
        print("Warning: Step-level data not found. Skipping ISAC strategy plot.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size
    
    # Use the latter half of step data for the converged policy
    converged_steps = df_steps.iloc[int(len(df_steps) * 0.5):]
    
    near_attacker_df = converged_steps[converged_steps['True_Attacker_Range'] < 75.0]
    far_attacker_df = converged_steps[converged_steps['True_Attacker_Range'] >= 75.0]

    if not near_attacker_df.empty:
        sns.kdeplot(near_attacker_df['ISAC_Effort'], ax=ax, color='orangered', label='Attacker Near (< 75m)', fill=True, alpha=0.6, linewidth=3, bw_adjust=0.5)
    if not far_attacker_df.empty:
        sns.kdeplot(far_attacker_df['ISAC_Effort'], ax=ax, color='dodgerblue', label='Attacker Far (>= 75m)', fill=True, alpha=0.6, linewidth=3, bw_adjust=0.5)

    ax.set_title('Learned Strategy for ISAC Resource Allocation', fontsize=22, fontweight='bold')
    ax.set_xlabel('ISAC Effort', fontsize=18, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=18, fontweight='bold')
    ax.legend(fontsize=16) # Increased legend font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig('figure_publication_3_isac_strategy_updated.pdf', format='pdf', dpi=300)
    plt.savefig('figure_publication_3_isac_strategy_updated.png', format='png', dpi=300)
    print("✅ Figure 3 (Updated ISAC Resource Allocation Strategy) saved.")


if __name__ == '__main__':
    # --- Correctly point to the final output files ---
    episode_file = 'ppo_final_v4_challenging_episodes.csv'
    step_file = 'ppo_final_v4_challenging_steps.csv'
    
    # Initialize DataFrames to prevent NameError
    episode_df = pd.DataFrame()
    step_level_df = pd.DataFrame()

    # Load episode data with robust error handling
    if os.path.exists(episode_file):
        try:
            episode_df = pd.read_csv(episode_file)
            print(f"Successfully loaded '{episode_file}'.")
        except Exception as e:
            print(f"Error loading '{episode_file}': {e}. Proceeding with empty DataFrame.")
    else:
        print(f"Error: '{episode_file}' not found. Episode-level analysis is not possible.")

    # Load step-level data with robust error handling
    if os.path.exists(step_file):
        try:
            step_level_df = pd.read_csv(step_file)
            print(f"Successfully loaded '{step_file}'.")
        except Exception as e:
            print(f"Error loading '{step_file}': {e}. Proceeding with empty DataFrame.")
    else:
        print(f"\nWarning: Step-level data file ('{step_file}') not found.")
        
    print("\n--- Generating Publication-Ready Performance Figures ---")
    plot_publication_training_curves(episode_df)
    plot_publication_tradeoff(episode_df)
    plot_publication_isac_strategy(step_level_df)

    # --- In-depth statistical analysis on the converged policy ---
    if not episode_df.empty:
        print("\n--- Final Converged Policy Performance Analysis ---")
        # --- FIX: Use the correct variable name 'episode_df' ---
        converged_episodes = episode_df.iloc[int(len(episode_df) * 0.7):]
        
        print(f"Analysis performed on the last {len(converged_episodes)} episodes:\n")
        
        # Calculate statistics
        reward_stats = converged_episodes['Total_Reward'].describe()
        sinr_stats = converged_episodes['Avg_SINR'].describe()
        detection_stats = converged_episodes['Detection_Rate_Steps'].describe()
        
        print("--- Reward Statistics ---")
        print(f"  - Mean: {reward_stats['mean']:.2f}")
        print(f"  - Std Dev (Stability): {reward_stats['std']:.2f}")
        print(f"  - Median (Typical Value): {reward_stats['50%']:.2f}")
        print(f"  - Min / Max: {reward_stats['min']:.2f} / {reward_stats['max']:.2f}\n")
        
        print("--- Communication Quality (User SINR) Statistics ---")
        print(f"  - Mean: {sinr_stats['mean']:.2f} dB")
        print(f"  - Std Dev: {sinr_stats['std']:.2f} dB")
        print(f"  - Median: {sinr_stats['50%']:.2f} dB")
        print(f"  - Min / Max: {sinr_stats['min']:.2f} / {sinr_stats['max']:.2f} dB\n")

        print("--- Security Performance (Detection Rate) Statistics ---")
        print(f"  - Mean Detection Rate: {detection_stats['mean']:.2f}%")
        print(f"  - Std Dev: {detection_stats['std']:.2f}%")
        print(f"  - Median: {detection_stats['50%']:.2f}%")
        print(f"  - Max Detection Rate in an episode: {detection_stats['max']:.2f}%\n")

    print("\nAnalysis successfully completed.")
