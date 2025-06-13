import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_publication_training_curves(df_episode, window_size=100):
    """
    Plots the final learning curves for reward, SINR, detection rate, and policy stability,
    formatted for publication.
    """
    if df_episode.empty:
        print("Error: Episode data is empty. Skipping training curve plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('PPO Agent Training Performance (Shaped Reward)', fontsize=22, fontweight='bold')

    # Plot 1: Cumulative Reward
    reward_moving_avg = df_episode['Total_Reward'].rolling(window=window_size, min_periods=1).mean()
    axes[0].plot(df_episode['Episode'], reward_moving_avg, color='royalblue', linewidth=2.5, label=f'Moving Average (window={window_size})')
    axes[0].scatter(df_episode['Episode'], df_episode['Total_Reward'], alpha=0.1, s=15, color='lightsteelblue')
    axes[0].set_ylabel('Cumulative Reward', fontsize=14)
    axes[0].set_title('Reward Progression', fontsize=16)
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Detection Rate
    detection_moving_avg = df_episode['Detection_Rate_Steps'].rolling(window=window_size, min_periods=1).mean()
    axes[1].plot(df_episode['Episode'], detection_moving_avg, color='crimson', linewidth=2.5)
    axes[1].scatter(df_episode['Episode'], df_episode['Detection_Rate_Steps'], alpha=0.1, s=15, color='lightcoral')
    axes[1].set_ylabel('Successful Detection Rate (%)', fontsize=14)
    axes[1].set_title('Security Performance: Attacker Detection Rate', fontsize=16)
    axes[1].set_ylim(0, 105)

    # Plot 3: Communication Quality (SINR)
    sinr_moving_avg = df_episode['Avg_SINR'].rolling(window=window_size, min_periods=1).mean()
    axes[2].plot(df_episode['Episode'], sinr_moving_avg, color='forestgreen', linewidth=2.5)
    axes[2].scatter(df_episode['Episode'], df_episode['Avg_SINR'], alpha=0.1, s=15, color='lightgreen')
    axes[2].set_ylabel('Average SINR (dB)', fontsize=14)
    axes[2].set_title('Communication Performance', fontsize=16)
    axes[2].axhline(y=5.0, color='darkorange', linestyle='--', linewidth=2, label='Adequate SINR Threshold (5 dB)')
    axes[2].legend()

    # Plot 4: Policy Stability (Standard Deviation of Reward)
    reward_std_moving_avg = df_episode['Total_Reward'].rolling(window=window_size, min_periods=1).std()
    axes[3].plot(df_episode['Episode'], reward_std_moving_avg, color='purple', linewidth=2.5)
    axes[3].set_ylabel('Standard Deviation of Reward', fontsize=14)
    axes[3].set_title('Learned Policy Stability Analysis', fontsize=16)
    axes[3].set_xlabel('Episode', fontsize=14)
    
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('figure_publication_1_training_curves.pdf', format='pdf', dpi=300)
    plt.savefig('figure_publication_1_training_curves.png', format='png', dpi=300)
    print("✅ Figure 1 (Publication Training Curves) saved.")

def plot_publication_tradeoff(df_episode):
    """
    Creates a scatter plot to visualize the final trade-off between SINR and Detection Rate.
    """
    if df_episode.empty:
        print("Warning: Episode data is empty. Skipping trade-off plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use the latter half of training data to represent the converged policy
    converged_df = df_episode.iloc[int(len(df_episode) * 0.5):]

    scatter = ax.scatter(
        converged_df['Avg_SINR'],
        converged_df['Detection_Rate_Steps'],
        c=converged_df['Total_Reward'],
        cmap='viridis',
        alpha=0.6,
        s=60,
        edgecolor='black',
        linewidth=0.7
    )
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Cumulative Reward', fontsize=14)
    
    ax.set_title('Final Learned Policy: SINR vs. Detection Trade-off', fontsize=18, fontweight='bold')
    ax.set_xlabel('Average SINR per Episode (dB)', fontsize=14)
    ax.set_ylabel('Detection Rate per Episode (%)', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('figure_publication_2_tradeoff_plot.pdf', format='pdf', dpi=300)
    plt.savefig('figure_publication_2_tradeoff_plot.png', format='png', dpi=300)
    print("✅ Figure 2 (SINR/Detection Rate Trade-off) saved.")

def plot_publication_isac_strategy(df_steps):
    """
    Creates a Kernel Density Estimate plot to show the agent's ISAC effort strategy.
    """
    if df_steps.empty:
        print("Warning: Step-level data not found. Skipping ISAC strategy plot.")
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use the latter half of step data for the converged policy
    converged_steps = df_steps.iloc[int(len(df_steps) * 0.5):]
    
    near_attacker_df = converged_steps[converged_steps['True_Attacker_Range'] < 75.0]
    far_attacker_df = converged_steps[converged_steps['True_Attacker_Range'] >= 75.0]

    if not near_attacker_df.empty:
        sns.kdeplot(near_attacker_df['ISAC_Effort'], ax=ax, color='orangered', label='Attacker Near (< 75m)', fill=True, alpha=0.5, linewidth=2.5, bw_adjust=0.5)
    if not far_attacker_df.empty:
        sns.kdeplot(far_attacker_df['ISAC_Effort'], ax=ax, color='dodgerblue', label='Attacker Far (>= 75m)', fill=True, alpha=0.5, linewidth=2.5, bw_adjust=0.5)

    ax.set_title('Learned Strategy for ISAC Resource Allocation', fontsize=18, fontweight='bold')
    ax.set_xlabel('ISAC Effort', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('figure_publication_3_isac_strategy.pdf', format='pdf', dpi=300)
    plt.savefig('figure_publication_3_isac_strategy.png', format='png', dpi=300)
    print("✅ Figure 3 (ISAC Resource Allocation Strategy) saved.")


if __name__ == '__main__':
    # --- Correctly point to the final output files ---
    episode_file = 'ppo_final_intensive_curriculum_episodes.csv'
    step_file = 'ppo_final_intensive_curriculum_steps.csv'
    
    # Load episode data
    if os.path.exists(episode_file):
        episode_df = pd.read_csv(episode_file)
        print(f"Successfully loaded '{episode_file}'.")
    else:
        print(f"Error: '{episode_file}' not found. Episode-level analysis is not possible.")
        episode_df = pd.DataFrame()

    # Load step-level data
    if os.path.exists(step_file):
        step_level_df = pd.read_csv(step_file)
        print(f"Successfully loaded '{step_file}'.")
    else:
        print(f"\nWarning: Step-level data file ('{step_file}') not found.")
        step_level_df = pd.DataFrame()

    print("\n--- Generating Publication-Ready Performance Figures ---")
    plot_publication_training_curves(episode_df)
    plot_publication_tradeoff(episode_df)
    plot_publication_isac_strategy(step_level_df)

    # --- In-depth statistical analysis on the converged policy ---
    if not episode_df.empty:
        print("\n--- Final Converged Policy Performance Analysis ---")
        # Use the last 30% of episodes for final policy analysis
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
