import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_reward(df_episode, window_size=25):
    """
    Plots a clean and clear training reward curve.
    This shows that the DRL agent was successfully learning.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    moving_avg = df_episode['Total_Reward'].rolling(window=window_size, center=True, min_periods=1).mean()
    ax.scatter(df_episode['Episode'], df_episode['Total_Reward'], alpha=0.2, s=10, label='Reward per Episode', color='lightgray')
    ax.plot(df_episode['Episode'], moving_avg, color='royalblue', linewidth=2.5, label=f'Moving Average (window={window_size})')

    ax.set_title('DRL Agent Training Progression', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('figure_1_training_curve.pdf', format='pdf', dpi=300)
    plt.savefig('figure_1_training_curve.png', format='png', dpi=300)
    print("✅ Figure 1 (Training Curve) saved.")

def plot_sinr_distribution(df_eval):
    """
    Creates a box plot to compare the SINR distribution of the DRL agent and the baseline.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = pd.DataFrame({
        'Baseline': df_eval['Baseline_SINR'],
        'DRL-ISAC (Ours)': df_eval['DRL_SINR_Avg_Eval']
    })
    sns.boxplot(data=plot_data, ax=ax, palette=['skyblue', 'lightgreen'], width=0.5,
                boxprops=dict(alpha=.8), whiskerprops=dict(alpha=.8), capprops=dict(alpha=.8))
    sns.stripplot(data=plot_data, ax=ax, color=".25", size=3)
    ax.set_title('SINR Distribution over 100 Test Scenarios', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average UE SINR (dB)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    median_baseline = plot_data['Baseline'].median()
    median_drl = plot_data['DRL-ISAC (Ours)'].median()
    ax.text(0, median_baseline, f' {median_baseline:.2f}', 
            verticalalignment='center', size='small', color='black', weight='semibold',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.text(1, median_drl, f' {median_drl:.2f}', 
            verticalalignment='center', size='small', color='black', weight='semibold',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    plt.tight_layout()
    plt.savefig('figure_2_sinr_distribution.pdf', format='pdf', dpi=300)
    plt.savefig('figure_2_sinr_distribution.png', format='png', dpi=300)
    print("✅ Figure 2 (SINR Distribution) saved.")

def plot_isac_effort_strategy(df_steps):
    """
    Creates a histogram to show how the agent uses ISAC effort strategically.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    near_attacker_df = df_steps[df_steps['True_Attacker_Range'] < 75.0]
    far_attacker_df = df_steps[df_steps['True_Attacker_Range'] >= 75.0]
    sns.histplot(near_attacker_df['ISAC_Effort'], bins=np.arange(0.2, 1.3, 0.2), 
                 ax=ax, color='orangered', label='Attacker Near (< 75m)', stat='density', alpha=0.7)
    sns.histplot(far_attacker_df['ISAC_Effort'], bins=np.arange(0.2, 1.3, 0.2), 
                 ax=ax, color='dodgerblue', label='Attacker Far (>= 75m)', stat='density', alpha=0.6)
    ax.set_title('Learned ISAC Effort Strategy', fontsize=16, fontweight='bold')
    ax.set_xlabel('ISAC Effort', fontsize=12)
    ax.set_ylabel('Proportional Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure_3_isac_effort.pdf', format='pdf', dpi=300)
    plt.savefig('figure_3_isac_effort.png', format='png', dpi=300)
    print("✅ Figure 3 (ISAC Effort Strategy) saved.")


if __name__ == '__main__':
    try:
        # Load the data from the CSV files
        episode_df = pd.read_csv('episode_results_v7.csv')
        eval_df = pd.read_csv('evaluation_results_v7.csv')
        step_level_df = pd.read_csv('step_level_detection_data_v7.csv')
        print("CSV files loaded successfully.")

        print("\n--- Generating Publication-Quality Figures ---")
        plot_training_reward(episode_df)
        plot_sinr_distribution(eval_df)
        plot_isac_effort_strategy(step_level_df)

        # --- NEW: Advanced Statistical Analysis ---
        print("\n--- Advanced KPI Analysis ---")
        
        # --- ISAC Effort Analysis ---
        near_attacker_df = step_level_df[step_level_df['True_Attacker_Range'] < 75.0]
        far_attacker_df = step_level_df[step_level_df['True_Attacker_Range'] >= 75.0]
        avg_isac_effort_near = near_attacker_df['ISAC_Effort'].mean()
        avg_isac_effort_far = far_attacker_df['ISAC_Effort'].mean()
        print(f"Avg ISAC Effort (Attacker Near < 75m): {avg_isac_effort_near:.2f}")
        print(f"Avg ISAC Effort (Attacker Far >= 75m): {avg_isac_effort_far:.2f}")
        
        # --- Detection Rate Analysis based on Proximity ---
        # Get the episode numbers where the attacker was near vs far
        near_episodes = near_attacker_df['Episode'].unique()
        far_episodes = far_attacker_df['Episode'].unique()
        
        # Filter evaluation data for these episodes
        eval_near = eval_df[eval_df['Test_Scenario'].isin(near_episodes)]
        eval_far = eval_df[eval_df['Test_Scenario'].isin(far_episodes)]

        if not eval_near.empty:
            critical_detection_rate = eval_near['DRL_Detected_In_Episode_Binary'].mean() * 100
            print(f"Critical Detection Rate (Attacker Near < 75m): {critical_detection_rate:.2f}%")
        else:
            print("No 'near attacker' scenarios found in evaluation data.")

        if not eval_far.empty:
            non_critical_detection_rate = eval_far['DRL_Detected_In_Episode_Binary'].mean() * 100
            print(f"Non-Critical Detection Rate (Attacker Far >= 75m): {non_critical_detection_rate:.2f}%")
        else:
            print("No 'far attacker' scenarios found in evaluation data.")
            
        print("\nAll figures and analysis completed successfully!")

    except FileNotFoundError as e:
        print(f"\nError: {e}. Make sure the required CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")