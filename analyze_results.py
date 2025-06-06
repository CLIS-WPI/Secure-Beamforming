import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_reward(df_episode, window_size=25):
    """
    Plots a clean and clear training reward curve.
    This shows that the DRL agent was successfully learning.
    """
    # Use a professional, publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5)) # Appropriate size for a single-column paper

    # Calculate the moving average to show the overall trend
    moving_avg = df_episode['Total_Reward'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Plot the raw reward data as semi-transparent scatter points
    ax.scatter(df_episode['Episode'], df_episode['Total_Reward'], alpha=0.2, s=10, label='Reward per Episode', color='lightgray')
    
    # Plot the smooth moving average with a clear, bold line
    ax.plot(df_episode['Episode'], moving_avg, color='royalblue', linewidth=2.5, label=f'Moving Average (window={window_size})')

    ax.set_title('DRL Agent Training Progression', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # Save as PDF for high quality in LaTeX
    plt.savefig('figure_1_training_curve.pdf', format='pdf', dpi=300)
    plt.savefig('figure_1_training_curve.png', format='png', dpi=300)
    print("✅ Figure 1 (Training Curve) saved as figure_1_training_curve.pdf/.png")

def plot_sinr_distribution(df_eval):
    """
    Creates a box plot to compare the SINR distribution of the DRL agent and the baseline.
    This is the most important figure for showing the performance trade-off.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data for the plot
    plot_data = pd.DataFrame({
        'Baseline': df_eval['Baseline_SINR'],
        'DRL-ISAC (Ours)': df_eval['DRL_SINR_Avg_Eval']
    })

    # Create the boxplot with professional, muted colors
    sns.boxplot(data=plot_data, ax=ax, palette=['skyblue', 'lightgreen'], width=0.5,
                boxprops=dict(alpha=.8), whiskerprops=dict(alpha=.8), capprops=dict(alpha=.8))

    # Add individual data points for better visualization of the distribution
    sns.stripplot(data=plot_data, ax=ax, color=".25", size=3)

    ax.set_title('SINR Distribution over 100 Test Scenarios', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average UE SINR (dB)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add median value annotations for clarity
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
    print("✅ Figure 2 (SINR Distribution) saved as figure_2_sinr_distribution.pdf/.png")

def plot_isac_effort_strategy(df_steps):
    """
    Creates a histogram to show how the agent uses ISAC effort strategically.
    It compares ISAC effort when the attacker is near vs. far.
    This plot directly uses the step-level data.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    # We define "near" as closer than half the max sensing range (75m)
    near_attacker_df = df_steps[df_steps['True_Attacker_Range'] < 75.0]
    far_attacker_df = df_steps[df_steps['True_Attacker_Range'] >= 75.0]

    # Plot histograms
    sns.histplot(near_attacker_df['ISAC_Effort'], bins=np.arange(0.2, 1.3, 0.2), 
                 ax=ax, color='orangered', label='Attacker Near (< 75m)', stat='density', alpha=0.7)
    sns.histplot(far_attacker_df['ISAC_Effort'], bins=np.arange(0.2, 1.3, 0.2), 
                 ax=ax, color='dodgerblue', label='Attacker Far (>= 75m)', stat='density', alpha=0.6)

    ax.set_title('Learned ISAC Effort Strategy', fontsize=16, fontweight='bold')
    ax.set_xlabel('ISAC Effort (Sensing-to-Communication Power Ratio)', fontsize=12)
    ax.set_ylabel('Proportional Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('figure_3_isac_effort.pdf', format='pdf', dpi=300)
    plt.savefig('figure_3_isac_effort.png', format='png', dpi=300)
    print("✅ Figure 3 (ISAC Effort Strategy) saved as figure_3_isac_effort.pdf/.png")


if __name__ == '__main__':
    try:
        # Load the data from the CSV files generated by your simulation
        episode_df = pd.read_csv('episode_results_v6.csv')
        eval_df = pd.read_csv('evaluation_results_v6.csv')
        step_level_df = pd.read_csv('step_level_detection_data_v6.csv') # This file is key
        print("CSV files loaded successfully.")

        print("\n--- Generating Publication-Quality Figures ---")
        
        # --- Figure 1: Training Curve ---
        plot_training_reward(episode_df)
        
        # --- Figure 2: SINR Distribution Box Plot ---
        plot_sinr_distribution(eval_df)

        # --- Figure 3: ISAC Effort Strategy Analysis ---
        plot_isac_effort_strategy(step_level_df)

        # --- NEW: Add the statistical analysis at the end ---
        print("\n--- Additional KPI Analysis ---")
        
        # Analysis suggested by your friend (based on detection outcome)
        avg_isac_effort_detected = step_level_df[step_level_df['Detection_Outcome'] == 1]['ISAC_Effort'].mean()
        avg_isac_effort_not_detected = step_level_df[step_level_df['Detection_Outcome'] == 0]['ISAC_Effort'].mean()
        print(f"Avg ISAC Effort (When Attacker was Detected): {avg_isac_effort_detected:.2f}")
        print(f"Avg ISAC Effort (When Attacker was Not Detected): {avg_isac_effort_not_detected:.2f}")

        # The analysis we used in the paper's text (based on attacker proximity)
        near_attacker_df = step_level_df[step_level_df['True_Attacker_Range'] < 75.0]
        far_attacker_df = step_level_df[step_level_df['True_Attacker_Range'] >= 75.0]
        avg_isac_effort_near = near_attacker_df['ISAC_Effort'].mean()
        avg_isac_effort_far = far_attacker_df['ISAC_Effort'].mean()
        print(f"Avg ISAC Effort (When Attacker was Near < 75m): {avg_isac_effort_near:.2f}")
        print(f"Avg ISAC Effort (When Attacker was Far >= 75m): {avg_isac_effort_far:.2f}")
        
        print("\nAll figures and analysis completed successfully!")

    except FileNotFoundError as e:
        print(f"\nError: {e}. Make sure the required CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")