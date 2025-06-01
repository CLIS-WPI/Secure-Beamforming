import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# --- Plotting Parameters ---
plt.rcParams.update({'font.size': 12, 'figure.figsize': (18, 12)}) # Larger font and figure size

# --- Load Data ---
try:
    # Make sure to use the correct filenames for your latest results
    episode_df = pd.read_csv('episode_results_v5.csv')
    eval_df = pd.read_csv('evaluation_results_v5.csv')
    print("CSV files loaded successfully.")
except FileNotFoundError:
    print("Error: Ensure 'episode_results_v5.csv' and 'evaluation_results_v5.csv' are in the correct path.")
    exit()

# --- 1. Distribution Plot (Histogram) of DRL SINR and Baseline SINR in Evaluation Phase ---
plt.figure()
plt.hist(eval_df['DRL_SINR_Avg_Eval'], bins=10, alpha=0.7, label='DRL SINR (Greedy)')
plt.hist(eval_df['Baseline_SINR'], bins=10, alpha=0.7, label='Baseline SINR')
plt.title('Evaluation SINR Distribution')
plt.xlabel('SINR (dB)')
plt.ylabel('Number of Test Scenarios')
plt.legend()
plt.grid(True)
plt.savefig('evaluation_sinr_distribution.png')
print("Evaluation SINR distribution plot saved: evaluation_sinr_distribution.png")

# --- 2. Plot DRL SINR vs. Baseline SINR for each test scenario ---
plt.figure()
num_eval_scenarios = len(eval_df)
scenario_indices = np.arange(1, num_eval_scenarios + 1)
plt.plot(scenario_indices, eval_df['Baseline_SINR'], marker='o', linestyle='--', label='Baseline SINR')
plt.plot(scenario_indices, eval_df['DRL_SINR_Avg_Eval'], marker='x', linestyle='-', label='DRL SINR (Greedy)')
plt.title('SINR Comparison per Test Scenario')
plt.xlabel('Test Scenario')
plt.ylabel('Average SINR (dB)')
plt.xticks(scenario_indices) # Ensure all scenario numbers are shown
plt.legend()
plt.grid(True)
plt.savefig('evaluation_sinr_comparison_per_scenario.png')
print("Per-scenario SINR comparison plot saved: evaluation_sinr_comparison_per_scenario.png")

# --- 3. Detection Rate vs. True Attacker Range (Requires more granular data from training) ---
# To plot this, you need to log during training, for each step where detection occurs:
# - The true_attacker_range (from next_state[5])
# - The detection outcome (e.g., attack_detected_this_step: 0 or 1)
# The current `episode_results_vX.csv` only has average detection rate per episode.

# Assuming you will have a DataFrame named `step_log_df` with columns `True_Attacker_Range`
# and `Detection_Outcome` (0 or 1) from a modified main simulation script.
# Here's placeholder code for how you might plot it:

# # Example with hypothetical data (replace with your actual data loading and processing)
# if 'True_Attacker_Range' in episode_df.columns and 'Attack_Detected_Step' in episode_df.columns: # Assuming these columns are added
#     # This section will need to be rewritten if you save step-wise data to a separate file
#     # or if you parse it from the simulation.log
#     print("Plotting Detection Rate vs. Attacker Range (requires more granular step-level data).")
# else:
#     print("Required columns for 'Detection Rate vs. Attacker Range' not found in episode_results_v5.csv.")
#     print("You need to log true_attacker_range and detection_outcome for each step.")

print("\nNote: For the 'Detection Rate vs. Attacker Range' plot, you need data that records")
print("the true attacker range and detection outcome at each step. This is not in the current CSVs.")


# --- 4. Statistical Analysis: 95% Confidence Intervals ---
# For average SINRs in the evaluation phase
baseline_sinr_eval = eval_df['Baseline_SINR']
drl_sinr_eval = eval_df['DRL_SINR_Avg_Eval']

# Calculate 95% CI for SINR means
if len(baseline_sinr_eval) > 1:
    ci_baseline_sinr = st.t.interval(confidence=0.95, df=len(baseline_sinr_eval)-1,
                                      loc=np.mean(baseline_sinr_eval),
                                      scale=st.sem(baseline_sinr_eval))
else:
    ci_baseline_sinr = (np.nan, np.nan) # Cannot compute SEM for a single value

if len(drl_sinr_eval) > 1:
    ci_drl_sinr = st.t.interval(confidence=0.95, df=len(drl_sinr_eval)-1,
                                loc=np.mean(drl_sinr_eval),
                                scale=st.sem(drl_sinr_eval))
else:
    ci_drl_sinr = (np.nan, np.nan)

print("\n--- Statistical Analysis (95% Confidence Intervals) ---")
print(f"Baseline SINR (Evaluation): Mean = {np.mean(baseline_sinr_eval):.2f} dB, 95% CI = ({ci_baseline_sinr[0]:.2f}, {ci_baseline_sinr[1]:.2f})")
print(f"DRL SINR (Greedy, Evaluation): Mean = {np.mean(drl_sinr_eval):.2f} dB, 95% CI = ({ci_drl_sinr[0]:.2f}, {ci_drl_sinr[1]:.2f})")

# For detection rates in the evaluation phase (assuming binary outcomes 0 or 1)
baseline_detection_eval = eval_df['Baseline_Detected_Binary']
drl_detection_eval = eval_df['DRL_Detected_In_Episode_Binary']

# Calculate CI for proportions (e.g., using normal approximation for binomial proportion)
# p_hat +/- z * sqrt(p_hat*(1-p_hat)/n)
z_score = st.norm.ppf(0.975) # For 95% CI

# Baseline Detection Rate CI
p_baseline = np.mean(baseline_detection_eval)
n_baseline = len(baseline_detection_eval)
if n_baseline > 0 and 0 < p_baseline < 1: # Ensure p_baseline is not 0 or 1 for meaningful interval
    ci_baseline_detection_lower = p_baseline - z_score * np.sqrt(p_baseline * (1 - p_baseline) / n_baseline)
    ci_baseline_detection_upper = p_baseline + z_score * np.sqrt(p_baseline * (1 - p_baseline) / n_baseline)
    print(f"Baseline Detection Rate (Evaluation): Mean = {p_baseline*100:.2f}%, 95% CI = ({max(0, ci_baseline_detection_lower)*100:.2f}%, {min(1, ci_baseline_detection_upper)*100:.2f}%)")
else:
    print(f"Baseline Detection Rate (Evaluation): Mean = {p_baseline*100:.2f}% (CI not applicable or rate is 0% or 100%)")

# DRL Detection Rate CI
p_drl = np.mean(drl_detection_eval)
n_drl = len(drl_detection_eval)
if n_drl > 0 and 0 < p_drl < 1: # Ensure p_drl is not 0 or 1
    ci_drl_detection_lower = p_drl - z_score * np.sqrt(p_drl * (1 - p_drl) / n_drl)
    ci_drl_detection_upper = p_drl + z_score * np.sqrt(p_drl * (1 - p_drl) / n_drl)
    print(f"DRL Detection Rate (Greedy, Evaluation): Mean = {p_drl*100:.2f}%, 95% CI = ({max(0, ci_drl_detection_lower)*100:.2f}%, {min(1, ci_drl_detection_upper)*100:.2f}%)")
else:
    print(f"DRL Detection Rate (Greedy, Evaluation): Mean = {p_drl*100:.2f}% (CI not applicable or rate is 0% or 100%)")

plt.show() # Display all plots at the end (if running in an environment that supports it)
print("\nAnalysis and plotting complete.")