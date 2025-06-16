# Secure mmWave Beamforming with Proactive ISAC Defense

This repository contains the source code for the paper: **"Secure mmWave Beamforming with Proactive ISAC Defense Against Beam Stealing Attacks"**. This work introduces a novel framework using Deep Reinforcement Learning (DRL) to defend against sophisticated beam-stealing attacks in millimeter-wave (mmWave) communication systems.

### **System Overview**

Our framework models a scenario where a Base Station (BS) must maintain a secure communication link with a legitimate User Equipment (UE) while under threat from a malicious Attacker. The key innovation is the use of an advanced DRL agent that leverages Integrated Sensing and Communications (ISAC) to proactively detect and neutralize the threat.

### **Key Features**

* **Advanced DRL Agent:** Built upon the **Proximal Policy Optimization (PPO)** algorithm, our agent demonstrates stable and robust performance in a complex, high-dimensional environment.
* **Intensive Curriculum Learning:** We designed a novel, two-phase curriculum learning strategy to solve the "exploration trap" problem. This includes a **Forced Success Curriculum** that guarantees the agent experiences successful detections, enabling it to learn an effective security policy.
* **Proactive ISAC Defense:** The agent learns a dynamic and adaptive policy to intelligently allocate ISAC resources, performing targeted sensing only when a threat is perceived, thus ensuring both security and resource efficiency.
* **High-Fidelity Simulation:** The environment is simulated using **Sionna**, a GPU-accelerated library for link-level simulations, based on the 3GPP TR 38.901 channel model.

### **Requirements**

The project is designed to run in a containerized environment to ensure reproducibility. The main dependencies are:

* **Docker**
* **NVIDIA Container Toolkit** (for GPU acceleration)
* Python 3.12+
* TensorFlow 2.17+
* Sionna 
* NumPy, Pandas, Matplotlib, Seaborn

### **Setup and Execution**

The entire simulation is designed to be run inside a Docker container.

**Step 1: Grant Execute Permissions**

First, make the startup script executable:
```bash
chmod +x start.sh
```

**Step 2: Build the Docker Image**

Build the container image which includes all necessary libraries (Python, TensorFlow, Sionna, etc.).
```bash
docker build -t sionna-with-tf-container .
```

**Step 3: Run the Simulation**

Execute the main simulation script inside the container. This command mounts the current directory into the container's `/workspace` and allocates GPU resources.
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace sionna-with-tf-container
```

Inside the container, run the main simulation:
```bash
python3 main.py
```

*The training process is computationally intensive and may take several hours to complete, even on high-end GPUs.*

### **Analyzing the Results**

After the simulation is complete, it will generate two primary CSV files: `ppo_final_intensive_guaranteed_episodes.csv` and `ppo_final_intensive_guaranteed_steps.csv`.

To generate all figures and statistical summaries used in the paper, run the analysis script:
```bash
python3 analyze_results.py
```

This will save the figures as both `.png` and `.pdf` files in the root directory.

### **Citing Our Work**

If you use this code or our methodology in your research, please consider citing our paper:

```bibtex
@inproceedings{hashemi2024secure,
  title={Secure mmWave Beamforming with Proactive ISAC Defense Against Beam Stealing Attacks},
  author={Hashemi Natanzi, Seyed Bagher and Mohammadi, Hossein and Tang, Bo and Marojevic, Vuk},
  organization={IEEE}
}
```

### **License**

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

### **Acknowledgments**

This material is based upon work supported in part by the National Science Foundation (NSF) under Awards CNS-2120442 and IIS-2325863, and the National Telecommunications and Information Administration (NTIA) under Award No. 51-60-IF007.
