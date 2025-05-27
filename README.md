# Secure-Beamforming-DRL-ISAC

## 🛡️ Intelligent mmWave Beam Security through Deep Reinforcement Learning and ISAC

This project explores a novel framework for **proactive and adaptive secure beamforming** in millimeter-wave (mmWave) communication systems. It leverages **Deep Reinforcement Learning (DRL)** to defend against sophisticated **beam-stealing attacks**. A key innovation is the use of **Integrated Sensing and Communications (ISAC)** capabilities for active, intelligent **threat assessment**, enabling the DRL agent to make more informed decisions to protect the communication link.

---

## 🎯 Project Goal

The primary goal is to develop and simulate a mmWave base station agent that can:
* Autonomously learn optimal beamforming strategies to maintain secure communication.
* Intelligently utilize ISAC to detect and characterize potential beam-stealing threats.
* Proactively adapt its defenses to thwart sophisticated attackers.
* Demonstrate improved resilience and security compared to traditional or static defense mechanisms.

---

## 🧪 Core Technologies & Approach

* **mmWave Communication:** Simulating directed beamforming and channel characteristics.
* **Beam-Stealing Attacks:** Modeling intelligent adversaries attempting to hijack communication beams.
* **Deep Reinforcement Learning (DRL):** A DQN-based agent is trained to make decisions regarding beam steering and ISAC utilization.
* **Integrated Sensing and Communications (ISAC):** Used as an active sensing tool to provide situational awareness and threat intelligence to the DRL agent.
* **Simulation Platform:** The conceptual framework is designed with [NVIDIA Sionna](https://nvlabs.github.io/sionna/) library in mind for channel modeling and TensorFlow for DRL implementation. The ultimate aim is to simulate this within a more comprehensive platform like NVIDIA SIONA.

---

## 🚀 Current Status & Simulation Outline

The current Python code provides a foundational simulation environment using `sionna` for channel modeling and `tensorflow` for the DQN agent.

**High-Level Simulation Steps:**
1.  **Environment Setup:** A mmWave scenario with a base station, legitimate user(s), and attacker(s) is modeled.
2.  **DRL Agent Formulation:**
    * **State Space:** Includes SINR, current beam angles, and ISAC-derived threat data (e.g., detected attacker DoA, range, confidence).
    * **Action Space:** Discrete actions for beam adjustments (azimuth, elevation) and ISAC effort levels (e.g., high/low probing effort).
    * **Reward Function:** Designed to incentivize high user SINR, accurate attacker localization via ISAC, and penalize beam-stealing or misdirecting the beam towards threats.
3.  **Training:** The DRL agent is trained over numerous episodes to learn optimal policies.
4.  **Evaluation:** Performance is assessed based on metrics like beam-stealing success rate, legitimate user SINR/throughput, and attack detection/mitigation effectiveness.

---

## 🛠️ To-Do / Future Enhancements

* Develop more sophisticated attacker models and strategies.
* Refine the ISAC sensing model for more detailed environmental perception.
* Expand the DRL agent's state and action spaces for finer control and awareness.
* Implement and compare against more baseline defense mechanisms.
* Explore advanced DRL algorithms (e.g., PPO, SAC) for potentially better performance.
* Scale simulations and integrate with more comprehensive platforms like NVIDIA SIONA for larger-scale experiments and potential hardware-in-the-loop testing.

---


## 📄Publication

This research aims to build upon concepts in mmWave security and advance the state-of-the-art by integrating DRL with active ISAC-based threat assessment for proactive defense.

