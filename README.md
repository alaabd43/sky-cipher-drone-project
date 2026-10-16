# 🚁 Sky Cipher: Drone Resilience & Mission Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Reinforcement Learning](https://img.shields.io/badge/RL-Reinforcement_Learning-red?style=for-the-badge)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

**Sky Cipher** is a predictive AI simulation designed to answer one critical question: Will the drone survive the jam? By merging Deep Reinforcement Learning with high-fidelity *PyBullet physics*, this platform simulates severe **Electronic Warfare (EW)** and GPS-denial scenarios. It provides a purely predictive framework to evaluate mission success and evasive strategies when the skies are actively contested.

## 🛡️ The Mission: Defeating the Jamming

In modern conflict zones, drones face constant threats from GPS denial and communication interference. **Sky Cipher** provides a predictive framework to:

- **Simulate Hostile Environments:** Model complex signal interference patterns.
- **Predict Mission Success:** Use AI to forecast whether a drone will reach its target or fail due to jamming.
- **Adaptive Behavior:** Train RL agents to develop "evasive" maneuvers or alternative navigation logic when primary signals are lost.

## 🧠 Technical Core

- **Physics Engine:** Powered by `PyBullet` for realistic 6-DOF drone dynamics.
- **AI Architecture:** Deep Reinforcement Learning agents trained to navigate through high-interference zones.
- **Data-Driven Insights:** Includes the `SkyCipher Dataset`, a comprehensive collection of flight frames categorized by mission outcomes (Evading, Crash, Success).
- **Predictive Dashboard:** A `Streamlit` interface for real-time visualization of mission success probabilities.
- **👁️ Tactical Visual Intelligence**

The **Vision Detection** module utilizes **YOLOv8** to provide the drone with real-time object detection capabilities. This allows the system to identify potential threats or targets even during electronic interference, ensuring the mission continues using visual backup when GPS or signal loss occurs.

## ⚙️ Enterprise Integration & Threat Response

To bridge the gap between AI prediction and real-world operations, **Sky Cipher** utilizes an **n8n** automation pipeline. This workflow acts as a tactical router, instantly processing the AI's predictive outcomes and triggering automated alerts (e.g., when a drone enters a severe jamming zone or faces an imminent crash).

![n8n Automation Workflow](https://github.com/user-attachments/assets/969b5561-5c8a-4876-a885-e6b8220c1b37)

## 📂 Repository Structure

- 🛠️ `environment.py`: The core simulation arena featuring jamming variables and physics constraints.
- 🚀 `train.py` & `test.py`: Scripts for training the neural networks and evaluating mission resilience.
- 👁️ `Vision_Detection/`: A comprehensive Computer Vision suite featuring YOLOv8 detection and AirSim integration for tactical visual intelligence.
- 📊 `mission_control.py`: The interactive command center (Dashboard) for mission prediction.
- ⚙️ `threat_automation_pipeline.json`: An n8n workflow designed to automate system alerts and data routing based on AI predictions.
- 🖼️ `dataset_images/`: Over 9,000+ labeled frames capturing drone behavior under electronic attack.
- 📦 `drone_assets/`: Custom URDF and 3D models for the quadrotor simulation.

## 🛠️ Tech Stack & Tools

- **AI/ML:** Python, YOLOv8 (Ultralytics), Deep Reinforcement Learning.
- **Simulation:** AirSim, Unreal Engine 4.
- **Automation:** n8n Workflow Engine.
- **Data Analysis:** Pandas, Matplotlib, Streamlit (for Dashboard).

## 🚀 Quick Start

1. **Clone the project:**

   ```bash
   git clone https://github.com/alaabd43/sky-cipher-drone-project.git
   cd sky-cipher-drone-project
   ```
