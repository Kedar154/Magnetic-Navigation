Markdown
# MagNav: GPS-Denied Aircraft Navigation 🧭✈️

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced magnetic navigation (MagNav) pipeline and real-time telemetry dashboard. This project demonstrates how aircraft can navigate using Earth's magnetic crustal anomalies as a "geological fingerprint" when GPS is jammed or unavailable.

## 🚀 Overview

Traditional Inertial Navigation Systems (INS) suffer from "drift" over time. This project solves that by fusing INS data with magnetic field observations via an **Extended Kalman Filter (EKF)**. By matching real-time magnetometer readings against the **EMAG2** anomaly grid, the system provides a zero-drift positioning solution.

### Key Features
* **Magnetic Compensation:** Implements a Tolles-Abeley model (Ridge Regression) to filter aircraft-induced magnetic interference.
* **Anomaly Extraction:** Isolates crustal signals by subtracting the IGRF (International Geomagnetic Reference Field) core model.
* **Neural Network Integration:** Uses an MLP Regressor to predict local magnetic variations.
* **Live Dashboard:** A custom-built Flask/Chart.js telemetry suite to visualize GPS vs. INS vs. MagNav trajectories.

* ## 🔍 Technical Deep Dive

### The Problem: GPS Vulnerability & INS Drift
In modern aviation, reliance on Global Navigation Satellite Systems (GNSS) is a single point of failure. While Inertial Navigation Systems (INS) provide an alternative, they suffer from **integration drift**, where small sensor errors accumulate into kilometers of positional uncertainty. 

### What We Did
We developed a secondary navigation layer that uses the Earth's magnetic field as a "map" to correct the INS.

1. **Magnetic Cleaning (The Tolles-Abeley Model):** An aircraft is a magnetically "loud" environment. We used **Ridge Regression** to model the 18 coefficients of the Tolles-Abeley equation, effectively removing magnetic interference caused by the aircraft's maneuvers (Pitch, Roll, and Yaw).

2. **Residual Anomaly Extraction:** By subtracting the **IGRF (Core Field)** from our compensated readings, we isolated the **Crustal Anomaly**. This small signal (~50-500 nT) is unique to specific geographic coordinates, allowing it to function as a landmark.

3. **Sensor Fusion via EKF:** We implemented an **Extended Kalman Filter (EKF)**. The EKF uses the INS to *predict* the next state and uses the observed magnetic anomaly to *update* and correct that prediction.

### Why We Did It
* **Strategic Autonomy:** Magnetic navigation is passive and unjammable, unlike GPS which can be spoofed or blocked.
* **Accuracy over Distance:** While INS error grows with time, MagNav error is bounded by the resolution of the magnetic map, providing a "truth" source that doesn't drift.
* **Weight & Cost:** This system utilizes existing magnetometer hardware found on most aircraft, requiring only a software-level upgrade to implement.

## 🛠️ The Tech Stack

* **Language:** Python 3.x
* **Math/Physics:** NumPy, SciPy (Signal Processing), h5py
* **ML:** Scikit-learn (Ridge Regression, MLP)
* **Backend:** Flask
* **Frontend:** HTML5, CSS3 (Syne & Space Mono typography), Chart.js


⚙️ Installation & Usage
Clone the repository:

Bash
git clone [https://github.com/Kedar154/Magnetic-Navigation.git](https://github.com/Kedar154/Magnetic-Navigation.git)
cd Magnetic-Navigation
Install dependencies:

Bash
pip install flask h5py numpy pandas scipy scikit-learn
Data Setup:
Ensure your .h5 flight files are located in ~/Desktop/data/ or update the DATA_DIR in app.py.

Launch the Dashboard:

Bash
python3 app.py
Open http://localhost:5050 in your browser.

📊 Results
The simulation demonstrates that while INS (Inertial) drift increases over time, the MagNav estimate stays tightly coupled with the GPS (Ground Truth), proving the efficacy of magnetic anomaly matching for long-range flight.

🤝 Contributors
Ram Upadhyay (IIT Indore CSE)
Kedar Hadnurkar(IIt Indore CSE)
