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
