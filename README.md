# Autonomous UAV Path Optimization

## 🚀 Project Overview
This project focuses on **Autonomous UAV Path Optimization** using spatial and temporal constraints, enabling UAVs (Unmanned Aerial Vehicles) to navigate optimally through urban airspaces while avoiding collisions and ensuring timely deliveries.

## ⚙️ Prerequisites
Ensure the following software and libraries are installed:

### ✅ Required Software
- Python 3.8+
- Git
- pip (Python package manager)

### ✅ Required Python Libraries
Install dependencies using:
```bash
pip install -r requirements.txt
```
**`requirements.txt` content:**
```
numpy
matplotlib
networkx
scipy
tensorflow
```

## 🛠️ Setup & Run
Clone the repository and run the main script:
```bash
git clone https://github.com/your-username/autonomous-uav-path-optimizer.git
cd autonomous-uav-path-optimizer
python task_main.py
```

Optional arguments:
- `--realtime` : Enable live drone simulation.
- `--test` : Run predefined test scenarios.

## 📈 Output
- Optimized path visualized using Matplotlib.
- Log files of spatial violations and response time.
- AI-predicted risk zones, if enabled.

## 🧪 Testing
Run test suite:
```bash
python -m unittest discover tests
```
