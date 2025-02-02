# OceanView
OceanView is Team 537’s custom vision solution for the 2025 REAFSCAPE season. It is the core framework behind our autoscoring system, intelligently incorporating vision data to aid in fully autonomous robotic actions. To our knowledge, no other FRC team has developed a vision system as robust and integrated as this one.

## Overview
OceanView leverages cutting-edge technologies including DepthAI, OpenCV, and KDTree-based spatial search algorithms to detect and process objects such as algae and coral. This information is then used to determine scoring positions on the field. In addition, a Flask web server provides a live dashboard for vision feedback and system diagnostics.
Our system consists of multiple coordinated modules:
- **Vision Processing**: Captures color and depth frames via a custom DepthAI pipeline and processes them with OpenCV.
- **Map Management**: Uses branch configuration files and KDTree searches to locate target scoring positions.
- **Block Detection**: Differentiates between algae-blocked and coral-blocked positions.
- **Data Transmission**: Communicates with the RoboRIO using UDP (and optionally TCP) and serves status and video streams through a Flask server.
- **File Handling**: Saves images for debugging or post-match analysis.

## Features
- **DepthAI Pipeline**: Real-time acquisition of undistorted RGB and depth frames.
- **Advanced OpenCV Processing**: Custom image masking, contour detection, and 3D reprojection for precise object localization.
- **Dynamic Map Management**: Loads scoring positions from a YAML configuration file and computes branch positions based on alliance and reef geometry.
- **Obstacle Detection**: Uses KDTree data structures to quickly assess which scoring locations are blocked by algae or coral.
- **Data Transmission**: UDP-based packet transmission (with packet numbering) to the RoboRIO and a Flask-based web dashboard for live video and system health.
- **Cross-Platform Image Saving**: Easily capture and store frames during operation for debugging or review.
- **Comprehensive Diagnostics**: The Flask server provides system metrics (CPU, GPU, network, disk, and process information) and log outputs for real-time troubleshooting.

## Setup / Installation
### Requirements:
Ensure you have Python 3.8 or higher installed. The project relies on the following Python packages with the specified versions:
```
blinker==1.9.0
click==8.1.8
colorama==0.4.6
Cython==3.0.11
depthai==2.29.0.0
Flask==3.1.0
importlib_metadata==8.5.0
itsdangerous==2.2.0
Jinja2==3.1.5
MarkupSafe==3.0.2
meson==1.6.1
numpy==1.24.4
opencv-python==4.10.0.84
psutil==6.1.1
pynetworktables==2021.0.0
PyYAML==6.0.2
scipy==1.9.1
Werkzeug==3.1.3
zipp==3.21.0
```
### Installation Steps:
1. Clone the repository:
```bash
git clone https://github.com/team537/OceanView.git
cd OceanView
```
2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required packages:
```bash
pip install -r requirements.txt
```
4. Configure your system:
- Ensure that your DepthAI device is properly connected.
- Update the RoboRIO IP and port settings in main_controller.py and udp_sender.py if needed.
- Review and modify config/scoring_positions.yml to match your field and alliance configuration.
## Competition Sensor Calibration Procedure
Due to current limitations with HSV threshold tuning, we recommend using the [pseudopencv HSV Color Mask Tuner](https://pseudopencv.site/utilities/hsvcolormask/) for calibrating your sensor. This tool allows you to adjust HSV values until the masking accurately identifies the algae and coral objects in your environment.
## Usage
To launch OceanView, simply run the main controller script:
```bash
python src/main_controller.py
```
During operation:
- The system continuously captures color and depth frames.
- Processed frames with bounding boxes and 3D positions are sent to the web dashboard.
- Press k in the display window (if enabled) to capture and save input frames.
- Data is transmitted over UDP (and optionally via TCP) to the RoboRIO for integration into autonomous routines.

_Note: Some display functions (like cv2.imshow) may be disabled when running on resource-constrained devices (e.g., Raspberry Pi)._
## Data Transmission & Communication
OceanView uses several methods to ensure robust communication:
- UDP Sender: Packages vision data (available positions, algae-blocked positions, raw algae positions) with an incrementing packet number for reliability.
- TCP Receiver: Listens for incoming commands from the RoboRIO (e.g., to update the robot pose or request frame captures).
- Flask Server: Provides a live video stream, system health metrics, and diagnostic logs via a web interface. Access the dashboard by navigating to http://<device-ip>:<roborio_port>/ in your web browser.
## Map & Block Detection Management
- **Branch Manager & KDTreeManager**:
The branch manager loads reef configuration details (branch positions, level heights, alliance-specific transformations) from a YAML file. A KDTree is built for efficient spatial querying, ensuring that the nearest scoring positions are identified quickly.

- **Block Detection Manager**:
This module uses separate KDTree instances for detected algae and coral positions. It determines if a scoring location is **available**, **algae-blocked**, or **coral-blocked** (with coral-blocked locations being excluded from further processing).
## Additional Tools
- **Image Saver**:
Saves input, output, and depth frames to a dedicated folder for later review and debugging.
