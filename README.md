# STAMP Parking CV 🅿️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **[🇷🇺 Read in Russian (Читать по-русски)](README_RU.md)**

A comprehensive Computer Vision (CV) system for monitoring parking lot occupancy, developed for the **STAMP ParkCloud** platform.

This project implements a scalable, multi-camera pipeline that determines the real-time occupancy of parking spots using both Deep Learning (YOLOv8) and traditional Feature-Based computer vision techniques.

## ✨ Key Features

- **Scalable Architecture**: Designed to handle 100+ parking spots simultaneously.
- **Multi-Camera Support**: Aggregates data from multiple overlapping camera views (default 4 cameras) to eliminate blind spots and improve accuracy.
- **Camera Work Zones**: Intelligent filtering based on predefined reliable detection zones for each camera.
- **Confidence Scoring (0-100%)**: Replaces binary occupied/free status with a granular confidence percentage.
- **Hybrid Detection Engine**:
  - `YOLO Mode`: Uses a pre-trained YOLOv8 model for vehicle detection.
  - `Feature Mode`: Analyzes edge density and color variance within the parking spot region of interest (excellent for synthetic/IR cameras).
  - `Hybrid Mode`: Combines both approaches for maximum reliability.
- **Interactive Web GUI**: A modern, dark-themed dashboard to visualize real-time occupancy, camera coverage, and system statistics.

---

## 🏗️ System Architecture

1. **Data Generation (`generate_test_data.py`)**: Creates a synthetic environment with 200 parking spots, 4 cameras, and various occupancy scenarios (15%, 55%, 90%).
2. **Calibration (`src/calibration.py`)**: Handles perspective transformation (homography) between camera views and the top-down Bird's-Eye View (BEV). Connects GeoJSON markup to camera pixels.
3. **Detection (`src/detection.py`)**: Analyzes images, calculates occupancy percentages, and handles multi-camera voting algorithms (Weighted Average, Majority Vote, Max Confidence).
4. **Pipeline (`src/pipeline.py`)**: The orchestrator. Loads data, runs detection across all cameras, aggregates results, and saves the final JSON outputs required by the STAMP specification.
5. **Web API & GUI (`gui_server.py` + `web/`)**: Serves the interactive dashboard and REST API for external integrations.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-org/stamp-parking-cv.git
cd stamp-parking-cv
pip install -r requirements.txt
```

### 2. Generate Test Data

First, generate the synthetic parking lot (200 spots), camera calibrations, and test scenarios:

```bash
python generate_test_data.py
```
*This will populate the `data/` directory with GeoJSON, calibration files, and test images.*

### 3. Run the Pipeline

Run the full multi-camera pipeline on Test 1 (55% occupancy) using the Feature-based method and compare it with the Ground Truth:

```bash
python src/pipeline.py --park_idx 1 --test_idx 1 --multi-camera --mode feature --compare-gt
```

**Pipeline Arguments:**
- `--park_idx`: Parking lot ID (default: 1)
- `--test_idx`: Test scenario ID (1, 2, or 3)
- `--camera_idx`: Specific camera to run (if not using `--multi-camera`)
- `--multi-camera`: Run all available cameras and aggregate the results
- `--mode`: Detection mode (`feature`, `yolo`, or `hybrid`)
- `--strategy`: Aggregation strategy (`weighted_avg`, `majority_vote`, `max_confidence`)
- `--compare-gt`: Print Precision/Recall/F1 metrics against the generated ground truth
- `--no-visualize`: Skip generating static PNG visualizations in `results/`

### 4. Launch the Web GUI

To start the interactive dashboard:

```bash
python gui_server.py
```
Open your browser and navigate to: **http://localhost:8080**

---

## 📄 Output Data Formats

The system generates outputs in strict adherence to the STAMP technical specification.

### Aggregated Result (`results/result_1_1.json`)

```json
{
  "params": {
    "park_idx": 1,
    "calibrate_idx": "multi",
    "cameras": [1, 2, 3, 4],
    "strategy": "weighted_avg"
  },
  "result": {
    "1": { "detected": true },
    "2": { "detected": false }
  }
}
```

### Detailed Result (`results/detailed_1_1_aggregated.json`)

Contains extended metrics for debugging, including the granular `occupancy_pct`.

```json
{
  "spots": {
    "1": {
      "detected": true,
      "confidence": 0.95,
      "occupancy_pct": 95,
      "num_cameras": 2,
      "method": "aggregated_weighted_avg"
    }
  }
}
```

---

## 🧪 Testing Results

Using the `feature` detection mode on synthetic data (Test 1, 55% occupancy):

- **Accuracy**: 92.5%
- **Precision**: 100.0%
- **Recall**: 86.4%
- **F1-Score**: 92.7%

*Note: YOLOv3/v8 models typically struggle with synthetic 2D boxes. The feature-based approach was implemented specifically to provide high accuracy on the generated test data, while YOLO remains available for real-world photographic feeds.*
