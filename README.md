# ⚽ AR Tactical Overlay: Automated Football Formation Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

## 📌 Project Overview
This repository contains the source code for an automated tactical football formation recognition system designed for broadcast footage. 

The system utilizes an **Augmented Reality (AR) Tactical Overlay** to visualize player positioning, team structure, and real-time dynamic formations (e.g., 4-3-3, 4-4-2). It combines state-of-the-art Deep Learning object detection with unsupervised spatial clustering and a rule-based memory system to handle complex occlusion and single-camera limitations.

> **Note:** This project was developed as a Final Project for the Applied Artificial Intelligence Course at Universitas Muhammadiyah Yogyakarta (UMY).

## ✨ Key Features
* **Robust Detection & Tracking:** Integrates **YOLOv8** for small-object player detection and **ByteTrack** to eliminate flickering and maintain continuous tracking IDs.
* **Unsupervised Team Segmentation:** Applies **K-Means Clustering (k=2)** on extracted chest Region of Interest (ROI) colors to automatically separate teams without manual labeling.
* **Tactical Line Mapping:** Uses spatial **K-Means Clustering (k=3)** on the X-axis coordinates to dynamically group players into three structural lines (Defender, Midfielder, Attacker).
* **Rule-Based Memory System:** Overcomes YOLOv8 *False Negatives* (due to motion blur or small object size) by implementing a latch operation that locks the displayed formation only when exactly 10 outfield players are detected.

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/football-ar-tactics.git](https://github.com/yourusername/football-ar-tactics.git)
   cd football-ar-tactics

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## 🚀 Usage
1. Create necessary directories: You must create the data and output directories manually before running the script.
2. Place your broadcast video inside the data/ folder (e.g., data/match_clip.mp4).
3. Run the main processing script:
   ```bash
   python ar_taktik.py --source data/match_clip.mp4 --save-path output/ar_result.avi

## 📚 Acknowledgements & References
Dataset: The model was trained using the football-tactical dataset provided by Alfonso Secretara Tecnica on Roboflow Universe.

Architecture: YOLO by Ultralytics | ByteTrack

Developed by Chesta Yurcel Zebada | 2026
