# ⚽ Tactical Video Overlay: Automated Football Formation Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

## 📌 Project Overview
This repository contains the source code for an automated tactical football formation recognition system designed for broadcast footage.

The system draws a **tactical video overlay** — a stable 2D graphical augmentation rendered on top of the source video — to visualize player positioning, team structure, and dynamic formations (e.g., 4-3-3, 4-4-2). It combines state-of-the-art Deep Learning object detection with unsupervised spatial clustering and a rule-based temporal memory system to handle complex occlusion and single-camera limitations.

> **Terminology note:** The output of this project is a *tactical video overlay* (2D video augmentation), not augmented reality in the strict sense: graphical elements are registered in image space on recorded footage, without 3-D registration or interactive coupling with the physical world.

> **Note:** This project was developed as a Final Project for the Applied Artificial Intelligence Course at Universitas Muhammadiyah Yogyakarta (UMY).

## ✨ Key Features
* **Robust Detection & Tracking:** Integrates **YOLOv8** for small-object player detection and **ByteTrack** to eliminate flickering and maintain continuous tracking IDs.
* **Unsupervised Team Segmentation:** Applies **K-Means Clustering (k=2)** on illumination-invariant chest Region of Interest (ROI) color features, with periodic centroid adaptation and per-track majority voting to stay stable under lighting changes and brief occlusions.
* **Tactical Line Mapping:** Uses spatial **K-Means Clustering (k=3)** on the X-axis coordinates (with per-track line voting) to dynamically group players into three structural lines (Defender, Midfielder, Attacker).
* **Rule-Based Temporal Memory:** Overcomes YOLOv8 *False Negatives* (due to motion blur or small object size) through a four-state machine (update / hold / latch / unknown): the displayed formation updates only when exactly 10 outfield players are detected and the triple belongs to a valid-formation dictionary; otherwise the last valid label is latched, and an explicit `---` is shown when invalid conditions persist beyond the timeout.

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/chestayurcel/Tactical-Video-Overlay--Automated-Football-Formation-Recognition.git
   cd Tactical-Video-Overlay--Automated-Football-Formation-Recognition

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## 🚀 Usage
1. Place your broadcast video in the project root as video_utama.mp4, or edit the video_path constant at the top of the main script.
2. Run the main processing script:
   ```bash
   python ar_taktik.py

## 📚 Acknowledgements & References
Dataset: The model was trained using the football-tactical dataset provided by Alfonso Secretara Tecnica on Roboflow Universe.

Architecture: YOLO by Ultralytics | ByteTrack

Developed by Chesta Yurcel Zebada | 2026
