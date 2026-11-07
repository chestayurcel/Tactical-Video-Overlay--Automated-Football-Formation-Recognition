# ⚽ Tactical Video Overlay: Automated Football Formation Recognition from Match Footage

[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8m-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/ByteTrack-MOT-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official implementation and evaluation benchmark for the paper:  
**"Deep Learning-Based Tactical Analysis: Automated Formation Recognition from Football Match Footage"**  
*Nasy’an Taufiq Al Ghifari & Chesta Yurcel Zebada (Universitas Muhammadiyah Yogyakarta, 2026)*

---

## 📌 Project Overview

Tactical analysis in modern football increasingly demands fast, objective, and interpretable formation recognition directly from broadcast video feeds. However, single-camera footage (Tactical Cam) poses severe challenges: dynamic camera panning, perspective distortion, severe multi-player occlusions, and **partial observability** (players moving outside the frame).

This project introduces a lightweight, broadcast-based tactical analysis pipeline that generates a **Tactical Video Overlay** — a stable, 2D graphical augmentation rendered directly on match footage — displaying:
1. Individual player tracking ellipses with team classification.
2. Independent goalkeeper localization.
3. Clean, intra-line structural tactical mesh connecting players in the same tactical corridor.
4. Real-time dynamic formation triples (e.g., `4-3-3`, `4-4-2`, `3-5-2`) validated through a finite-state **Temporal Memory Mechanism**.

> **Terminology Note:** Following standard computer vision and HCI taxonomies (Billinghurst et al., 2014), the visual output is formally termed a **tactical video overlay** (2D graphical augmentation in video image space), rather than Augmented Reality (AR) in the strict sense, as it operates without 3D geometric scene registration or physical world interactivity.

---

## 🏛️ Pipeline Architecture

```
[ Broadcast Tactical Video ]
             │
             ▼
     ┌───────────────┐
     │ YOLOv8m (640) │ ──► Detects: player, goalkeeper, referee, ball
     └───────────────┘
             │
             ▼
     ┌───────────────┐
     │   ByteTrack   │ ──► Kalman Filter trajectory association (IoU = 0.6)
     └───────────────┘     Eliminates visual flickering & handles brief occlusions
             │
             ├──────────────────────────┐
             ▼                          ▼
      [ Referee Filter ]         [ Goalkeeper ]
      (Excluded from lines)      (Yellow AR indicator)
             │
             ▼
┌─────────────────────────┐
│ Hierarchical K-Means    │
│  ├─ Stage 1 (k=2)       │ ──► Team Separation via illumination-invariant chromaticity
│  │                            with centroid adaptation & 15-frame Track voting
│  └─ Stage 2 (k=3)       │ ──► Spatial corridor segmentation on X-axis (Def / Mid / Att)
└─────────────────────────┘     with 15-frame ordinal line voting
             │
             ▼
┌─────────────────────────┐
│ Temporal Memory Gate    │ ──► Four-State Machine: Update / Hold / Latch / Unknown
│  ├─ Valid Gate: N == 10 │
│  ├─ Valid Dictionary 𝒟 │ ──► Rejects tactical anomalies (e.g. 1-2-7, 2-2-6)
│  └─ Timeout (T_max=150) │ ──► Latches valid state; falls back to '---' (UNKNOWN) on stale
└─────────────────────────┘
             │
             ▼
[ Tactical Video Overlay Output ] ──► Ellipses + Structural Lines + Dynamic Formation Text
```

---

## ✨ Key Technical Contributions

1. **Lightweight Broadcast-Based Pipeline:** Operates on single-camera broadcast video without requiring multi-camera setups or computationally prohibitive Graph Neural Networks / Transformers. Measured throughput is **20.3 FPS** (including full video decode/encode I/O) on a laptop GPU (RTX 4060).
2. **Hierarchical Unsupervised K-Means:**
   - **Stage 1 (Team Clustering, $k=2$):** Extracts illumination-invariant normalized chromaticity $\frac{(r,g,b)}{r+g+b}$ from chest ROI ($25\%-65\%$ bbox height). Features smooth centroid adaptation (every 30 frames), collapse auto-retraining, and 15-frame Track ID majority voting.
   - **Stage 2 (Spatial Clustering, $k=3$):** Exploits horizontal density on foot coordinates to cluster outfield players into Defense $\to$ Midfield $\to$ Attack. Outperforms Nearest-Neighbor mesh (which creates chaotic criss-cross lines) and DBSCAN (which fails under non-uniform inter-line density).
3. **Finite-State Temporal Memory Mechanism:**
   Overcomes detector false negatives and partial camera observability through an explicit 4-state transition machine:
   - **Update ($\text{valid} \land f \neq \hat{y}$):** Updates displayed formation and resets stale timer $\tau \leftarrow 0$.
   - **Hold ($\text{valid} \land f = \hat{y}$):** Retains label, timer remains at 0.
   - **Latch ($\neg \text{valid} \land \tau < T_{max}$):** Holds previous valid formation while counting stale frames $\tau \leftarrow \tau + 1$.
   - **Unknown ($\tau \ge T_{max}$):** After timeout ($T_{max} = 150$ frames $\approx 5.0$ s), explicitly displays `---` (`UNKNOWN`) to prevent displaying obsolete tactical information.
4. **Valid Formation Dictionary ($\mathcal{D}$):**
   $$\mathcal{D} = \{(n_b, n_m, n_f) \mid n_b + n_m + n_f = 10, \; n_b \in \{3,4,5\}, \; n_f \in \{1,2,3,4\}\}$$
   Restricts formation updates strictly to recognized tactical structures, rejecting mathematically compliant yet tactically impossible clusters.

---

## 📊 Experimental Results & Ablation Study

Evaluated on **1,258 valid instances** ($|\mathcal{V}| = 1,258$) across **1,000 systematically sampled frames** (2,000 team-frame pairs) from high-definition broadcast footage against manual phase-based expert Ground Truth.

### Ablation Study (Quantitative Comparison)

| Configuration | Detection mAP@0.50 | Full Pipeline FPS | Formation Accuracy (Exact-Match) | Formation Accuracy ($\pm 1$ Player) | ULCR (Label Changes / 100 fr) | SLF (Stale Label Fraction) | Transition Detection Rate (TDR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8 Baseline** | 97.7% | 35.0 | N/A | N/A | N/A | N/A | N/A |
| **+ ByteTrack** | 97.7% | 31.0 | N/A | N/A | N/A | N/A | N/A |
| **+ K-Means (Instantaneous)** | 97.7% | 29.0 | 14.2% | 48.5% | 16.27 | 24.0% | 16.7% |
| **+ Temporal Memory (Ours)** | **97.7%** | **20.3** | **26.9%** | **75.0%** | **2.00** | **45.4%** | **29.6%** |

*Note: FPS for Configurations 1-3 reflects core algorithmic processing; Configuration 4 represents measured end-to-end throughput including high-resolution video read/write I/O.*

### Key Quantitative Findings:
* **Accuracy Improvement:** Temporal Memory boosts exact-match formation accuracy from **14.2% $\to$ 26.9%** (a **1.9× increase**, $+12.7$ percentage points).
* **Tolerance Accuracy:** When permitting a $\pm 1$ player ambiguity on line boundaries ($FA_{near}$), accuracy reaches **75.0%** (vs 48.5% without memory).
* **Flickering Elimination:** Unnecessary Label Change Rate (ULCR) drops by **8.1×** (from **16.27 down to 2.00 per 100 frames**), confirming visual and tactical stability.
* **Transition Responsiveness:** Detects tactical phase transitions with **0.0 frame delay** upon detection, increasing TDR from 16.7% to 29.6%.
* **Honest Design Cost:** The Stale Label Fraction (SLF) of 45.4% and Coverage of 98.1% are openly reported as the intentional stability cost of latching and conservative timeout abstention during partial observability.

---

## 🔍 Error Taxonomy Analysis

Analysis of all 919 erroneous instances ($|\mathcal{V}| = 1,258$) reveals that errors are dominated by conceptual conventions rather than system failures:

| Error Category | Description & Operational Definition | Count | % of $\|\mathcal{V}\|$ | Key Insight |
|---|---|:---:|:---:|---|
| **STALE** | Displayed label matches ground truth of previous tactical phase | 548 | 43.6% | Spatial vs. tactical phase gap: system tracks instantaneous spatial layout while GT reflects coach's intended phase role. |
| **GESER_1_PEMAIN** | Discrepancy of exactly one player between adjacent lines ($\Sigma\|\hat{y}-g\| = 2$) | 209 | 16.6% | Positional ambiguity (e.g. holding midfielder roaming between defense and midfield). |
| **STRUKTURAL** | Discrepancy $> 1$ player (not stale or flip) | 115 | 9.1% | Initial tracker warmup (frames 5-23) and multi-player goalmouth clumping. |
| **ABSTAIN** | System intentionally displays `---` (`UNKNOWN`) | 24 | 1.9% | Controlled timeout after $T_{max} = 150$ frames of invalid detections. |
| **FLIP_ORIENTASI** | Inverted formation order (e.g. 3-3-4 vs 4-3-3) | 23 | 1.8% | Side heuristic inversion during deep, sustained counter-attacks. |
| **Exact Match** | Exact semantic match with Ground Truth | 339 | **26.9%** | Correct structural classification across all 3 lines simultaneously. |

---

## ⚙️ Configuration Parameters

Key pipeline hyperparameters as reported in Table II of the paper for full experimental reproducibility:

| Module | Parameter | Value | Operational Purpose |
|---|---|---|---|
| **Input** | Video Resolution & Rate | 1280×720 @ 30 fps | Single-camera broadcast tactical feed |
| **Detection** | Architecture | YOLOv8m (fine-tuned) | Trained on Roboflow `football-tactical` |
| | Input Image Size (`imgsz`) | 640 | Preserves small/distant player features |
| | Confidence Threshold (`conf`) | 0.15 | Maximizes micro-object recall (FP filtered by tracker) |
| **Tracking** | Tracker | ByteTrack | Kalman Filter trajectory embedding |
| | IoU Threshold | 0.6 | Prevents ID switches during close duels |
| | Target Classes | player, goalkeeper | `referee` is filtered out from tactical calculation |
| **Team Clustering** | Feature Representation | Normalized Chromaticity | $(r,g,b)/(r+g+b)$ on $20\times 20$ chest ROI |
| | K-Means Stage 1 | $k = 2$ | `random_state=42`, `n_init=10` |
| | Centroid Adaptation | Every 30 frames | Smoothes illumination shifts |
| | Team Voting | 15 frames | Per-track majority voting with hysteresis |
| **Line Clustering** | K-Means Stage 2 | $k = 3$ | Unsupervised 1D clustering along X-axis |
| | Line Voting | 15 frames | Per-(team, Track ID) ordinal voting |
| **Memory** | Timeout ($T_{max}$) | 150 frames ($\approx 5$ s) | Latches last valid label; emits `---` when exceeded |
| | Seed | 42 | Ensures fully deterministic K-Means runs |

---

## 📁 Repository Structure

```
Tactical-Video-Overlay/
├── ar_taktik.py              # Main inference script (v5 with voting & temporal memory)
├── train_lokal.py            # Initial training script (YOLOv8m, 70 epochs)
├── finetune.py               # Fine-tuning script with referee class (30 epochs)
├── data.yaml                 # Dataset configuration (4 classes)
│
├── buat_gt_template.py       # Frame extraction & ground truth template generator
├── evaluate_formation.py     # Quantitative evaluation & ablation study calculator
├── investigasi_fa.py         # Error taxonomy & failure mode analyzer
│
├── gt_template.csv           # Ground truth benchmark annotations (1,000 frames)
├── hasil_evaluasi.csv        # Evaluated quantitative metrics summary
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Optimized Git exclusion rules
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA support (tested on RTX 4060, 16 GB RAM)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/chestayurcel/Tactical-Video-Overlay--Automated-Football-Formation-Recognition.git
cd Tactical-Video-Overlay--Automated-Football-Formation-Recognition

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Tactical Video Overlay
1. Place your target tactical video in the project directory as `video_utama.mp4` (or update `video_path` in [`ar_taktik.py`](file:///c:/Users/HP/Semester%206/AI%20Terapan/Football_AI_Project/ar_taktik.py)).
2. Ensure the fine-tuned model weights are located at:
   `runs/detect/runs/detect/Hasil_Finetune/Model_Dengan_Wasit-3/weights/best.pt`
3. Execute the pipeline:
   ```bash
   python ar_taktik.py
   ```
4. The augmented video will be generated as `tactical_formation.avi` and frame-by-frame logs stored in `log_formasi_per_frame.csv`.

### 4. Running Formation Evaluation Benchmark
To reproduce the ablation study and evaluate accuracy against the 1,000-frame ground truth dataset:
```bash
# Evaluate formation metrics
python evaluate_formation.py

# Inspect error taxonomy
python investigasi_fa.py
```

---

## ⚠️ Known Limitations & Future Work

* **Fixed $k=3$ Line Clustering:** The system models formations into three horizontal corridors (Defense, Midfield, Attack). Four-line formations (e.g., `4-2-3-1` or `4-1-4-1`) have their midfield layers consolidated ($n_m = 5$). Future work will investigate dynamic cluster validation (e.g., Silhouette Score or Gap Statistic).
* **Extreme High-Press Compression:** During whole-team high presses where only 1 defender remains in the defensive third, the $n_b \ge 3$ dictionary gate conservatively holds or emits `UNKNOWN`.
* **Perspective Distortions:** The pipeline operates on broadcast perspective without homography bird's-eye projection. Integrating planar homography or pitch keypoint registration will further refine line boundary assignments.

---

## 📖 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@article{alghifari2026deep,
  title={Deep Learning-Based Tactical Analysis: Automated Formation Recognition from Football Match Footage},
  author={Al Ghifari, Nasy’an Taufiq and Zebada, Chesta Yurcel},
  journal={Department of Information Technology, Universitas Muhammadiyah Yogyakarta},
  year={2026}
}
```

---

## 👨‍💻 Authors & Acknowledgements
* **Nasy'an Taufiq Al Ghifari** — *Universitas Muhammadiyah Yogyakarta* (nasyan.taufiq@umy.ac.id)
* **Chesta Yurcel Zebada** — *Universitas Muhammadiyah Yogyakarta* (chesta.yurcel.ft23@mail.umy.ac.id)

*Special thanks to Alfonso Secretaría Técnica for the open-source `football-tactical` dataset on Roboflow Universe, and the creators of YOLOv8 (Ultralytics) and ByteTrack.*
