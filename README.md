# Deepfake Detector — Image-based Deepfake Detection System

## Project Overview
This project focuses on building an image-based deepfake detection system that classifies cropped face images as **real** or **fake** using deep learning. The goal is to design a clean, reproducible pipeline starting from a strong baseline model and progressively improving it through fine-tuning and future extensions.

The project is implemented in phases, following a structured machine learning workflow rather than jumping directly to complex models.

---

## Current Scope
- Image-based deepfake detection (faces only)
- Input: cropped face images
- Dataset: FaceForensics++ (cropped faces)
- Model: ResNet18 (ImageNet pretrained)
- GPU-accelerated training (CUDA)
- Evaluation using validation AUC and accuracy
- Planned FastAPI inference endpoint

---

## Project Phases (High-Level)
- **Phase 0:** Planning, environment setup, GPU verification  
- **Phase 1:** Dataset preparation and preprocessing  
- **Phase 1.0:** Sanity checks and pipeline validation  
- **Phase 2:** Image-based deepfake detection  
  - **Phase 2.1:** Frozen ResNet18 baseline  
  - **Phase 2.2:** Partial fine-tuning of deeper layers *(in progress)*  
- **Phase 3 (Planned):** Model robustness and improvements  
- **Phase 4 (Planned):** Video-based deepfake detection  

Detailed implementation and results are documented separately.

---

## Baseline Results (Phase 2.1)
- Architecture: ResNet18 (pretrained on ImageNet)
- Training strategy: Frozen backbone, trainable classifier head
- Dataset: FaceForensics++ (cropped face images)
- Validation AUC: **~0.96**

This baseline serves as a reference point for further fine-tuning and model improvements.

---

## Folder Structure
deepfake-detector/
├─ data/
│ └─ processed/ # processed face images (ignored by git)
├─ src/
│ ├─ data.py # dataset loader
│ ├─ model.py # model definitions
│ ├─ train.py # training script
│ ├─ eval.py # evaluation script
│ └─ api/ # FastAPI app (planned)
├─ configs/ # training configuration files
├─ outputs/
│ └─ models/ # saved model checkpoints
└─ docs/
└─ report/ # detailed project documentation


---

## How to Run (Baseline Training)
```bash
python -m src.train --config configs/baseline.yaml

Temporal Deepfake Detection (Video-Level Extension)

While image-based deepfake detection provides a strong baseline, it is fundamentally limited by frame-level cues. Many modern deepfakes appear visually convincing in individual frames but exhibit temporal inconsistencies when analyzed over longer durations.

To address this limitation, this project was extended to video-level deepfake detection by explicitly modeling temporal structure across frames.

Motivation

Image-only models saturate quickly (high AUC, limited generalization insight)

Deepfakes often fail cumulatively over time, not instantaneously

Temporal coherence provides an orthogonal signal to spatial artifacts

The goal of the video extension is not merely higher accuracy, but understanding where and when deepfakes break.

Video Pipeline Overview
Data Flow

Raw videos are organized as real/fake

Videos are split strictly at the video level (no frame leakage)

Frames are extracted into short clips

A frozen ResNet18 backbone converts frames into 512-D embeddings

Embeddings are concatenated into full video sequences

A temporal model operates over the entire video trajectory

This design ensures:

No information leakage

Reproducible temporal experiments

Clear separation between spatial and temporal modeling

Temporal Modeling Approach
Frame Embeddings

Backbone: ResNet18 (frozen)

Output: 512-D embedding per frame

Classifier head removed for temporal modeling

Temporal Model

Architecture: single-layer LSTM

Input: variable-length sequence of frame embeddings

Output: video-level real/fake prediction

This setup isolates temporal signal without confounding it with spatial fine-tuning.

Experimental Findings & Analysis (Core Results)

All experiments use:

Identical embeddings

Identical splits

Identical model architecture

Only temporal structure is modified

Phase A — Video-Level Temporal Modeling

Full video sequences used

Model rapidly overfits the training video

Validation loss diverges (expected with minimal data)

Confirms existence of a strong global temporal signal

Phase B.1 — Shuffle Test (Order Sensitivity)

Experiment: Frame order randomly permuted

Finding:

Training behavior remains largely unchanged

Model still learns effectively

Interpretation:

Exact temporal order is not critical at low data scale

Model relies on cumulative temporal statistics, not strict ordering

Phase B.2 — Reverse-Time Test (Directionality)

Experiment: Entire video sequence reversed

Finding:

Higher initial loss

Slower early convergence

Learning remains possible

Interpretation:

Temporal direction introduces friction

Signal is not purely time-invariant

Direction matters, but is secondary to temporal coverage

Phase B.3 — Truncation Test (Time-to-Failure)

Experiments:

First 25% of video

First 50% of video

Last 50% of video

Key Findings:

First 25% → weakest signal, hardest to learn

First 50% → recovers most of the temporal signal

Last 50% → strong signal, comparable to full video

Interpretation:

Deepfake artifacts emerge progressively over time rather than appearing immediately.

This demonstrates that temporal exposure, not isolated frames, is the dominant detection factor.

Key Insights (Consolidated)

Image-only models saturate quickly and miss temporal failures

Deepfake artifacts accumulate over time

Temporal presence matters more than exact ordering

Temporal direction introduces resistance but is not dominant

Reliable detection often requires observing at least half of a video

Video-level modeling is fundamentally more explanatory than frame-level detection

Limitations

Extremely small number of videos (overfitting expected)

No stable AUC metrics at video level yet

LSTM used as a minimal temporal model

Results focus on mechanism understanding, not benchmark performance

Future Work

Increase number of videos for statistically meaningful metrics

Replace LSTM with:

Bidirectional LSTM

Temporal attention

Transformer-based temporal models

Analyze per-video confidence trajectories

Add temporal explainability visualizations

Deploy video-level inference via FastAPI

Ethics & Responsible Use

This project is intended for:

Research

Education

Forensic analysis

Deepfake detection systems can be misused or overtrusted.
All results should be interpreted with caution and contextual awareness.
