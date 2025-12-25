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
