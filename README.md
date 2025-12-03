# Deepfake Detector — Image-based Prototype

## Project Overview
**Objective:** Build a small, complete image-based deepfake detection prototype that accepts a single cropped face image, returns a real/fake prediction with a confidence score, and exposes a FastAPI inference endpoint. A small DCGAN experiment is included as optional exploration.

## Scope (Phase 0 → Phase 1)
**In scope (initial):**
- Image-only detection (no video processing).
- Input: cropped face images (face detection/cropping part of preprocessing).
- Small dataset to start (~400 labeled images).
- Baseline CNN model (ResNet18 or a small custom CNN).
- Training, evaluation, and an inference API (FastAPI + uvicorn).
- Optional: small DCGAN under `src/gan/` for exploration.

**Out of scope (for now):**
- Real-time video detection, multimodal models, large-scale distributed training, web-scraping data collection, cloud deployment, and production CI/CD.

## Folder Structure
deepfake-detector/
├─ .venv/ # local virtual environment (ignored by git)
├─ requirements.txt # python deps
├─ .gitignore
├─ README.md
├─ data/
│ ├─ raw/ # raw/original images
│ └─ processed/ # preprocessed / cropped faces
├─ src/
│ ├─ data.py # dataset & preprocessing
│ ├─ model.py # model architectures
│ ├─ train.py # training script
│ ├─ eval.py # evaluation script
│ ├─ utils.py # helper utilities
│ └─ api/
│ └─ app.py # FastAPI inference app
├─ notebooks/ # experiments & exploration notebooks
├─ experiments/
│ └─ logs/ # training logs / tensorboard
├─ outputs/
│ ├─ figures/ # plots & visualizations
│ └─ models/ # saved model checkpoints
└─ docs/
└─ report/ # short project report
