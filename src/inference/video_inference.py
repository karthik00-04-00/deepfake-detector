import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from torch.nn.utils.rnn import pack_padded_sequence

from src.video_model_video import VideoLSTM


# -------------------------------------------------
# Device
# -------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Cached Models (loaded once)
# -------------------------------------------------
_backbone = None
_lstm = None


# -------------------------------------------------
# Load Models
# -------------------------------------------------
def load_models(backbone_ckpt_path, lstm_ckpt_path):
    global _backbone, _lstm

    if _backbone is None:
        backbone = models.resnet18(pretrained=False)

        # IMPORTANT: match classifier shape to checkpoint
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)

        state_dict = torch.load(backbone_ckpt_path, map_location=_device)
        backbone.load_state_dict(state_dict)

        # Now remove classifier head for embeddings
        backbone.fc = nn.Identity()

        backbone.eval()
        backbone.to(_device)
        _backbone = backbone

    if _lstm is None:
        model = VideoLSTM(input_dim=512, hidden_dim=256)

        model.load_state_dict(
            torch.load(lstm_ckpt_path, map_location=_device)
        )

        model.eval()
        model.to(_device)
        _lstm = model


# -------------------------------------------------
# Frame Extraction
# -------------------------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


# -------------------------------------------------
# Preprocessing (must match training)
# -------------------------------------------------
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# Video Prediction
# -------------------------------------------------
@torch.no_grad()
def predict_video(video_path, mode="normal"):
    if _backbone is None or _lstm is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    frames = extract_frames(video_path)

    if len(frames) == 0:
        raise ValueError("No frames extracted from video.")

    embeddings = []

    for frame in frames:
        x = _transform(frame).unsqueeze(0).to(_device)
        emb = _backbone(x)
        embeddings.append(emb.squeeze(0))

    embeddings = torch.stack(embeddings)

    # last_50 rule
    T = embeddings.size(0)
    start_idx = T // 2
    embeddings = embeddings[start_idx:]

    if mode == "shuffle":
        idx = torch.randperm(embeddings.size(0))
        embeddings = embeddings[idx]
    elif mode == "reverse":
        embeddings = torch.flip(embeddings, dims=[0])

    if embeddings.size(0) == 0:
        raise ValueError("Sequence too short after last_50 rule.")

    embeddings = embeddings.unsqueeze(0)

    lengths = torch.tensor([embeddings.size(1)], device=_device)

    logit = _lstm(embeddings, lengths)
    prob = torch.sigmoid(logit).item()

    label = "fake" if prob > 0.5 else "real"

    return {
        "label": label,
        "confidence": prob,
        "frames_used": embeddings.size(1)
    }