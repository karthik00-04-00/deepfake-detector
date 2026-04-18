import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

from src.model import get_resnet18


# -------------------------
# Must match training config
# -------------------------
IMAGE_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CHECKPOINT_PATH = "outputs/models/best_baseline.pth"

DEVICE = torch.device("cpu")


# -------------------------
# Deterministic preprocessing
# -------------------------
_base_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# -------------------------
# Load Model Once (Cached)
# -------------------------
_model = None


def _load_model():
    global _model

    if _model is not None:
        return _model

    model = get_resnet18(
        num_classes=2,
        pretrained=False,
        finetune_layer4=False
    )

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


# -------------------------
# TTA Inference
# -------------------------
def _predict_tensor(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        fake_prob = probs[:, 1]
    return fake_prob


def predict_image(image_path: str) -> dict:
    """
    Deterministic image inference with simple TTA.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    model = _load_model()

    # Original
    tensor_orig = _base_preprocess(image).unsqueeze(0).to(DEVICE)

    # Horizontal flip
    image_flip = transforms.functional.hflip(image)
    tensor_flip = _base_preprocess(image_flip).unsqueeze(0).to(DEVICE)

    # Stack both for batch inference
    batch = torch.cat([tensor_orig, tensor_flip], dim=0)

    fake_probs = _predict_tensor(model, batch)

    # Average TTA probabilities
    final_prob = fake_probs.mean().item()

    label = "fake" if final_prob >= 0.5 else "real"

    return {
        "label": label,
        "confidence": float(final_prob)
    }