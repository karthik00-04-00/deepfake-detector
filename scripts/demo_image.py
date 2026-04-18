from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "outputs/models/best_baseline.pth"
IMAGE_PATH = "data/processed/ffpp/test/real/000_f3.jpg" # change to any test image

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
x = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    fake_conf = probs[0][1].item()

label = "fake" if fake_conf > 0.5 else "real"

print({
    "label": label,
    "confidence": fake_conf
})