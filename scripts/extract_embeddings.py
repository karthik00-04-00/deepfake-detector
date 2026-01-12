print("extract_embeddings.py loaded")

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from src.model import get_model
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_DIR = Path("data/processed/video_frames")
OUT_DIR = Path("data/processed/frame_embeddings")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def main():
    print("__main__ entered")
    print("Frames dir:", FRAMES_DIR.resolve())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = get_model(cfg)
    model.load_state_dict(
        torch.load("outputs/models/best_baseline.pth", map_location=DEVICE)
    )

    model.fc = torch.nn.Identity()
    model.to(DEVICE).eval()

    videos = list(FRAMES_DIR.iterdir())
    print("Videos found:", [v.name for v in videos])

    for video_dir in videos:
        if not video_dir.is_dir():
            continue

        video_out = OUT_DIR / video_dir.name
        video_out.mkdir(parents=True, exist_ok=True)

        clips = list(video_dir.iterdir())
        print(f"{video_dir.name}: {len(clips)} clips")

        for clip_dir in clips:
            feats = []

            for frame_path in sorted(clip_dir.glob("*.jpg")):
                img = Image.open(frame_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    emb = model(x).squeeze().cpu().numpy()

                feats.append(emb)

            if len(feats) > 0:
                feats = np.stack(feats)
                np.save(video_out / f"{clip_dir.name}.npy", feats)

        print(f"Processed {video_dir.name}")

if __name__ == "__main__":
    print("__main__ trigger")
    main()
