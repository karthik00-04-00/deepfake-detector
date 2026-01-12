import yaml
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from src.model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_DIR = Path("data/processed/video_frames")
OUT_DIR = Path("data/processed/frame_preds")
CKPT = Path("outputs/models/best_baseline.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model = get_model(cfg)

    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    out_path = OUT_DIR / "frame_predictions.jsonl"
    with open(out_path, "w") as f:
        for video_dir in sorted(FRAMES_DIR.iterdir()):
            if not video_dir.is_dir():
                continue
            for clip_dir in sorted(video_dir.iterdir()):
                for frame_path in sorted(clip_dir.glob("*.jpg")):
                    img = Image.open(frame_path).convert("RGB")
                    x = transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logit = model(x).squeeze()
                        with torch.no_grad():
                            logits = model(x)
                            prob = torch.softmax(logits, dim=1)[0, 1].item()


                    rec = {
                        "video": video_dir.name,
                        "clip": clip_dir.name,
                        "frame": frame_path.name,
                        "prob_fake": prob,
                    }
                    f.write(json.dumps(rec) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
