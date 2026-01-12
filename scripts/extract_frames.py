import cv2
from pathlib import Path

RAW_DIR = Path("data/raw/videos")
OUT_DIR = Path("data/processed/video_frames")

BURST_SIZE = 5
STRIDE_SECONDS = 1.0

def extract_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError(f"Could not read FPS for {video_path.name}")

    stride = int(fps * STRIDE_SECONDS)
    video_out = OUT_DIR / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    clip_idx = 0
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            buffer = []

        buffer.append(frame)

        if len(buffer) == BURST_SIZE:
            clip_dir = video_out / f"clip_{clip_idx:03d}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            for i, f in enumerate(buffer):
                cv2.imwrite(str(clip_dir / f"frame_{i:03d}.jpg"), f)

            clip_idx += 1
            buffer = []

        frame_idx += 1

    cap.release()
    print(f"{video_path.name}: extracted {clip_idx} clips")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for video in RAW_DIR.glob("*.mp4"):
        extract_video(video)

if __name__ == "__main__":
    main()
