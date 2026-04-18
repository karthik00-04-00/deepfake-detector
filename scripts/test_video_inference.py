from src.inference.video_inference import load_models, predict_video

VIDEO_PATH = "data/test_videos/clip.mp4"

load_models(
    backbone_ckpt_path="outputs/models/best_baseline.pth",
    lstm_ckpt_path="outputs/models/video_lstm_last.pth"
)

result = predict_video(VIDEO_PATH)
print(result)