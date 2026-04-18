from src.inference.video_inference import load_models, predict_video

load_models(
    backbone_ckpt_path="outputs/models/best_baseline.pth",
    lstm_ckpt_path="outputs/models/video_lstm_last.pth"
)

video_path = "data/test_videos/real/real01.mp4"

result = predict_video(video_path)
print(result)