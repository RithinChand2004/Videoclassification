import os
import av
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor

np.random.seed(0)

def read_video_pyav(container, indices):
    """Reads selected video frames based on provided indices."""
    frames = []
    container.seek(0)
    total_frames = sum(1 for _ in container.decode(video=0))
    print(f"[INFO] Total Frames in Video: {total_frames}")
    print(f"[INFO] Sampled Indices: {indices}")

    if not indices.size:
        raise ValueError("[ERROR] Frame indices are empty. Check frame sampling logic.")

    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    
    if not frames:
        print("[WARNING] No frames extracted with PyAV. Trying OpenCV instead...")
        return read_video_opencv(file_path, indices)

    return np.stack(frames)

def read_video_opencv(file_path, indices):
    """Fallback method to read video frames using OpenCV."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Unable to open video file: {file_path}")

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        raise ValueError("[ERROR] No frames extracted using OpenCV. Video may be corrupt.")
    
    return np.stack(frames)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """Samples valid frame indices within the correct range."""
    if seg_len <= clip_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len, dtype=int)
    else:
        end_idx = np.random.randint(clip_len, seg_len)
        start_idx = max(0, end_idx - clip_len)
        indices = np.linspace(start_idx, end_idx - 1, num=clip_len, dtype=int)
    
    indices = np.clip(indices, 0, seg_len - 1)  # Ensure indices are within video length
    print(f"[INFO] Corrected Sampled Indices: {indices}")
    return indices

# File path for video file
file_path = r"VideaMAE\Robbery035_x264.mp4"

# Open video container
try:
    container = av.open(file_path)
    video_stream = container.streams.video[0]
    seg_len = video_stream.frames if video_stream.frames > 0 else int(cv2.VideoCapture(file_path).get(cv2.CAP_PROP_FRAME_COUNT))
except Exception as e:
    raise RuntimeError(f"[ERROR] Could not open video: {e}")

print(f"[INFO] Corrected Segment Length: {seg_len}")

# Sample and read video frames
indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)
video = read_video_pyav(container, indices)

# Transform video frames
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])
video = torch.stack([transform(Image.fromarray(frame)) for frame in video])  # (T, C, H, W)
video_list = [frame for frame in video.squeeze(0)]  # Convert (1, C, T, H, W) → (T, C, H, W) → List[Tensors]

print(f"[INFO] Video Tensor Shape: {video.shape}")

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "OPear/videomae-large-finetuned-UCF-Crime"
model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
processor = VideoMAEImageProcessor.from_pretrained(model_name)

# Process input
inputs = processor(video_list, return_tensors="pt", do_rescale=False)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"[INFO] Processed Inputs Shape: {inputs['pixel_values'].shape}")

# Predict
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()

# Decode class label
id2label = model.config.id2label

print("========================================")
print(f"[RESULT] Segment Length: {seg_len}")
print(f"[RESULT] Sampled Indices: {indices}")
print(f"[RESULT] Predicted class: {id2label.get(predicted_class, 'Unknown')}")
print("========================================")
