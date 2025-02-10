import cv2
import torch
import numpy as np
import threading
from queue import Queue
from PIL import Image
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor
from collections import deque
import os

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "OPear/videomae-large-finetuned-UCF-Crime"
model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
processor = VideoMAEImageProcessor.from_pretrained(model_name)

# Frame preprocessing
transform = Compose([Resize((224, 224)), ToTensor()])

# Frame buffer & queue
frame_buffer = deque(maxlen=16)
frame_queue = Queue(maxsize=5)

# Skip frames to speed up
frame_skip = 2  
frame_count = 0
'''
#Use this path to test the live feed. replace '0' with the path to the video file in 'cap = cv2.VideoCapture(0)'.
video_path = r"VideaMAE\shooting.mp4"


if not os.path.exists(video_path):
    print("[ERROR] Video file not found! Check the file path.")
'''
def capture_frames():
    """Thread for capturing frames."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video reached or cannot read video!")
            break
        print("[INFO] Capturing frame...")  # Debugging step
        if not frame_queue.full():
            frame_queue.put(frame)  # Add only if queue is not full
        else:
            frame_queue.get()  # Remove old frame
            frame_queue.put(frame)  # Insert latest frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print("[INFO] Video capture thread ended!")

def process_video():
    """Thread for processing video and running inference."""
    global frame_buffer, frame_count
    while True:
        if frame_queue.empty():
            continue  # Wait for frames

        frame = frame_queue.get()
        frame_count += 1
        print(f"[INFO] Processing frame {frame_count}")  # Debugging

        if frame_count % frame_skip != 0:
            continue  # Skip frames

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(Image.fromarray(frame_rgb))
        frame_buffer.append(frame_tensor)

        if len(frame_buffer) == 16:
            video_list = list(frame_buffer)
            inputs = processor(video_list, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            predicted_class = outputs.logits.argmax(-1).item()
            label = model.config.id2label.get(predicted_class, "Unknown")
            print(f"[INFO] Predicted anomaly: {label}")  # Debugging

            # Overlay result
            cv2.putText(frame, f"Anomaly: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255) if label != "Normal" else (0, 255, 0), 2)
        #cv2.namedWindow("Simulated Live Anomaly Detection", cv2.WINDOW_NORMAL)
        # Display live video
        cv2.imshow("Live Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[INFO] Video processing thread ended!")

# Start threads
thread1 = threading.Thread(target=capture_frames)
thread2 = threading.Thread(target=process_video)

thread1.start()
thread2.start()

thread1.join()
thread2.join()
