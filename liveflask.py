import cv2
from flask import Flask, Response, stream_with_context
import time

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Opens the default camera

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            time.sleep(0.1)  # Delay to limit frame rate (0.1~10 fps)
            frame = cv2.resize(frame, (640, 480)) # Resize the frame, Compression
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)