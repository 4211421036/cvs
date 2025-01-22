from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Gunakan kamera default
    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Proses frame di sini (misalnya, deteksi landmark wajah)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
