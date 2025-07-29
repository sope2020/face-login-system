import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Usa la webcam

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Dibuja un rectángulo de "detección facial" (simulado)
            height, width, _ = frame.shape
            cv2.rectangle(frame, (int(width*0.25), int(height*0.2)), (int(width*0.75), int(height*0.8)), (0, 255, 255), 2)

            # Convierte a JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Devuelve en formato de flujo de video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('login_with_face.html')

if __name__ == "__main__":
    app.run(debug=True)
