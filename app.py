"""
Aplicación Flask de inicio de sesión por reconocimiento facial.

Versión extendida: Se agrega visualización opcional de *landmarks* faciales (ojos, nariz,
bo ca, cejas y aproximación de cachetes) usando dlib sin alterar la lógica existente
(de registro, almacenamiento en BD, comparación de rostros y manejo de sesión).

***IMPORTANTE***
1. Esta versión conserva TODAS las rutas y comportamientos originales.
2. Se añaden importaciones y funciones auxiliares **sin romper compatibilidad**.
3. Si no cuentas con el modelo de dlib (`shape_predictor_68_face_landmarks.dat`),
   las funciones de landmarks se desactivan silenciosamente (el sistema sigue funcionando).
4. Los landmarks sólo se dibujan en las ventanas de cámara de *registro* y en la
   captura única de *login* (cuando hay rostro). No se guardan ni afectan el encoding.

Para activar landmarks:
- Coloca el archivo `shape_predictor_68_face_landmarks.dat` en `models/` (o ajusta la ruta).
- Asegúrate de haber instalado dlib: `pip install dlib`.
"""

# =============================
# Importación de librerías necesarias
# =============================
from flask import Flask, render_template, request, redirect, session  # Módulos para crear la app web
import face_recognition  # Librería para reconocimiento facial
import cv2  # OpenCV, para capturar imágenes desde la cámara
import os  # Para trabajar con el sistema de archivos
import numpy as np  # Para operaciones numéricas y con matrices
import mysql.connector  # Para conectarse con MySQL
import pickle  # Para guardar y cargar datos de entrenamiento en archivo

# === NUEVO PARA LANDMARKS ===
try:
    import dlib  # Modelo de landmarks faciales (68 puntos)
except ImportError:  # Si dlib no está instalado, continuamos sin fallar
    dlib = None

# =============================
# Inicialización de la aplicación Flask
# =============================
app = Flask(__name__)
app.secret_key = 'clave_secreta'  # Clave para mantener sesiones seguras

# =============================
# Configuración de la base de datos
# =============================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Cambia esta línea si tu MySQL tiene contraseña
    'database': 'facelogin'  # Nombre de la base de datos
}

# =============================
# Configuración de predictor de landmarks (NUEVO)
# =============================
# Ubicación del archivo .dat (ajusta si está en otra ruta)
PREDICTOR_PATH = os.path.join('models', 'shape_predictor_68_face_landmarks.dat')

# Intentar cargar el predictor al iniciar; si falla, se deshabilitan landmarks.
_landmark_detector = None
if dlib is not None and os.path.exists(PREDICTOR_PATH):
    try:
        _landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)
    except Exception as e:  # noqa
        _landmark_detector = None

# =============================
# Utilidades de entrenamiento persistente
# =============================

def load_trained_faces():
    """Cargar rostros previamente entrenados desde archivo pickle."""
    if os.path.exists('trained_faces.pkl'):  # Verifica si existe el archivo
        with open('trained_faces.pkl', 'rb') as f:
            return pickle.load(f)  # Carga los datos en formato binario
    return {}  # Si no existe, regresa un diccionario vacío


def save_trained_faces(data):
    """Guardar rostros entrenados en archivo pickle."""
    with open('trained_faces.pkl', 'wb') as f:
        pickle.dump(data, f)  # Guarda los datos serializados


# Se cargan los rostros entrenados al iniciar el servidor
trained_faces = load_trained_faces()

# =============================
# Funciones auxiliares de Landmarks (NO rompen flujo si no hay modelo)
# =============================

def _shape_to_np(shape, dtype="int"):
    """Convierte un objeto dlib.full_object_detection a un arreglo NumPy (68x2)."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def _draw_poly(frame, pts, color, is_closed=True, thickness=1):
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=is_closed, color=color, thickness=thickness)


def _draw_text(frame, text, xy, color):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_facial_regions(frame, gray, face_rect_bgr):
    """Dibuja regiones (cejas, ojos, nariz, boca, cachetes aprox.) sobre el frame.

    Parameters
    ----------
    frame : np.ndarray (BGR)
    gray : np.ndarray (grayscale)
    face_rect_bgr : tupla (top, right, bottom, left) en coordenadas estilo face_recognition.
    """
    if _landmark_detector is None:
        return  # Landmarks deshabilitados

    top, right, bottom, left = face_rect_bgr
    rect = dlib.rectangle(left, top, right, bottom)
    shape = _landmark_detector(gray, rect)
    pts = _shape_to_np(shape)

    # Mapas de regiones según el modelo 68 puntos
    # Índices estándar:
    #  0-16 mandíbula
    # 17-21 ceja derecha
    # 22-26 ceja izquierda
    # 27-30 puente nariz
    # 30-35 base nariz
    # 36-41 ojo derecho
    # 42-47 ojo izquierdo
    # 48-59 labio externo
    # 60-67 labio interno

    # Cejas
    brow_r = pts[17:22]
    brow_l = pts[22:27]
    # Ojos
    eye_r = pts[36:42]
    eye_l = pts[42:48]
    # Nariz
    nose = pts[27:36]
    # Boca exterior (48-60) usamos 48:60 porque 60 es excluido => 48..59
    mouth = pts[48:60]

    # Cachetes (aprox): puntos mandibulares laterales + extremos de boca + lateral nariz
    # Izquierdo (desde observador): mandíbula 1-3 + punto nariz 31 + labio 48
    cheek_left = np.vstack([pts[1:4], pts[31], pts[48]])
    # Derecho: mandíbula 13-15 + nariz 35 + labio 54
    cheek_right = np.vstack([pts[13:16], pts[35], pts[54]])

    # Dibujar polígonos
    green = (0, 255, 0)
    blue = (255, 0, 0)
    magenta = (255, 0, 255)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)

    _draw_poly(frame, brow_r, green, False)
    _draw_poly(frame, brow_l, green, False)
    _draw_poly(frame, eye_r, blue, True)
    _draw_poly(frame, eye_l, blue, True)
    _draw_poly(frame, nose, magenta, False)
    _draw_poly(frame, mouth, cyan, True)
    _draw_poly(frame, cheek_left, yellow, True)
    _draw_poly(frame, cheek_right, yellow, True)

    # Etiquetas (usamos primeros puntos de cada región)
    _draw_text(frame, 'Cejas', tuple(brow_r[0]), green)
    _draw_text(frame, 'Ojo', tuple(eye_r[0]), blue)
    _draw_text(frame, 'Nariz', tuple(nose[0]), magenta)
    _draw_text(frame, 'Boca', tuple(mouth[0]), cyan)
    _draw_text(frame, 'Cachete', tuple(cheek_left[0]), yellow)
    _draw_text(frame, 'Cachete', tuple(cheek_right[0]), yellow)

    # También, si quieres ver TODOS los puntos (opcional, comentar si molesta)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


# =============================
# Ruta principal: Página de inicio
# =============================
@app.route('/')
def index():
    return render_template('index.html')  # Muestra la página principal


# =============================
# Ruta de registro: Captura y guarda rostros de nuevos usuarios
# =============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Obtiene datos del formulario
        name = request.form['name']
        surname = request.form['surname']
        password = request.form['password']
        access_key = request.form['access_key']

        # Validación de clave de acceso
        if access_key != '6473462':
            return "Clave incorrecta", 401

        # Crea carpeta para guardar imágenes del usuario
        folder_name = f"{name}_{surname}"
        folder_path = os.path.join("dataset", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Captura imágenes desde la cámara
        cam = cv2.VideoCapture(0)
        count = 0
        encodings = []

        while count < 10:
            ret, frame = cam.read()
            if not ret:
                continue

            # Convierte a RGB para reconocimiento
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)

            # Si se detecta solo un rostro
            if len(face_locations) == 1:
                encoding = face_recognition.face_encodings(rgb, face_locations)[0]
                encodings.append(encoding)
                count += 1

                # === NUEVO: Dibujar landmarks en la ventana de captura ===
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                draw_facial_regions(frame, gray, face_locations[0])

                # Guarda imagen capturada
                img_path = os.path.join(folder_path, f"{count}.jpg")
                cv2.imwrite(img_path, frame)

                # Muestra la cantidad de capturas en pantalla
                cv2.putText(frame, f"Captura {count}/10", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                # Si hay 0 o más de 1 rostros, avisar en pantalla (no cambia funcionalidad)
                cv2.putText(frame, "Coloca 1 rostro frente a la camara", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Capturando rostro", frame)

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera la cámara y cierra ventana
        cam.release()
        cv2.destroyAllWindows()

        # Si no se capturaron rostros válidos
        if len(encodings) == 0:
            return "No se pudieron capturar rostros válidos"

        # Calcula el promedio del encoding para representar al usuario
        average_encoding = np.mean(encodings, axis=0)

        # Inserta usuario en la base de datos
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, surname, password, access_key, image_path) VALUES (%s, %s, %s, %s, %s)",
                           (name, surname, password, access_key, folder_path))
            conn.commit()
        finally:
            # Cerrar conexiones aunque ocurra error
            try:
                cursor.close()
            except Exception:  # noqa
                pass
            try:
                conn.close()
            except Exception:  # noqa
                pass

        # Guarda el encoding en memoria y archivo
        trained_faces[f"{name} {surname}"] = average_encoding
        save_trained_faces(trained_faces)

        return "Registro exitoso"

    return render_template('register.html')  # Muestra formulario si método es GET


# =============================
# Ruta para inicio de sesión con reconocimiento facial
# =============================
@app.route('/login', methods=['POST'])
def login():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    # Verifica si se obtuvo un frame
    if not ret:
        return "Error de cámara"

    # Convertir a RGB (corrección: face_recognition espera RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta rostros en la imagen capturada
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        return "No se detectó rostro, intenta de nuevo"

    # Obtiene encoding del primer rostro detectado
    face_encoding = face_recognition.face_encodings(rgb, face_locations)[0]

    # === Opcional: mostrar una ventana con landmarks del frame capturado ===
    if len(face_locations) == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        draw_facial_regions(frame, gray, face_locations[0])
        cv2.imshow("Login - Vista de rostro", frame)
        cv2.waitKey(1000)  # 1 segundo para mostrar; ajusta o elimina si no quieres ventana
        cv2.destroyAllWindows()

    # Compara con los rostros registrados
    for name, encoding in trained_faces.items():
        match = face_recognition.compare_faces([encoding], face_encoding)[0]
        if match:
            session['user'] = name  # Guarda el nombre del usuario en sesión
            return redirect('/welcome')  # Redirige a página de bienvenida

    return "Rostro no reconocido, intenta nuevamente"


# =============================
# Ruta de bienvenida luego del inicio de sesión exitoso
# =============================
@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect('/')  # Redirige si no hay sesión activa
    return render_template('welcome.html', name=session['user'])  # Muestra bienvenida con el nombre


# =============================
# Ejecuta la app en modo debug
# =============================
if __name__ == '__main__':
    # Mostrar advertencia si landmarks deshabilitados
    if _landmark_detector is None:
        print("[ADVERTENCIA] No se cargó el modelo de landmarks. El reconocimiento facial funciona igual, pero no se mostrarán partes específicas del rostro.")
    else:
        print("[OK] Modelo de landmarks cargado. Se dibujarán cejas, ojos, nariz, boca y cachetes en las capturas.")
    app.run(debug=True)
