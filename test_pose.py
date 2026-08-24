import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Conexiones estándar del esqueleto humano (pares de índices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos
    (11, 23), (12, 24), (23, 24),                      # Tronco
    (23, 25), (24, 26), (25, 27), (26, 28),           # Piernas
    (27, 29), (28, 30), (29, 31), (30, 32)            # Pies
]

def main():
    # 1. Configurar opciones del PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    # 2. Iniciar webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam.")
        return

    prev_time = time.time()
    print("Iniciando captura con MediaPipe Tasks API. Pulsa 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Conversión a formato MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp monotónico en milisegundos
        frame_timestamp_ms = int(time.time() * 1000)

        # Inferencia en tiempo real
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        h, w, _ = frame.shape

        # Dibujar landmarks y conexiones si detecta una persona
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                # Dibujar conexiones (huesos)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    p1 = pose_landmarks[start_idx]
                    p2 = pose_landmarks[end_idx]
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Dibujar articulaciones (puntos)
                for lm in pose_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Medición de FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        cv2.putText(frame, f"FPS CPU: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("KineTrack AI - MediaPipe Modern API", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()