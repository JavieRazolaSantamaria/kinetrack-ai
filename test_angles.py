import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Conexiones estándar del esqueleto
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos
    (11, 23), (12, 24), (23, 24),                      # Tronco
    (23, 25), (24, 26), (25, 27), (26, 28),           # Piernas
    (27, 29), (28, 30), (29, 31), (30, 32)            # Pies
]

def calculate_angle(a, b, c):
    """Calcula el ángulo en grados formado por A -> B -> C (B es el vértice)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))

def main():
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
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam.")
        return

    prev_time = time.time()
    print("\n=== Kinetrack AI: Test de Angulos Activo ===")
    print("Pulsa 'q' sobre la ventana de vídeo para salir.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        h, w, _ = frame.shape

        # Valores por defecto
        angle_r = 0
        angle_l = 0

        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                # 1. Dibujar esqueleto (huesos verdes y puntos rojos)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    p1 = pose_landmarks[start_idx]
                    p2 = pose_landmarks[end_idx]
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                for lm in pose_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                # 2. Brazo Derecho: 12 (Hombro), 14 (Codo), 16 (Muñeca)
                s_r = [pose_landmarks[12].x, pose_landmarks[12].y]
                e_r = [pose_landmarks[14].x, pose_landmarks[14].y]
                w_r = [pose_landmarks[16].x, pose_landmarks[16].y]
                angle_r = calculate_angle(s_r, e_r, w_r)

                # 3. Brazo Izquierdo: 11 (Hombro), 13 (Codo), 15 (Muñeca)
                s_l = [pose_landmarks[11].x, pose_landmarks[11].y]
                e_l = [pose_landmarks[13].x, pose_landmarks[13].y]
                w_l = [pose_landmarks[15].x, pose_landmarks[15].y]
                angle_l = calculate_angle(s_l, e_l, w_l)

                # 4. Dibujar texto dinámico sobre el codo derecho
                codo_x, codo_y = int(e_r[0] * w), int(e_r[1] * h)
                cv2.putText(frame, f"{int(angle_r)} deg", (codo_x + 15, codo_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Medición de FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        # PANEL FIJO DE INFORMACIÓN (Esquina superior izquierda)
        cv2.putText(frame, f"FPS CPU: {int(fps)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Codo Der: {int(angle_r)} deg", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Codo Izq: {int(angle_l)} deg", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("KineTrack AI - Biomecanica y Angulos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()