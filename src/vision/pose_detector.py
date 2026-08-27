import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    """
    Encapsula el modelo PoseLandmarker de MediaPipe Tasks para inferencia
    en tiempo real o sobre archivos de vídeo.
    """
    def __init__(self, model_path: str = "models/pose_landmarker_lite.task",
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en la ruta: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Conexiones estándar del esqueleto MediaPipe
        self.connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Brazos
            (11, 23), (12, 24), (23, 24),                      # Tronco
            (23, 25), (24, 26), (25, 27), (26, 28),           # Piernas
            (27, 29), (28, 30), (29, 31), (30, 32)            # Pies
        ]

    def detect(self, frame_bgr, timestamp_ms: int = None):
        """
        Procesa un fotograma BGR de OpenCV y devuelve la lista de landmarks.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def draw_skeleton(self, frame_bgr, landmarks, 
                      line_color=(0, 255, 0), point_color=(0, 0, 255)):
        """
        Dibuja los huesos y articulaciones sobre el fotograma BGR.
        """
        if landmarks is None:
            return frame_bgr

        h, w, _ = frame_bgr.shape

        # Dibujar líneas de conexión ósea
        for start_idx, end_idx in self.connections:
            p1, p2 = landmarks[start_idx], landmarks[end_idx]
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(frame_bgr, pt1, pt2, line_color, 2)

        # Dibujar puntos articulares
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 3, point_color, -1)

        return frame_bgr

    def close(self):
        """Libera los recursos del detector."""
        self.detector.close()