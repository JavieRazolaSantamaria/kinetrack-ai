import numpy as np
from src.filters.one_euro_filter import OneEuroFilter

class VelocityCalculator:
    def __init__(self, assumed_height_m: float = 1.75):
        self.assumed_height_m = assumed_height_m
        self.filters = {}         # {joint_id: (filt_x, filt_y, filt_z)}
        self.prev_positions = {}
        self.prev_timestamps = {}
        self.peak_velocities = {} # Guarda la velocidad maxima alcanzada

    def _get_filter(self, joint_id: int):
        if joint_id not in self.filters:
            self.filters[joint_id] = (
                OneEuroFilter(min_cutoff=1.2, beta=0.05),
                OneEuroFilter(min_cutoff=1.2, beta=0.05),
                OneEuroFilter(min_cutoff=1.2, beta=0.05)
            )
        return self.filters[joint_id]

    def _estimate_scale_factor(self, landmarks, frame_h: int, frame_w: int) -> float:
        sh_x = (landmarks[11].x + landmarks[12].x) / 2 * frame_w
        sh_y = (landmarks[11].y + landmarks[12].y) / 2 * frame_h
        hip_x = (landmarks[23].x + landmarks[24].x) / 2 * frame_w
        hip_y = (landmarks[23].y + landmarks[24].y) / 2 * frame_h

        torso_pixel_dist = np.hypot(sh_x - hip_x, sh_y - hip_y)
        if torso_pixel_dist < 10:
            return 1.0 / 500.0

        torso_real_m = self.assumed_height_m * 0.30
        return float(torso_real_m / torso_pixel_dist)

    def calculate_joint_velocity(self, joint_id: int, landmark, timestamp_s: float, 
                                 frame_h: int, frame_w: int, scale_m_per_px: float) -> dict:
        if landmark is None or (hasattr(landmark, 'visibility') and landmark.visibility < 0.5):
            return {"v_mps": 0.0, "v_kmh": 0.0, "v_peak_kmh": self.peak_velocities.get(joint_id, 0.0)}

        # 1. Posición cruda en píxeles
        raw_x = landmark.x * frame_w
        raw_y = landmark.y * frame_h
        raw_z = landmark.z * frame_w

        # 2. Suavizado con One Euro Filter
        fx, fy, fz = self._get_filter(joint_id)
        smooth_x = fx.filter(raw_x, timestamp_s)
        smooth_y = fy.filter(raw_y, timestamp_s)
        smooth_z = fz.filter(raw_z, timestamp_s)
        curr_pos = np.array([smooth_x, smooth_y, smooth_z])

        if joint_id not in self.prev_positions:
            self.prev_positions[joint_id] = curr_pos
            self.prev_timestamps[joint_id] = timestamp_s
            return {"v_mps": 0.0, "v_kmh": 0.0, "v_peak_kmh": 0.0}

        dt = timestamp_s - self.prev_timestamps[joint_id]
        if dt <= 1e-4:
            curr_v = self.peak_velocities.get(joint_id, 0.0)
            return {"v_mps": curr_v / 3.6, "v_kmh": curr_v, "v_peak_kmh": curr_v}

        # Desplazamiento y velocidad filtrada
        displacement_px = np.linalg.norm(curr_pos - self.prev_positions[joint_id])
        displacement_m = displacement_px * scale_m_per_px
        v_mps = displacement_m / dt
        v_kmh = v_mps * 3.6

        # Umbral de ruido estático (si es menor a 0.8 km/h, se considera quieto)
        if v_kmh < 0.8:
            v_kmh = 0.0
            v_mps = 0.0

        # Actualizar pico
        peak_v = max(self.peak_velocities.get(joint_id, 0.0), v_kmh)
        self.peak_velocities[joint_id] = peak_v

        self.prev_positions[joint_id] = curr_pos
        self.prev_timestamps[joint_id] = timestamp_s

        return {
            "v_mps": float(v_mps),
            "v_kmh": float(v_kmh),
            "v_peak_kmh": float(peak_v)
        }

    def compute_tkd_velocities(self, landmarks, timestamp_s: float, frame_h: int, frame_w: int) -> dict:
        if not landmarks:
            return {}
        scale = self._estimate_scale_factor(landmarks, frame_h, frame_w)
        return {
            "wrist_right": self.calculate_joint_velocity(16, landmarks[16], timestamp_s, frame_h, frame_w, scale),
            "wrist_left": self.calculate_joint_velocity(15, landmarks[15], timestamp_s, frame_h, frame_w, scale),
            "foot_right": self.calculate_joint_velocity(28, landmarks[28], timestamp_s, frame_h, frame_w, scale),
            "foot_left": self.calculate_joint_velocity(27, landmarks[27], timestamp_s, frame_h, frame_w, scale),
        }