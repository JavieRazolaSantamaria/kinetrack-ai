import numpy as np

def calculate_angle_2d(a, b, c) -> float:
    """Calcula el angulo 2D en grados para A - B - C (B es el vertice)."""
    if a is None or b is None or c is None:
        return None

    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_angle_3d(a, b, c) -> float:
    """Calcula el angulo 3D en grados para A - B - C (B es el vertice)."""
    if a is None or b is None or c is None:
        return None

    a = np.array(a[:3], dtype=np.float32)
    b = np.array(b[:3], dtype=np.float32)
    c = np.array(c[:3], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine_angle)))


def extract_tkd_joint_angles(landmarks, min_visibility: float = 0.5) -> dict:
    """
    Extrae los angulos articulares clave de Taekwondo.
    Si una articulacion tiene una visibilidad o presencia inferior a min_visibility,
    devuelve None en lugar de un angulo residual falso.
    """
    def get_valid_point(idx):
        lm = landmarks[idx]
        
        # Verificar visibilidad y presencia del landmark si estan disponibles
        if hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility < min_visibility:
            return None
        if hasattr(lm, 'presence') and lm.presence is not None and lm.presence < min_visibility:
            return None
            
        return [lm.x, lm.y, lm.z]

    angles = {
        # Miembros inferiores (patadas)
        "knee_left": calculate_angle_3d(get_valid_point(23), get_valid_point(25), get_valid_point(27)),
        "knee_right": calculate_angle_3d(get_valid_point(24), get_valid_point(26), get_valid_point(28)),
        "hip_left": calculate_angle_3d(get_valid_point(11), get_valid_point(23), get_valid_point(25)),
        "hip_right": calculate_angle_3d(get_valid_point(12), get_valid_point(24), get_valid_point(26)),
        
        # Miembros superiores (guardia)
        "elbow_left": calculate_angle_3d(get_valid_point(11), get_valid_point(13), get_valid_point(15)),
        "elbow_right": calculate_angle_3d(get_valid_point(12), get_valid_point(14), get_valid_point(16)),
    }

    return angles