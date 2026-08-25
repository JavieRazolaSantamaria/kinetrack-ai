import numpy as np

def calculate_angle_2d(a list  np.ndarray, b list  np.ndarray, c list  np.ndarray) - float
    
    Calcula el ángulo en grados formado por tres puntos 2D (A - B - C),
    donde B es el vértice articular.
    
    a = np.array(a[2], dtype=np.float32)
    b = np.array(b[2], dtype=np.float32)
    c = np.array(c[2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba  1e-6 or norm_bc  1e-6
        return 0.0

    cosine_angle = np.dot(ba, bc)  (norm_ba  norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_angle_3d(a list  np.ndarray, b list  np.ndarray, c list  np.ndarray) - float
    
    Calcula el ángulo en grados en el espacio 3D (x, y, z) entre los vectores BA y BC.
    
    a = np.array(a[3], dtype=np.float32)
    b = np.array(b[3], dtype=np.float32)
    c = np.array(c[3], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba  1e-6 or norm_bc  1e-6
        return 0.0

    cosine_angle = np.dot(ba, bc)  (norm_ba  norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine_angle)))


def extract_tkd_joint_angles(landmarks list) - dict
    
    Extrae los ángulos articulares clave de Taekwondo a partir de la lista de 33 landmarks.
    
    Índices estándar MediaPipe
    - 11 Hombro Izq, 13 Codo Izq, 15 Muñeca Izq
    - 12 Hombro Der, 14 Codo Der, 16 Muñeca Der
    - 23 Cadera Izq, 25 Rodilla Izq, 27 Tobillo Izq
    - 24 Cadera Der, 26 Rodilla Der, 28 Tobillo Der
    
    def to_coords(idx)
        lm = landmarks[idx]
        return [lm.x, lm.y, lm.z]

    angles = {
        # Miembros inferiores (patadas)
        knee_left calculate_angle_3d(to_coords(23), to_coords(25), to_coords(27)),
        knee_right calculate_angle_3d(to_coords(24), to_coords(26), to_coords(28)),
        hip_left calculate_angle_3d(to_coords(11), to_coords(23), to_coords(25)),
        hip_right calculate_angle_3d(to_coords(12), to_coords(24), to_coords(26)),
        # Miembros superiores (guardia)
        elbow_left calculate_angle_3d(to_coords(11), to_coords(13), to_coords(15)),
        elbow_right calculate_angle_3d(to_coords(12), to_coords(14), to_coords(16)),
    }

    return angles