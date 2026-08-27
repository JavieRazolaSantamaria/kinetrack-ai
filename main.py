import cv2
import time
from src.vision.pose_detector import PoseDetector
from src.kinematics.angles import extract_tkd_joint_angles
from src.kinematics.velocity import VelocityCalculator

def main():
    detector = PoseDetector()
    vel_calc = VelocityCalculator(assumed_height_m=1.75)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error al acceder a la webcam.")
        return

    prev_time = time.time()
    print("\n=== Kinetrack AI: Pipeline Biomecanico Completo (Angulos + Velocidad) ===")
    print("Pulsa 'q' en la ventana para salir.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        h, w, _ = frame.shape

        landmarks = detector.detect(frame)

        if landmarks:
            frame = detector.draw_skeleton(frame, landmarks)
            
            # 1. Calculo de angulos
            angles = extract_tkd_joint_angles(landmarks, min_visibility=0.5)
            
            # 2. Calculo de velocidades (m/s y km/h)
            velocities = vel_calc.compute_tkd_velocities(landmarks, curr_time, h, w)

            # Telemetria en pantalla
            elbow_r_txt = f"{int(angles['elbow_right'])} deg" if angles['elbow_right'] is not None else "N/A"
            v_wrist_r = velocities.get("wrist_right", {}).get("v_kmh", 0.0)
            v_foot_r = velocities.get("foot_right", {}).get("v_kmh", 0.0)

            cv2.putText(frame, f"Codo Der:     {elbow_r_txt}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Vel Muneca D: {v_wrist_r:.1f} km/h", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Vel Pie Der:  {v_foot_r:.1f} km/h", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # FPS
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Kinetrack AI - Biomecanica y Velocidad", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()