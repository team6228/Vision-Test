import cv2
import numpy as np
from pupil_apriltags import Detector

# [TODO]
# Look more to calibrating stuff
# Fix focal lengths, optical center
# Fix april tag size
# Add config.py
# [AND MOST IMPORTENTLY]
# Add networktables and test the latency

# Setup camera
cap = cv2.VideoCapture(0)  # Change index if using another camera

# Define camera matrix & distortion coefficients
# (Calibrate your camera to get real values)
fx, fy = 600, 600  # Focal lengths
cx, cy = 320, 240  # Optical center (assumes 640x480 resolution)
camera_params = (fx, fy, cx, cy)

# Define AprilTag Detector (Use tag36h11 for FRC)
at_detector = Detector(families='tag36h11')

# Define real-world AprilTag size (meters)
tag_size = 0.1524  # 6 inches (FRC Standard)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (AprilTags work best in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    for tag in tags:
        if tag.tag_id == 1:  # Only track AprilTag ID 1
            # Extract pose
            tvec = tag.pose_t.flatten()  # Translation (x, y, z)
            rvec = tag.pose_R  # Rotation matrix (3x3)

            # Convert rotation matrix to Euler angles (yaw, pitch, roll)
            yaw = np.arctan2(rvec[1, 0], rvec[0, 0]) * 180 / np.pi
            pitch = np.arctan2(-rvec[2, 0], np.sqrt(rvec[2, 1]**2 + rvec[2, 2]**2)) * 180 / np.pi
            roll = np.arctan2(rvec[2, 1], rvec[2, 2]) * 180 / np.pi

            print(f"Tag ID: {tag.tag_id}")
            print(f"Position (meters): X={tvec[0]:.3f}, Y={tvec[1]:.3f}, Z={tvec[2]:.3f}")
            print(f"Angles (degrees): Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")

            # Draw tag on frame
            for i in range(4):
                p1 = tuple(tag.corners[i].astype(int))
                p2 = tuple(tag.corners[(i+1) % 4].astype(int))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # Draw center
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Show frame
    cv2.imshow("AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()