import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Live Posture Detection")
st.title("📸 Live Posture Detection")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine))
    return angle

# Create a placeholder for the video frames
frame_placeholder = st.empty()

cap = cv2.VideoCapture(0)  # 0 = your MacBook camera

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not read camera. Check permissions.")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[11].x, landmarks[11].y]
            ear = [landmarks[7].x, landmarks[7].y]
            hip = [landmarks[23].x, landmarks[23].y]

            neck_angle = calculate_angle(ear, shoulder, hip)

            if neck_angle < 155:
                status = "⚠️ SLOUCHING!"
                color = (0, 0, 255)
            else:
                status = "Good posture"
                color = (0, 255, 0)

            cv2.putText(frame, f"{status,neck_angle}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

stop = st.button("Stop Camera")
if stop:
    break