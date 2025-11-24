import streamlit as st
import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(page_title="Posture Slouch Detector", layout="wide")
st.title("💺 Real-Time Posture Slouch Detector (MoveNet)")

# -----------------------------
# Load MoveNet once
# -----------------------------
@st.cache_resource
def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model.signatures['serving_default']

movenet = load_movenet()

# -----------------------------
# Detect keypoints
# -----------------------------
def detect_keypoints(frame):
    img = cv2.resize(frame, (192, 192))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    outputs = movenet(input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # 17 keypoints
    return keypoints

# -----------------------------
# Start webcam capture
# -----------------------------
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# Optional: process every Nth frame for speed
frame_skip = 2
counter = 0
keypoints_cache = None

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Cannot access camera")
        break

    counter += 1
    # Only process MoveNet every N frames
    if counter % frame_skip == 0:
        keypoints_cache = detect_keypoints(frame)

    # Draw posture info
    if keypoints_cache is not None:
        h, w, _ = frame.shape
        ls = keypoints_cache[5]  # left shoulder
        rs = keypoints_cache[6]  # right shoulder
        shoulder_y = ((ls[1] + rs[1]) / 2) * h

        if shoulder_y > h * 0.55:  # slouch threshold
            cv2.putText(frame, "SLOUCHING!", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 5)
        else:
            cv2.putText(frame, "Good Posture", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (0,0), (w,h), (0,255,0), 5)

    # Show frame in Streamlit
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Small delay to control FPS
    time.sleep(0.03)  # ~30 FPS

cap.release()