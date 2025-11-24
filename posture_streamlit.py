import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.title("Posture Slouch Detector (MoveNet)")

# Load MoveNet once
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

def movenet_detect(image):
    img = cv2.resize(image, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = tf.convert_to_tensor(img, dtype=tf.int32)
    inp = tf.expand_dims(inp, axis=0)
    outputs = movenet(inp)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        keypoints = movenet_detect(img)

        # Shoulders
        ls = keypoints[5]
        rs = keypoints[6]
        shoulder_y = (ls[1] + rs[1]) / 2

        if shoulder_y > 15:
            cv2.putText(img, "SLOUCHING!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Good Posture", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)