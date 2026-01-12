import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp

# Explicitly import the solutions to avoid the AttributeError
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

class ASLProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Flip for selfie view
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        label = "No hand detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmarks
                lm = hand_landmarks.landmark
                
                # Logic to determine finger states (Open/Closed)
                # Tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
                index_open = lm[8].y < lm[6].y
                middle_open = lm[12].y < lm[10].y
                ring_open = lm[16].y < lm[14].y
                pinky_open = lm[20].y < lm[18].y
                
                # Thumb Logic (Calculated by horizontal distance for side-thumb signs)
                thumb_open = lm[4].x < lm[3].x 

                # --- ASL Logic Mapping (Based on your chart) ---
                if index_open and middle_open and ring_open and pinky_open:
                    label = "Letter: B"
                elif index_open and not middle_open and not ring_open and not pinky_open:
                    # L has thumb out, D has thumb tucked
                    if lm[4].x < lm[3].x: label = "Letter: L"
                    else: label = "Letter: D"
                elif index_open and middle_open and not ring_open and not pinky_open:
                    label = "Letter: V"
                elif not index_open and not middle_open and not ring_open and pinky_open:
                    label = "Letter: I"
                elif thumb_open and pinky_open and not index_open and not middle_open:
                    label = "Letter: Y"
                elif not index_open and not middle_open and not ring_open and not pinky_open:
                    label = "Letter: A / S" # Fist signs
                else:
                    label = "Searching..."

        # Overlay text on the video
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# Streamlit UI
st.set_page_config(page_title="ASL Translator", layout="centered")
st.title("🤟 ASL Real-Time Translator")
st.write("Using MediaPipe for hand tracking and your provided ASL chart.")

# WebRTC configuration for Render (Stun servers help bypass firewalls)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="asl-translate",
    video_processor_factory=ASLProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)
