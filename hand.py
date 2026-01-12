import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import math
from spellchecker import SpellChecker

# Initialize Tools
spell = SpellChecker()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

if 'sentence' not in st.session_state:
    st.session_state.sentence = []

class SignLanguageProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        label = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark

                # Finger states [Thumb, Index, Middle, Ring, Pinky]
                fingers = []
                fingers.append(1 if lm[4].x > lm[3].x else 0) # Thumb
                for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                    fingers.append(1 if lm[tip].y < lm[pip].y else 0)

                # Advanced Logic Mapping
                if fingers == [0, 1, 1, 1, 1]: label = "B"
                elif fingers == [0, 1, 0, 0, 0]: label = "D"
                elif fingers == [0, 0, 0, 0, 1]: label = "I"
                elif fingers == [1, 1, 0, 0, 0]: label = "L"
                elif fingers == [1, 0, 0, 0, 1]: label = "Y"
                elif fingers == [0, 1, 1, 0, 0]: label = "V"
                elif fingers == [0, 1, 1, 1, 0]: label = "W"
                elif fingers == [0, 0, 0, 0, 0]: # Fist variants
                    d_idx = get_dist(lm[4], lm[7])
                    d_mid = get_dist(lm[4], lm[11])
                    if d_idx < 0.04: label = "T"
                    elif d_mid < 0.04: label = "N"
                    else: label = "A"
                else: label = "Looking..."

        cv2.putText(img, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        return img

# --- STREAMLIT UI ---
st.set_page_config(page_title="ASL AI Translator", layout="wide")
st.title("🤟 ASL Translator + Autocorrect")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(key="asl-full", video_processor_factory=SignLanguageProcessor)

with col2:
    st.write("### Controls")
    current_char = st.text_input("Detected Letter (Manual Edit)", value="")
    
    if st.button("Add Letter to Word"):
        st.session_state.sentence.append(current_char.upper())
    
    full_word = "".join(st.session_state.sentence)
    st.write(f"**Current Spelling:** `{full_word}`")
    
    if full_word:
        corrected = spell.correction(full_word)
        if corrected and corrected.upper() != full_word:
            st.success(f"Did you mean: **{corrected.upper()}**?")
            if st.button("Use Correction"):
                st.session_state.sentence = list(corrected.upper())
                st.rerun()

    if st.button("Clear All"):
        st.session_state.sentence = []
        st.rerun()

st.divider()
st.info("How to use: Sign a letter, type it into the box, and click 'Add'. The AI will suggest the correct word if you make a mistake!")
