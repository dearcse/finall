import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Page Configuration ---
st.set_page_config(page_title="AI Real-time Posture Correction", page_icon="🐢")

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good { color: #2ecc71; font-weight: bold;}
    .mild { color: #f1c40f; font-weight: bold;}
    .severe { color: #e74c3c; font-weight: bold; background-color: #fadbd8; padding: 5px; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🐢 AI Real-time Turtle Neck Diagnosis")
st.write("Turn on the webcam to analyze your posture in real-time.")

# --- Load Model & MediaPipe ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # Initialize model and pose solution
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 2. Feature Extraction (Same logic as training)
            try:
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                
                # Store coordinates (for drawing)
                h, w, _ = img.shape
                draw_points = []

                for idx in indices:
                    lm = landmarks[idx]
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    draw_points.append((int(lm.x * w), int(lm.y * h)))

                # 3. Prediction
                if self.model:
                    prediction = self.model.predict([features])[0]
                    # probs = self.model.predict_proba([features])[0] # Unused variable
                    
                    # 4. Draw results on screen
                    # Draw points
                    for px, py in draw_points:
                        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                    
                    # Display status text
                    status_text = f"Status: {prediction.upper()}"
                    color = (0, 255, 0) # Green
                    
                    if prediction == 'severe':
                        color = (0, 0, 255) # Red
                        cv2.putText(img, "WARNING!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    elif prediction == 'mild':
                        color = (0, 255, 255) # Yellow

                    cv2.putText(img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
            except Exception as e:
                pass # Ignore errors and output only video

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Tab Configuration ---
tab1, tab2 = st.tabs(["📷 Real-time Analysis", "🖼️ Upload Photo"])

with tab1:
    st.header("Real-time Webcam")
    if model is None:
        st.error("Model file (posture_model.pkl) is missing.")
    else:
        # Start real-time streaming
        webrtc_streamer(
            key="posture-check",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

with tab2:
    st.header("File Upload Diagnosis")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # (Existing image analysis logic is handled here - omitted for brevity, can be added if needed)
        st.info("Try the Real-time Analysis tab!")
