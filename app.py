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

# 스타일 설정 (3가지 색상 막대)
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold;}
    .mild-text { color: #f1c40f; font-weight: bold;}
    .severe-text { color: #e74c3c; font-weight: bold;}
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
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # 2. Feature Extraction (학습 때와 동일한 정규화)
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1

                indices = [0, 2, 5, 7, 8, 11, 12]
                features = []
                
                h, w, _ = img.shape
                draw_points = []

                for idx in indices:
                    lm = landmarks[idx]
                    norm_x = (lm.x - center_x) / width
                    norm_y = (lm.y - center_y) / width
                    features.extend([norm_x, norm_y])
                    draw_points.append((int(lm.x * w), int(lm.y * h)))

                # 3. Prediction & Probability Calculation
                if self.model:
                    probs = self.model.predict_proba([features])[0]
                    classes = self.model.classes_ # 모델의 클래스 순서 (예: ['good', 'mild', 'severe'])
                    
                    # 클래스 이름과 확률 매핑
                    prob_dict = {cls: p for cls, p in zip(classes, probs)}
                    p_good = prob_dict.get('good', 0)
                    p_mild = prob_dict.get('mild', 0)
                    p_severe = prob_dict.get('severe', 0)
                    
                    # 4. 화면에 그리기 (Draw Results)
                    # 랜드마크 점 찍기
                    for px, py in draw_points:
                        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                    
                    # === [핵심] 3개의 Bar 그리기 (OpenCV) ===
                    # 위치 설정
                    bar_x, bar_y = 20, 60
                    bar_w, bar_h = 200, 20
                    gap = 35
                    
                    # 배경 박스 (반투명 검정)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (10, 10), (280, 180), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

                    # (1) Good Bar (Green)
                    cv2.putText(img, f"Good: {int(p_good*100)}%", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * p_good), bar_y + bar_h), (0, 255, 0), -1) # 채워진 바
                    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1) # 테두리

                    # (2) Mild Bar (Yellow/Cyan in BGR)
                    cv2.putText(img, f"Mild: {int(p_mild*100)}%", (bar_x, bar_y + gap - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(img, (bar_x, bar_y + gap), (bar_x + int(bar_w * p_mild), bar_y + gap + bar_h), (0, 255, 255), -1)
                    cv2.rectangle(img, (bar_x, bar_y + gap), (bar_x + bar_w, bar_y + gap + bar_h), (255, 255, 255), 1)

                    # (3) Severe Bar (Red/Blue in BGR)
                    cv2.putText(img, f"Severe: {int(p_severe*100)}%", (bar_x, bar_y + 2*gap - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(img, (bar_x, bar_y + 2*gap), (bar_x + int(bar_w * p_severe), bar_y + 2*gap + bar_h), (0, 0, 255), -1)
                    cv2.rectangle(img, (bar_x, bar_y + 2*gap), (bar_x + bar_w, bar_y + 2*gap + bar_h), (255, 255, 255), 1)
                    
                    # 경고 메시지 (Severe가 가장 높을 때)
                    if p_severe > p_good and p_severe > p_mild:
                        cv2.putText(img, "BAD POSTURE!", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
            except Exception as e:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Tab Configuration ---
tab1, tab2 = st.tabs(["📷 Real-time Analysis", "🖼️ Upload Photo"])

# Tab 1: Real-time
with tab1:
    st.header("Real-time Webcam")
    if model is None:
        st.error("Model file (posture_model.pkl) is missing.")
    else:
        webrtc_streamer(
            key="posture-check",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

# Tab 2: Upload
with tab2:
    st.header("File Upload Diagnosis")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Feature Extraction for Uploaded Image
        img_np = np.array(image.convert('RGB'))
        pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
        results = pose_static.process(img_np)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # Normalization
                l_sh = landmarks[11]; r_sh = landmarks[12]
                center_x = (l_sh.x + r_sh.x) / 2
                center_y = (l_sh.y + r_sh.y) / 2
                width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
                if width == 0: width = 1
                
                features = []
                indices = [0, 2, 5, 7, 8, 11, 12]
                for idx in indices:
                    lm = landmarks[idx]
                    features.extend([(lm.x - center_x)/width, (lm.y - center_y)/width])
                
                # Prediction
                probs = model.predict_proba([features])[0]
                classes = model.classes_
                prob_dict = {cls: round(p * 100, 1) for cls, p in zip(classes, probs)}
                
                # === [핵심] 3개의 Bar 표시 (Streamlit UI) ===
                st.subheader("Analysis Result")
                
                # Good
                st.markdown(f"<span class='good-text'>Good: {prob_dict.get('good', 0)}%</span>", unsafe_allow_html=True)
                st.progress(int(prob_dict.get('good', 0)))
                
                # Mild
                st.markdown(f"<span class='mild-text'>Mild: {prob_dict.get('mild', 0)}%</span>", unsafe_allow_html=True)
                st.progress(int(prob_dict.get('mild', 0)))
                
                # Severe
                st.markdown(f"<span class='severe-text'>Severe: {prob_dict.get('severe', 0)}%</span>", unsafe_allow_html=True)
                st.progress(int(prob_dict.get('severe', 0)))
                
                # Final Text Result
                pred = model.predict([features])[0]
                if pred == 'severe':
                    st.error("🚨 WARNING: Severe Forward Head Posture detected!")
                elif pred == 'mild':
                    st.warning("🟡 Caution: Mild Forward Head Posture.")
                else:
                    st.success("🟢 Good Posture!")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
        else:
            st.error("Could not detect a person in the image.")
