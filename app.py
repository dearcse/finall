import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from collections import deque

# --- Page Configuration ---
st.set_page_config(page_title="AI Real-time Posture Correction", page_icon="🐢")

# 스타일 설정
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 20px;}
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 20px;}
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 20px;}
    .warning-box { background-color: #fadbd8; border: 2px solid #e74c3c; padding: 15px; border-radius: 10px; text-align: center; }
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

# --- Helper Function: Adjust Probabilities ---
def adjust_probabilities(probs, classes):
    """
    Severe 확률을 낮추고 Good 확률을 높여주는 보정 함수
    """
    prob_dict = {cls: p for cls, p in zip(classes, probs)}
    
    # Severe 확률에 0.7을 곱해 낮춤 (너무 민감하지 않게)
    if 'severe' in prob_dict:
        prob_dict['severe'] *= 0.7
    
    # Mild 확률도 약간 조정 (선택 사항)
    # if 'mild' in prob_dict:
    #     prob_dict['mild'] *= 0.9

    # 줄어든 확률만큼 다시 정규화 (합이 1이 되도록)
    total = sum(prob_dict.values())
    if total > 0:
        for cls in prob_dict:
            prob_dict[cls] /= total
            
    # 확률을 기준으로 다시 정렬된 클래스 예측
    # (가장 높은 확률을 가진 클래스를 새로운 예측값으로 설정)
    new_pred = max(prob_dict, key=prob_dict.get)
            
    return prob_dict, new_pred

# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.model = model
        self.latest_probs = {'good': 0, 'mild': 0, 'severe': 0}
        self.latest_pred = None
        self.history = deque(maxlen=10)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                l_sh = landmarks[11]; r_sh = landmarks[12]
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

                if self.model:
                    # 1. 모델 예측
                    current_probs = self.model.predict_proba([features])[0]
                    self.history.append(current_probs)
                    
                    # 2. 평균 계산 (Smoothing)
                    avg_probs = np.mean(self.history, axis=0)
                    classes = self.model.classes_
                    
                    # 3. [수정됨] 확률 보정 (Severe 낮추기)
                    final_prob_dict, final_pred = adjust_probabilities(avg_probs, classes)
                    
                    self.latest_probs = final_prob_dict
                    self.latest_pred = final_pred
                    
                    for px, py in draw_points:
                        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                    
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
        col1, col2 = st.columns([2, 1])
        with col1:
            ctx = webrtc_streamer(
                key="posture-check",
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True
            )
        with col2:
            st.subheader("Live Status")
            status_text_ph = st.empty()
            
            # 3개의 Bar를 위한 공간 생성
            st.write("**Prediction Confidence:**")
            col_severe, col_mild, col_good = st.columns(3)
            
            with col_severe:
                st.markdown("<p style='text-align: center; color: #e74c3c; font-weight: bold;'>Severe</p>", unsafe_allow_html=True)
                bar_severe_ph = st.empty()
            
            with col_mild:
                st.markdown("<p style='text-align: center; color: #f1c40f; font-weight: bold;'>Mild</p>", unsafe_allow_html=True)
                bar_mild_ph = st.empty()
                
            with col_good:
                st.markdown("<p style='text-align: center; color: #2ecc71; font-weight: bold;'>Good</p>", unsafe_allow_html=True)
                bar_good_ph = st.empty()

            warning_ph = st.empty()

        if ctx.state.playing:
            while True:
                if ctx.video_processor:
                    probs = ctx.video_processor.latest_probs
                    pred = ctx.video_processor.latest_pred
                    
                    if pred:
                        p_good = int(probs.get('good', 0) * 100)
                        p_mild = int(probs.get('mild', 0) * 100)
                        p_severe = int(probs.get('severe', 0) * 100)

                        if pred == 'good':
                            status_text_ph.markdown(f"<p class='good-text'>Status: GOOD 😊</p>", unsafe_allow_html=True)
                        elif pred == 'mild':
                            status_text_ph.markdown(f"<p class='mild-text'>Status: MILD 😐</p>", unsafe_allow_html=True)
                        else:
                            status_text_ph.markdown(f"<p class='severe-text'>Status: SEVERE 🐢</p>", unsafe_allow_html=True)

                        # 3개의 Bar 업데이트 (세로형 Bar는 Streamlit 기본 기능으로 어려워 가로형 Bar 사용)
                        # 또는 각 컬럼에 metric이나 progress bar 사용
                        bar_severe_ph.progress(p_severe)
                        bar_mild_ph.progress(p_mild)
                        bar_good_ph.progress(p_good)
                        
                        # 텍스트로 퍼센트 표시 추가 (선택사항)
                        # bar_severe_ph.metric("Severe", f"{p_severe}%") ...

                        if pred == 'severe':
                            warning_ph.markdown("""
                                <div class='warning-box'>
                                    🚨 <b>BAD POSTURE DETECTED!</b><br>
                                    Please straighten your neck.
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            warning_ph.empty()
                import time
                time.sleep(0.1)

# Tab 2: Upload
with tab2:
    st.header("File Upload Diagnosis")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_np = np.array(image.convert('RGB'))
        pose_static = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
        results = pose_static.process(img_np)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
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
                
                # [수정됨] 업로드 모드에서도 확률 보정 적용
                raw_probs = model.predict_proba([features])[0]
                classes = model.classes_
                
                prob_dict, pred = adjust_probabilities(raw_probs, classes)
                
                st.subheader("Analysis Result")
                
                # 3개의 Bar 표시 (업로드 탭)
                col_u_severe, col_u_mild, col_u_good = st.columns(3)
                
                with col_u_severe:
                    st.markdown("<p style='text-align: center; color: #e74c3c; font-weight: bold;'>Severe</p>", unsafe_allow_html=True)
                    st.progress(int(prob_dict.get('severe', 0)*100))
                    st.write(f"{int(prob_dict.get('severe', 0)*100)}%")
                    
                with col_u_mild:
                    st.markdown("<p style='text-align: center; color: #f1c40f; font-weight: bold;'>Mild</p>", unsafe_allow_html=True)
                    st.progress(int(prob_dict.get('mild', 0)*100))
                    st.write(f"{int(prob_dict.get('mild', 0)*100)}%")
                    
                with col_u_good:
                    st.markdown("<p style='text-align: center; color: #2ecc71; font-weight: bold;'>Good</p>", unsafe_allow_html=True)
                    st.progress(int(prob_dict.get('good', 0)*100))
                    st.write(f"{int(prob_dict.get('good', 0)*100)}%")

                if pred == 'severe':
                    st.error("🚨 WARNING: Severe Forward Head Posture detected!")
                elif pred == 'mild':
                    st.warning("🟡 Caution: Mild Forward Head Posture.")
                else:
                    st.success("🟢 Good Posture!")
            except:
                st.error("Analysis failed.")
        else:
            st.error("Person not found.")
