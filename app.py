import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image
import av
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# --- Page Configuration ---
st.set_page_config(page_title="AI Posture Correction Pro", page_icon="🐢", layout="wide")

# --- CSS & Audio Script ---
def get_audio_html():
    # 브라우저 기본 오디오 사용 (간단한 beep)
    js_code = """
        <script>
        function playAlert() {
            var audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
            audio.volume = 0.5;
            audio.play();
        }
        </script>
        <div id="audio-container"></div>
    """
    return js_code

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 30px; }
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 30px; }
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 30px; animation: blink 1s infinite; }
    
    .advice-box {
        background-color: #fff9c4;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #fbc02d;
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-top: 10px;
    }

    @keyframes blink {
        50% { opacity: 0.5; }
    }
    
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(get_audio_html(), unsafe_allow_html=True)

st.title("🐢 AI Posture Correction Pro")
st.markdown("Turn on the webcam to analyze your posture. **First, set your own best posture as the standard.**")

# --- Load Model & MediaPipe (모델은 로드만 하고 사용은 안 함: 호환용) ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()
mp_pose = mp.solutions.pose

# --- 거리 기반 확률 계산 함수 (Calibration 전용) ---
def distance_to_probs(distance, t_good=0.12, t_mild=0.28):
    """
    baseline과의 거리(distance)를 받아
    good / mild / severe의 확률 분포를 만들어서 반환.
    t_good: 이 값보다 작으면 거의 good
    t_mild: 이 값보다 크면 severe로 기울기 시작
    """
    d = float(distance)

    # Good 점수: 0에서 t_good까지 선형으로 감소
    good_score = max(0.0, 1.0 - d / max(t_good, 1e-6))

    # Mild 점수: t_good 근처에서 높고, 0과 t_mild에서 0이 되도록
    if d <= t_good:
        mild_score = d / max(t_good, 1e-6)
    elif d <= t_mild:
        mild_score = 1.0 - (d - t_good) / max(t_mild - t_good, 1e-6)
    else:
        mild_score = 0.0

    # Severe 점수: t_mild 이후부터 증가
    if d <= t_mild:
        severe_score = 0.0
    else:
        severe_score = min(1.0, (d - t_mild) / max(t_mild, 1e-6))

    scores = {
        "good": good_score,
        "mild": mild_score,
        "severe": severe_score,
    }
    total = sum(scores.values())
    if total <= 0:
        return {"good": 1/3, "mild": 1/3, "severe": 1/3}

    for k in scores:
        scores[k] /= total

    return scores


# --- 포즈 랜드마크에서 feature 추출 (학습 코드와 동일 논리) ---
def extract_features_from_landmarks(landmarks, img_shape):
    """
    MediaPipe pose_landmarks와 이미지 크기에서
    어깨 기준으로 정규화된 상반신 특징 벡터와 화면에 찍을 포인트 좌표를 반환.
    """
    # 왼/오른 어깨
    l_sh = landmarks[11]
    r_sh = landmarks[12]

    center_x = (l_sh.x + r_sh.x) / 2.0
    center_y = (l_sh.y + r_sh.y) / 2.0
    width = np.linalg.norm(
        np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y])
    )
    if width == 0:
        width = 1.0

    indices = [0, 2, 5, 7, 8, 11, 12]  # 코, 눈, 귀, 어깨
    features = []

    h, w, _ = img_shape
    keypoints = {}

    for idx in indices:
        lm = landmarks[idx]
        norm_x = (lm.x - center_x) / width
        norm_y = (lm.y - center_y) / width
        features.extend([norm_x, norm_y])
        px, py = int(lm.x * w), int(lm.y * h)
        keypoints[idx] = (px, py)

    return features, keypoints


# --- Real-time Video Processing Class (Calibration + Distance 기반 판단) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1
        )

        # 1. Baseline (내 기준 자세)
        self.baseline = None
        self.calibrate_now = False   # 버튼 눌렸을 때 True로 바뀌고, 다음 프레임에서 baseline 저장

        # 2. 거리 smoothing
        self.distance_history = deque(maxlen=10)

        # 3. 결과 공유용 변수
        self.latest_probs = {'good': 0.0, 'mild': 0.0, 'severe': 0.0}
        self.latest_pred = "good"
        self.latest_distance = 0.0

        # 4. 사운드용
        self.severe_consecutive_frames = 0
        self.trigger_sound = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            try:
                # 1) Feature 추출
                features, keypoints = extract_features_from_landmarks(
                    landmarks, img.shape
                )

                # 2) 캘리브레이션 버튼이 눌린 경우 → 현재 자세를 baseline으로 저장
                if self.calibrate_now:
                    self.baseline = np.array(features)
                    self.distance_history.clear()
                    self.calibrate_now = False

                # 3) baseline이 설정된 경우 → 거리 계산 + 확률/레이블 업데이트
                if self.baseline is not None:
                    diff = np.array(features) - np.array(self.baseline)
                    dist = float(np.linalg.norm(diff))
                    self.distance_history.append(dist)
                    avg_dist = float(np.mean(self.distance_history))

                    self.latest_distance = avg_dist

                    # 거리 → 확률 분포
                    prob_dict = distance_to_probs(avg_dist)
                    self.latest_probs = prob_dict
                    self.latest_pred = max(prob_dict, key=prob_dict.get)
                else:
                    # baseline이 아직 없으면, 임시로 모두 good으로
                    self.latest_probs = {'good': 1.0, 'mild': 0.0, 'severe': 0.0}
                    self.latest_pred = 'good'
                    self.latest_distance = 0.0

                current_pred = self.latest_pred

                # 4) Skeleton 시각화 (색상: good=초록, mild=노랑, severe=빨강)
                color = (0, 255, 0)  # Green
                if current_pred == 'mild':
                    color = (0, 255, 255)  # Yellow
                if current_pred == 'severe':
                    color = (0, 0, 255)  # Red

                # 점 찍기
                for idx, (px, py) in keypoints.items():
                    cv2.circle(img, (px, py), 5, color, -1)

                # 어깨선, 목선
                if 11 in keypoints and 12 in keypoints:
                    cv2.line(img, keypoints[11], keypoints[12], color, 2)
                if 0 in keypoints and 11 in keypoints and 12 in keypoints:
                    sh_center = (
                        (keypoints[11][0] + keypoints[12][0]) // 2,
                        (keypoints[11][1] + keypoints[12][1]) // 2,
                    )
                    cv2.line(img, sh_center, keypoints[0], color, 2)

                # 5) 사운드 트리거 (severe가 일정 프레임 이상 지속되면)
                if current_pred == 'severe':
                    self.severe_consecutive_frames += 1
                    if self.severe_consecutive_frames > 30:  # 대략 1초 이상
                        self.trigger_sound = True
                else:
                    self.severe_consecutive_frames = 0
                    self.trigger_sound = False

            except Exception:
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- UI Layout ---
col_main, col_sidebar = st.columns([3, 1])

ctx = None

with col_main:
    # Calibration Button
    st.markdown("### 📏 Calibration")
    st.markdown("1. 편안하지만 **가장 바른 자세**를 만든 뒤<br>2. 아래 버튼을 눌러 현재 자세를 기준으로 저장하세요.", unsafe_allow_html=True)

    calib_msg_ph = st.empty()

    # webrtc_streamer 먼저 생성
    if model is None:
        # 모델은 안 쓰지만, 파일이 없어도 문제없이 동작하게 그냥 정보만
        st.info("Model file (posture_model.pkl) is missing, but calibration-based mode works without it.")
    ctx = webrtc_streamer(
        key="posture-pro",
        video_processor_factory=VideoProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # 버튼: 현재 프레임 기준으로 baseline을 세팅하도록 플래그만 켬
    if st.button("📏 Set Current Posture as 'Standard'"):
        if ctx and ctx.video_processor:
            ctx.video_processor.calibrate_now = True
            calib_msg_ph.success("✅ Standard posture captured! (Hold similar pose when you want 'GOOD')")
        else:
            calib_msg_ph.warning("Webcam is not ready yet. Please wait a moment and try again.")


with col_sidebar:
    st.markdown("### 📊 Live Status")
    status_ph = st.empty()
    advice_ph = st.empty()
    
    st.write("---")
    st.markdown("### Posture Score (Good %)")
    score_ph = st.empty()
    
    st.write("---")
    dist_ph = st.empty()

    # Hidden placeholder for sound
    sound_ph = st.empty()

# --- Main Loop ---
if ctx and ctx.state.playing:
    while True:
        if not ctx.state.playing:
            break

        if ctx.video_processor:
            probs = ctx.video_processor.latest_probs
            pred = ctx.video_processor.latest_pred
            trigger_sound = ctx.video_processor.trigger_sound
            dist = ctx.video_processor.latest_distance

            # 1. Update Status Text & Advice
            if pred == 'good':
                status_ph.markdown("<div class='good-text'>GOOD 😊</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>✅ Perfect alignment! Keep it up.</div>", unsafe_allow_html=True)
            
            elif pred == 'mild':
                status_ph.markdown("<div class='mild-text'>MILD 😐</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>💡 Lift your head slightly.<br>Relax your shoulders.</div>", unsafe_allow_html=True)
            
            else:  # severe
                status_ph.markdown("<div class='severe-text'>SEVERE 🐢</div>", unsafe_allow_html=True)
                advice_ph.markdown("<div class='advice-box'>🚨 <b>Pull chin back!</b><br>Open your chest.</div>", unsafe_allow_html=True)
            
            # 2. Update Single Posture Score Bar (Probability of Good)
            good_score = int(probs.get('good', 0) * 100)
            score_ph.progress(good_score, text=f"{good_score}%")

            # 3. Baseline과의 거리 표시 (참고용)
            dist_ph.markdown(f"Current deviation from standard posture: <b>{dist:.3f}</b>", unsafe_allow_html=True)

            # 4. Sound Alert
            if trigger_sound:
                sound_ph.markdown(
                    """
                    <script>
                    playAlert();
                    </script>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                sound_ph.empty()

        time.sleep(0.1)
