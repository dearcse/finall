import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from collections import deque
import time

# --- Page Configuration ---
st.set_page_config(page_title="AI Real-time Posture Calibration", page_icon="🐢")

# --- 스타일 설정 ---
st.markdown(
    """
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good-text { color: #2ecc71; font-weight: bold; font-size: 20px;}
    .mild-text { color: #f1c40f; font-weight: bold; font-size: 20px;}
    .severe-text { color: #e74c3c; font-weight: bold; font-size: 20px;}
    .warning-box { background-color: #fadbd8; border: 2px solid #e74c3c; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐢 AI Real-time Turtle Neck Calibration")
st.write("First, hold your **best posture** for a few seconds. The app will use it as your personal standard.")

mp_pose = mp.solutions.pose


# --- 공통 Feature 추출 함수 (학습 때와 동일한 방식) ---
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
    draw_points = []

    for idx in indices:
        lm = landmarks[idx]
        norm_x = (lm.x - center_x) / width
        norm_y = (lm.y - center_y) / width
        features.extend([norm_x, norm_y])
        draw_points.append((int(lm.x * w), int(lm.y * h)))

    return features, draw_points


# --- 거리 기반 확률 계산 함수 (fuzzy membership 비슷하게) ---
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
        # t_mild 이후로 점점 1에 가까워지도록
        severe_score = min(1.0, (d - t_mild) / max(t_mild, 1e-6))

    scores = {
        "good": good_score,
        "mild": mild_score,
        "severe": severe_score,
    }
    total = sum(scores.values())
    if total <= 0:
        # 전부 0이면 균등분포
        return {"good": 1 / 3, "mild": 1 / 3, "severe": 1 / 3}

    # 정규화
    for k in scores:
        scores[k] /= total

    return scores


# --- Real-time Video Processing Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1,
        )

        # 캘리브레이션 관련
        self.collecting_baseline = True
        self.baseline_buffer = []
        self.baseline = None

        # 거리 smoothing
        self.distance_history = deque(maxlen=10)

        # 실시간 상태 공유용
        self.latest_probs = {"good": 0.0, "mild": 0.0, "severe": 0.0}
        self.latest_pred = None
        self.latest_distance = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1. MediaPipe 처리
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            try:
                # 2. Feature 추출
                features, draw_points = extract_features_from_landmarks(
                    landmarks, img.shape
                )

                # 3. Calibration / Distance 계산
                if self.collecting_baseline:
                    # baseline 수집 단계
                    self.baseline_buffer.append(features)
                    # 20프레임 정도 모으면 평균을 baseline으로 사용
                    if len(self.baseline_buffer) >= 20:
                        self.baseline = np.mean(self.baseline_buffer, axis=0)
                        self.collecting_baseline = False
                        self.distance_history.clear()
                elif self.baseline is not None:
                    # baseline이 준비된 이후 → 현재 자세와 거리 계산
                    diff = np.array(features) - np.array(self.baseline)
                    dist = float(np.linalg.norm(diff))
                    self.distance_history.append(dist)
                    avg_dist = float(np.mean(self.distance_history))

                    self.latest_distance = avg_dist

                    # 거리 → good/mild/severe 확률 분포
                    prob_dict = distance_to_probs(avg_dist)
                    self.latest_probs = prob_dict

                    # 가장 높은 확률을 pred로 사용
                    self.latest_pred = max(prob_dict, key=prob_dict.get)

                # 4. 화면에는 점만 찍기
                for px, py in draw_points:
                    cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

            except Exception:
                # 에러 발생 시 프레임만 그대로 반환
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Main Tab Configuration ---
tab1, tab2 = st.tabs(["📷 Real-time (Calibrated)", "🖼️ Upload Photo (disabled)"])

# Tab 1: Real-time with Calibration
with tab1:
    st.header("Real-time Webcam (Personal Calibration)")

    col1, col2 = st.columns([2, 1])

    # 왼쪽: 웹캠
    with col1:
        ctx = webrtc_streamer(
            key="posture-calibration",
            video_processor_factory=VideoProcessor,
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            async_processing=True,
        )

    # 오른쪽: 상태 표시
    with col2:
        st.subheader("Live Status")

        calib_text_ph = st.empty()
        status_text_ph = st.empty()

        st.write("**Prediction Confidence:**")

        # 라벨 (Good / Mild / Severe)
        label_good, label_mild, label_severe = st.columns(3)

        with label_good:
            st.markdown(
                "<p style='text-align: center; color: #2ecc71; font-weight: bold;'>Good</p>",
                unsafe_allow_html=True,
            )

        with label_mild:
            st.markdown(
                "<p style='text-align: center; color: #f1c40f; font-weight: bold;'>Mild</p>",
                unsafe_allow_html=True,
            )

        with label_severe:
            st.markdown(
                "<p style='text-align: center; color: #e74c3c; font-weight: bold;'>Severe</p>",
                unsafe_allow_html=True,
            )

        # 가로 Progress bar (전체 폭)
        st.write("Good:")
        bar_good_ph = st.empty()

        st.write("Mild:")
        bar_mild_ph = st.empty()

        st.write("Severe:")
        bar_severe_ph = st.empty()

        warning_ph = st.empty()
        distance_ph = st.empty()

    # 실시간 업데이트 루프
    if ctx and ctx.state.playing:
        while True:
            if not ctx.state.playing:
                break

            vp = ctx.video_processor

            if vp is not None:
                # 캘리브레이션 상태 표시
                if vp.collecting_baseline or vp.baseline is None:
                    calib_text_ph.info(
                        "🧭 Calibrating… Please hold your **best neutral posture**."
                    )
                else:
                    calib_text_ph.success(
                        "✅ Calibration complete! Now analyzing your posture."
                    )

                probs = vp.latest_probs
                pred = vp.latest_pred
                dist = vp.latest_distance

                # distance 표시 (참고용)
                if vp.baseline is not None:
                    distance_ph.markdown(
                        f"<p>Current deviation from baseline: <b>{dist:.3f}</b></p>",
                        unsafe_allow_html=True,
                    )
                else:
                    distance_ph.empty()

                if pred is not None:
                    p_good = int(probs.get("good", 0.0) * 100)
                    p_mild = int(probs.get("mild", 0.0) * 100)
                    p_severe = int(probs.get("severe", 0.0) * 100)

                    # 상태 텍스트
                    if pred == "good":
                        status_text_ph.markdown(
                            "<p class='good-text'>Status: GOOD 😊</p>",
                            unsafe_allow_html=True,
                        )
                    elif pred == "mild":
                        status_text_ph.markdown(
                            "<p class='mild-text'>Status: MILD 😐</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        status_text_ph.markdown(
                            "<p class='severe-text'>Status: SEVERE 🐢</p>",
                            unsafe_allow_html=True,
                        )

                    # Progress bars
                    bar_good_ph.progress(p_good, text=f"Good: {p_good}%")
                    bar_mild_ph.progress(p_mild, text=f"Mild: {p_mild}%")
                    bar_severe_ph.progress(p_severe, text=f"Severe: {p_severe}%")

                    # Warning box
                    if pred == "severe":
                        warning_ph.markdown(
                            """
                            <div class='warning-box'>
                                🚨 <b>BAD POSTURE DETECTED!</b><br>
                                Please straighten your neck.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        warning_ph.empty()
                else:
                    # 아직 baseline만 모으는 중이거나, 정보 부족
                    status_text_ph.markdown(
                        "<p>Waiting for stable posture...</p>",
                        unsafe_allow_html=True,
                    )
                    bar_good_ph.progress(0, text="Good: 0%")
                    bar_mild_ph.progress(0, text="Mild: 0%")
                    bar_severe_ph.progress(0, text="Severe: 0%")
                    warning_ph.empty()

            time.sleep(0.1)

# Tab 2: Upload (현재 비활성화)
with tab2:
    st.header("Upload Photo Diagnosis (Disabled in Calibration Mode)")
    st.info(
        "This prototype focuses on **real-time calibrated analysis**.\n\n"
        "Please use the **Real-time (Calibrated)** tab to analyze your posture "
        "relative to your own best baseline."
    )


