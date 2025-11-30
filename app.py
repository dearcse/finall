import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="AI Frontal Posture Analysis", page_icon="🐢")

# CSS for Styling (Red warning box, etc.)
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .good { color: #2ecc71; }
    .mild { color: #f1c40f; }
    .severe { color: #e74c3c; border: 2px solid #e74c3c; padding: 10px; border-radius: 10px; background-color: #fadbd8; }
    .warning-text { color: #e74c3c; font-weight: bold; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title(" AI Forward Head Posture Diagnosis")
st.write("Analyzes the risk of forward head posture from a frontal photo using a trained AI model.")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        # The model file created in step 1 must be in the same folder
        return joblib.load('posture_model.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.error("⚠️ 'posture_model.pkl' not found. Please train the model and upload the file.")
    st.stop()

# --- MediaPipe Configuration ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

def extract_features(image):
    """Extract landmarks from the image using the same method as training"""
    image_rgb = np.array(image.convert('RGB'))
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, image_rgb

    landmarks = results.pose_landmarks.landmark
    
    # Coordinate normalization logic (Must match training code)
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    
    center_x = (l_sh.x + r_sh.x) / 2
    center_y = (l_sh.y + r_sh.y) / 2
    width = np.linalg.norm(np.array([l_sh.x, l_sh.y]) - np.array([r_sh.x, r_sh.y]))
    if width == 0: width = 1

    indices = [0, 2, 5, 7, 8, 11, 12]
    features = []
    
    # Save coordinates for visualization
    draw_points = []
    h, w, _ = image_rgb.shape
    
    for idx in indices:
        lm = landmarks[idx]
        norm_x = (lm.x - center_x) / width
        norm_y = (lm.y - center_y) / width
        features.extend([norm_x, norm_y])
        
        # Save points to draw on original image
        draw_points.append((int(lm.x * w), int(lm.y * h)))

    # Draw points on image
    for px, py in draw_points:
        cv2.circle(image_rgb, (px, py), 5, (0, 255, 0), -1)

    return [features], image_rgb

# --- Main Functions ---
tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])

input_image = None

with tab1:
    st.header("Real-time Webcam")
    camera_file = st.camera_input("Take a picture looking straight ahead")
    if camera_file:
        input_image = Image.open(camera_file)

with tab2:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        input_image = Image.open(uploaded_file)

# --- Analysis and Result Output ---
if input_image is not None:
    st.divider()
    st.subheader(" Analysis Results")
    
    features, annotated_image = extract_features(input_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(annotated_image, caption="Analyzed Image", use_column_width=True)
        
    with col2:
        if features:
            # Prediction (Calculate probabilities)
            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            classes = model.classes_ # Check order: ['good', 'mild', 'severe']
            
            # Create probability dictionary
            prob_dict = {cls: round(p * 100, 1) for cls, p in zip(classes, probs)}
            
            # Result Display UI
            st.write("### Posture Analysis Report")
            
            # Progress Bars
            st.write(f"**Good: {prob_dict.get('good', 0)}%**")
            st.progress(int(prob_dict.get('good', 0)))
            
            st.write(f"**Mild: {prob_dict.get('mild', 0)}%**")
            st.progress(int(prob_dict.get('mild', 0)))
            
            st.write(f"**Severe: {prob_dict.get('severe', 0)}%**")
            st.progress(int(prob_dict.get('severe', 0)))
            
            st.divider()
            
            # Final Judgment and Warning
            if prediction == 'severe':
                st.markdown(f"""
                <div class='severe'>
                    <p class='warning-text'>🚨 WARNING: Severe Forward Head Posture!</p>
                    <p>Your neck is significantly forward or shoulder balance is off.<br>
                    Please stretch immediately and correct your posture.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == 'mild':
                st.markdown("<h3 class='mild'>🟡 Caution Required.</h3>", unsafe_allow_html=True)
                st.write("Consciously tuck your chin and straighten your shoulders.")
            else:
                st.markdown("<h3 class='good'>🟢 Excellent Posture!</h3>", unsafe_allow_html=True)
                st.write("Keep up the good work.")
                
        else:
            st.error("Person not found. Please retake the photo ensuring your face is clearly visible.")