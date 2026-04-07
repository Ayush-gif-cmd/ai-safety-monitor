import streamlit as st
import cv2
import numpy as np
from core.detector import HandGloveDetector
from core.alert import AlertManager
from core.logger import EventLogger
import time
from PIL import Image

# ----------------- #
# Page Configuration
# ----------------- #
st.set_page_config(
    page_title="AI Safety Monitoring",
    page_icon="🧤",
    layout="wide"
)

st.title("🛡️ AI-Based Safety Monitoring System")
st.markdown("**Objective:** Real-time detection of safety gloves using Computer Vision.")

# ----------------- #
# Initialize Modules
# ----------------- #
@st.cache_resource
def load_detector():
    return HandGloveDetector()

@st.cache_resource
def load_alert_manager():
    return AlertManager()

@st.cache_resource
def load_logger():
    return EventLogger()

detector = load_detector()
alert_manager = load_alert_manager()
logger = load_logger()

# ----------------- #
# Layout Setup
# ----------------- #
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Camera Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Control Panel")
    
    # Selecting the camera will auto-reset the run state if it was running
    camera_index = st.selectbox("Select Camera Index", options=[0, 1], index=0, help="0 is usually your iPhone, 1 is your Mac Webcam")
    run_camera = st.checkbox("Turn On Camera", value=False)
    
    st.markdown("---")
    st.subheader("System Status")
    status_placeholder = st.empty()
    status_placeholder.success("System Ready")
    
    st.markdown("---")
    st.subheader("Recent Activity")
    log_placeholder = st.empty()

# ----------------- #
# Main Processing Loop
# ----------------- #

if run_camera:
    # Initialize the camera safely
    cap = cv2.VideoCapture(camera_index)
    
    # Configure resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_log_time = 0
    log_cooldown = 2.0  # seconds

    if not cap.isOpened():
        st.error(f"Error: Could not open webcam at index {camera_index}.")
        st.info("Try unchecking 'Turn On Camera', changing the index to 1, and checking it again.")
    else:
        status_placeholder.info(f"Camera {camera_index} Active - Monitoring...")
        
        try:
            retry_count = 0
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    time.sleep(0.5)
                    if retry_count > 5:
                        st.error("Camera disconnected or permissions denied.")
                        break
                    continue
                
                retry_count = 0
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)

                # Process Frame
                processed_frame, is_violation = detector.detect(frame)
                
                # Handle Violations
                if is_violation:
                    alert_manager.trigger()
                    
                    current_time = time.time()
                    if current_time - last_log_time > log_cooldown:
                        last_log_time = current_time
                        logger.log_violation(processed_frame)
                        status_placeholder.error("🚨 ALERT: GLOVES NOT DETECTED!")
                else:
                    status_placeholder.success("✅ Safe: Gloves Detected or Hands Clear")
                
                # Display frame
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB")
                
                time.sleep(0.01)
        finally:
            # This finally block ensures the camera is ALWAYS released even if you change the dropdown
            cap.release()
            
else:
    status_placeholder.warning("Camera Offline")
    frame_placeholder.empty()
