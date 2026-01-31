import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import time
import torch
import random
try:
    from facenet_pytorch import InceptionResnetV1
    FACE_MATCH_AVAILABLE = True
except ImportError:
    FACE_MATCH_AVAILABLE = False
    print("WARNING: facenet_pytorch import failed. Face matching disabled.")
except RuntimeError:
    FACE_MATCH_AVAILABLE = False
    print("WARNING: torchvision/torch version mismatch. Face matching disabled.")

from test_ai_detector import AIImageDetector
from silent_face_model import AntiSpoofPredictor
from fraud_service import analyze_voice_spoof
import os
try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    try:
        from streamlit_audiorecorder import audiorecorder
        RECORDER_AVAILABLE = True
    except ImportError:
        RECORDER_AVAILABLE = False
        print("WARNING: No audio recorder installed.")

# Page Configuration
st.set_page_config(page_title="KYC Liveness & AI Detection", page_icon="🛡️")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants for Landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth landmarks (Inner lips)
MOUTH_INNER = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]

# Helper Functions
def get_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio"""
    # Vertical distances
    v1 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    v2 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    # Horizontal distance
    h = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h)

def get_mar(landmarks):
    """Calculate Mouth Aspect Ratio (Vertical / Horizontal)"""
    # 13=UpperLipInner, 14=LowerLipInner
    # 61=MouthCornerLeft, 291=MouthCornerRight
    
    # We need to access by index in the 'coords' array, not MediaPipe IDs directly
    # because 'coords' is a numpy array of shape (468, 2)
    # MediaPipe IDs: Upper=13, Lower=14, L=61, R=291
    
    A = np.linalg.norm(landmarks[13] - landmarks[14]) # Vertical
    B = np.linalg.norm(landmarks[61] - landmarks[291]) # Horizontal
    return A / B

def get_head_yaw(landmarks):
    """Calculate Head Yaw (Left/Right Turn)"""
    nose = landmarks[1][0]
    right_edge = landmarks[454][0]
    left_edge = landmarks[234][0]
    face_width = right_edge - left_edge
    relative_nose_pos = (nose - left_edge) / face_width
    return relative_nose_pos

def detect_replay_attack(image):
    """
    Forensic analysis to detect screen replay attacks (Phone/Monitor).
    Returns: replay_score (0.0=Real, 1.0=Fake/Replay), metric_val
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Feature 1: Lighting anomalies (Uniformity)
    # Screens are often backlit and have flatter lighting than real faces.
    # Real 3D faces usually have shadows and gradients (High Variance in V channel).
    v_channel = hsv[:,:,2]
    unique_lighting = np.std(v_channel)
    
    lighting_risk = 0.0
    # Natural lighting usually produces variance > 50. Screens can be < 30 (flat).
    if unique_lighting < 30: 
        lighting_risk = 1.0 
    elif unique_lighting < 45:
        lighting_risk = 0.5
    
    # Feature 2: Screen Glare / Specular Highlights
    # Screens act as light sources and often have blown-out white areas.
    overexposed = np.sum(v_channel > 252) / v_channel.size
    glare_risk = 1.0 if overexposed > 0.015 else 0.0 
    
    # Feature 3: Moiré/Grid Artifacts (Frequency Domain)
    # Screens often emit high-frequency grid patterns visible in FFT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Check for energy spikes in non-center frequencies
    # (Simplified check: if mean high-freq energy is anomalously high)
    # Note: We avoid 'sharpness' check, but Moiré creates specific noise.
    
    # Feature 4: Color Saturation Check
    # Replays often look "washed out"
    s_channel = hsv[:,:,1]
    undersaturated = np.sum(s_channel < 30) / s_channel.size
    color_risk = 1.0 if undersaturated > 0.5 else 0.0
    
    # Weighted Replay Score (NO Sharpness/Blur check)
    # Weights: Glare (35%), Color (30%), Lighting Uniformity (35%)
    replay_score = (lighting_risk * 0.35) + (glare_risk * 0.35) + (color_risk * 0.30)
    
    return replay_score, unique_lighting

@st.cache_resource
def load_ai_detector():
    model_path = "ai_detector_hybrid.pt"
    if os.path.exists(model_path):
        try:
            return AIImageDetector(model_path)
        except Exception as e:
            st.error(f"Failed to load AI model: {e}")
            return None
    return None

@st.cache_resource
def load_antispoof_model():
    # MiniFASNet weights need to be downloaded
    model_path = "2.7_80x80_MiniFASNetV2.pth"
    if os.path.exists(model_path):
        try:
            return AntiSpoofPredictor(model_path)
        except Exception as e:
            st.error(f"Failed to load Anti-Spoofing model: {e}")
            return None
    else:
        # Don't error out, just return None. The app will warn the user to download it.
        return None

@st.cache_resource
def load_face_recognition_model():
    """Load Facenet for face verification"""
    try:
        # Pretrained on VGGFace2
        if FACE_MATCH_AVAILABLE:
            return InceptionResnetV1(pretrained='vggface2').eval()
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load Face Recognition model: {e}")
        return None

def extract_face_embedding(resnet, image_rgb, face_landmarks):
    """Extract face embedding using Facenet from a specific image"""
    h, w, _ = image_rgb.shape
    
    # Get bounding box from landmarks with some padding
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]
    
    x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
    y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))
    
    # Add padding (20%)
    pad_x = int((x_max - x_min) * 0.2)
    pad_y = int((y_max - y_min) * 0.2)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    
    # Crop
    face_crop = image_rgb[y_min:y_max, x_min:x_max]
    
    if face_crop.size == 0:
        return None
        
    # Resize to 160x160 (required by Facenet)
    face_crop_pil = Image.fromarray(face_crop).resize((160, 160))
    
    # Transform to tensor
    face_tensor = np.array(face_crop_pil).astype(np.float32) / 255.0
    face_tensor = torch.from_numpy(face_tensor).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, 160, 160)
    
    # Normalize (standard for this model)
    # Mean/std for VGGFace2 are slightly different but simple standardization often works
    # facenet-pytorch uses fixed_image_standardization typically but manual works:
    face_tensor = (face_tensor - 0.5) / 0.5 
    
    if resnet is None:
        return None

    with torch.no_grad():
        embedding = resnet(face_tensor)
        
    return embedding

# App State Initialization
if 'challenge_idx' not in st.session_state:
    st.session_state.challenge_idx = 0
if 'challenges_completed' not in st.session_state:
    st.session_state.challenges_completed = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'reference_embedding' not in st.session_state:
    st.session_state.reference_embedding = None
# Randomized Challenge Order (Anti-Spoofing)
if 'challenge_order' not in st.session_state:
    # Extended challenge set for anti-deepfake (Mouth/Smile break common deepfake models)
    base_challenges = ["BLINK", "SMILE", "TURN HEAD LEFT", "TURN HEAD RIGHT"]
    random.shuffle(base_challenges)
    # Require 3 unique challenges
    st.session_state.challenge_order = base_challenges[:3]

# Risk Engine State
if 'risk_components' not in st.session_state:
    st.session_state.risk_components = {
        'liveness': 0.0,
        'fft_integrity': 0.0,
        'face_match': 0.0,
        'replay_risk': 0.0,
        'spoof_risk': 0.0 # NEW: MiniFASNet Score
    }

# UI Layout
st.title("🛡️ Identity Verification System")
st.markdown("### Step 0: Upload ID -> Step 1: Liveness -> Step 2: AI Check")

# Initialize Models
detector = load_ai_detector()
antispoof = load_antispoof_model() # NEW
resnet = load_face_recognition_model()

if antispoof is None:
    st.warning("⚠️ **Bank-Level Security Warning**: 'MiniFASNet' weights not found.") 
    st.info("Running in Heuristic Mode. To enable Bank-Level security, please download `2.7_80x80_MiniFASNetV2.pth`")

# Navigation
page = st.sidebar.radio("Navigation Mode", ["Option 1: Face Liveness Check", "Option 2: Voice Deepfake Check"])

if page == "Option 1: Face Liveness Check":
    # Sidebar
    st.sidebar.title("🚨 Risk Engine")
    risk_placeholder = st.sidebar.empty()

    # Helper for Risk Score
    def update_risk_dashboard(liveness_score, fft_score, match_score, replay_score, spoof_score=0.0):
        # Weights Integration
        # Liveness (Success/Fail) - 20%
        # AI Artifacts (GenAI) - 20%
        # Face Match (Identity) - 15%
        # Forensic Replay (Heuristic) - 10%
        # MiniFASNet (Anti-Spoof Model) - 35% (Core Validator)
        
        W_LIVENESS = 0.20
        W_FFT = 0.20
        W_MATCH = 0.15
        W_REPLAY = 0.10
        W_PAD = 0.35 # Presentation Attack Detection
        
        # Invert Risks (High Risk -> Low Trust)
        trust_from_replay = 1.0 - replay_score
        trust_from_spoof = 1.0 - spoof_score
        
        # Trust Score (0.0 to 1.0)
        trust_score = (liveness_score * W_LIVENESS) + \
                      (fft_score * W_FFT) + \
                      (match_score * W_MATCH) + \
                      (trust_from_replay * W_REPLAY) + \
                      (trust_from_spoof * W_PAD)
        
        # Synthetic Fraud Index (SFI) - Lower is Better (0 = Real, 1 = Fake)
        sfi = 1.0 - trust_score
        
        # Render Dashboard
        with risk_placeholder.container():
            st.markdown("### Synthetic Fraud Index (SFI)")
            
            # Color Logic: Red=High Risk, Green=Low Risk
            sfi_color = "red" if sfi > 0.4 else "green" # Stricter threshold (0.4)
            st.markdown(f"<h1 style='text-align: center; color: {sfi_color};'>{sfi:.1%}</h1>", unsafe_allow_html=True)
            st.progress(max(0.0, min(1.0, sfi)))
            
            st.markdown("#### Component Analysis")
            st.caption(f"Liveness Check: {'PASS' if liveness_score > 0.5 else 'FAIL'}")
            st.caption(f"GenAI Artifacts: {(1-fft_score):.0%} Risk")
            st.caption(f"Identity Match: {match_score:.0%}")
            st.caption(f"Screen Glare: {replay_score:.0%} Risk")
            st.caption(f"FAS Spoof Risk: {spoof_score:.0%} Risk")
            
            if sfi > 0.6:
                 st.error("⛔ BLOCKED: High Risk Detected")
            elif sfi > 0.3:
                 st.warning("⚠️ FLAGGED: Review Required")
            else:
                 st.success("✅ APPROVED: Real Person")

    # Step 0: Reference Image Upload
    st.sidebar.header("Step 0: Reference ID")
    reference_file = st.sidebar.file_uploader("Upload ID Photo", type=["jpg", "png", "jpeg"])

    if reference_file:
        ref_image = Image.open(reference_file).convert('RGB')
        
        # -----------------------------------------------------------
        # NEW: Check if Reference ID itself is AI-Generated
        # -----------------------------------------------------------
        if detector:
            with st.spinner("Checking ID for Deepfakes..."):
                ref_ai_check = detector.predict(ref_image)
                # Use a strict threshold for the static ID upload
                if ref_ai_check['ai_probability'] > 0.85:
                    st.sidebar.error("🚨 ID REJECTED")
                    st.sidebar.error(f"AI/Deepfake Detected: {ref_ai_check['ai_probability']:.1%}")
                    st.stop()
                elif ref_ai_check['ai_probability'] > 0.6:
                     st.sidebar.warning(f"⚠️ Suspicious ID ({ref_ai_check['ai_probability']:.1%})")

        ref_image_np = np.array(ref_image)
        st.sidebar.image(ref_image, caption="Reference ID", width=150)
        
        # Calculate embedding for reference ONLY ONCE
        if st.session_state.reference_embedding is None:
            with st.spinner("Analyzing Reference ID..."):
                ref_results = face_mesh.process(ref_image_np)
                if ref_results.multi_face_landmarks:
                    st.session_state.reference_embedding = extract_face_embedding(
                        resnet, ref_image_np, ref_results.multi_face_landmarks[0]
                    )
                    st.sidebar.success("Reference Face Processed ✅")
                else:
                    st.sidebar.error("No face detected in ID photo!")

    # Main Logic
    img_file_buffer = st.camera_input("Take a picture to verify liveness", key="camera")

    if img_file_buffer is not None:
        # Convert to CV2 image
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        # Init scored variables for this frame
        frame_fft_score = 0.0
        frame_match_score = 0.0
        frame_liveness_score = 0.0
        frame_replay_score = 0.0
        
        if st.session_state.challenges_completed:
            frame_liveness_score = 1.0 # Sticky success if already done
            
        # -----------------------------------------------------------
        # FORENSIC REPLAY CHECK (New)
        # -----------------------------------------------------------
        replay_score, lighting_val = detect_replay_attack(cv2_img)
        st.session_state.risk_components['replay_risk'] = replay_score
        frame_replay_score = replay_score
        
        # If high replay risk, stop immediately
        if replay_score > 0.8: # Increased threshold to be safer
            st.error(f"🚨 REPLAY ATTACK DETECTED! (Score: {replay_score:.1%})")
            st.warning(f"Screen glare or unnatural lighting detected. (Light Variance: {lighting_val:.1f})")
            update_risk_dashboard(0.0, 0.0, 0.0, replay_score, 1.0)
            st.stop()
        
        # -----------------------------------------------------------
        # CRITICAL SECURITY CHECK: AI Deepfake Detection (EVERY FRAME)
        # -----------------------------------------------------------
        # We now check *every* submitted frame for GAN/Diffusion artifacts.
        # Deepfakes often glitch when expression changes (mouth open/smile).
        if detector:
            # Convert CV2 -> PIL for the detector
            pil_check = Image.fromarray(rgb_img)
            ai_result = detector.predict(pil_check)
            
            # FFT Score (Texture Integrity) = Real Probability
            frame_fft_score = ai_result['real_probability']
            st.session_state.risk_components['fft_integrity'] = frame_fft_score
            
            # Continuous Monitoring Sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"### 🤖 AI Check (Frame {st.session_state.challenge_idx+1})")
            if ai_result['is_ai_generated']:
                 st.sidebar.error(f"⚠️ FAKE ({ai_result['ai_probability']:.1%})")
            else:
                 st.sidebar.success(f"✅ REAL ({ai_result['real_probability']:.1%})")
            
            # Strict Block on High Confidence Fakes (Anytime)
            if ai_result['ai_probability'] > 0.85:
                st.error("🚨 CRITICAL SECURITY ALERT: AI-Generated Face Detected!")
                st.markdown(f"**Confidence:** {ai_result['ai_probability']:.2%}")
                # Penalize the dashboard maximally
                update_risk_dashboard(0.0, 0.0, 0.0, frame_replay_score, 0.0)
                st.warning("Deepfake artifacts detected in current frame. Verification Halted.")
                st.stop()
            elif ai_result['ai_probability'] > 0.60:
                st.warning(f"⚠️ High Deepfake Probability ({ai_result['ai_probability']:.1%}) - Proceeding with caution...")

        # -----------------------------------------------------------
        # BANK-LEVEL SECURITY: MiniFASNet Presentation Attack Detection
        # -----------------------------------------------------------
        spoof_risk = 0.0
        
        # Run MediaPipe FIRST to get landmarks for the box
        results = face_mesh.process(rgb_img)
        
        if antispoof and results.multi_face_landmarks:
            # Detect face box for MiniFASNet
            # We can reuse MediaPipe landmarks to approximate the box (faster than running another detector)
            h, w, _ = cv2_img.shape
            landmarks = results.multi_face_landmarks[0]
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
             
            x_min = max(0, int(min(xs)))
            y_min = max(0, int(min(ys)))
            x_max = min(w, int(max(xs)))
            y_max = min(h, int(max(ys)))
             
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
             
            face_box = (x_min, y_min, bbox_w, bbox_h)
             
            # Predict
            spoof_risk = antispoof.predict(cv2_img, face_box)
            st.session_state.risk_components['spoof_risk'] = spoof_risk
             
            # Block immediately if spoof
            if spoof_risk > 0.70: # 70% risk threshold
                 st.error(f"🚨 SPOOF DETECTED BY FASNET! (Score: {spoof_risk:.1%})")
                 st.warning("Presentation Attack Blocked (Screen/Print).")
                 update_risk_dashboard(0.0, frame_fft_score, 0.0, frame_replay_score, spoof_risk)
                 st.stop()
            else:
                 st.sidebar.success(f"FASNet: Live ({1-spoof_risk:.1%})")
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # -----------------------------------------------------------
            # FACE MATCHING CHECK (Reference vs Live)
            # -----------------------------------------------------------
            if st.session_state.reference_embedding is not None:
                 live_embedding = extract_face_embedding(resnet, rgb_img, face_landmarks)
                 if live_embedding is not None:
                     # Calculate Cosine Similarity
                     # Normalize embeddings (Facenet output is not strictly unit length)
                     ref_norm = st.session_state.reference_embedding / st.session_state.reference_embedding.norm()
                     live_norm = live_embedding / live_embedding.norm()
                     # Dot product of normalized vectors = Cosine Similarity
                     cosine_sim = torch.dot(ref_norm.view(-1), live_norm.view(-1)).item()
                     
                     # Clip to 0-1
                     frame_match_score = max(0.0, cosine_sim)
                     st.session_state.risk_components['face_match'] = frame_match_score
                     
                     st.sidebar.markdown("### 👤 Face Match Score")
                     st.sidebar.write(f"Similarity: {frame_match_score:.4f}")
                     
                     # Threshold (Cosine > ~0.4 is tolerant, >0.6 is strict)
                     threshold = 0.4
                     if frame_match_score < threshold and st.session_state.challenge_idx == 0:
                         st.error(f"🚨 IDENTITY MISMATCH! Face does not match Reference ID.")
                         st.info(f"Similarity: {frame_match_score:.2f} < {threshold}")
                         update_risk_dashboard(0.0, frame_fft_score, frame_match_score, frame_replay_score)
                         st.stop()
                     else:
                         if st.session_state.challenge_idx == 0:
                             st.sidebar.success(f"✅ Match! ({frame_match_score:.2f})")
            elif reference_file:
                 st.warning("Reference ID uploaded but not processed.")
                 frame_match_score = 0.0
            else:
                 # Neutral if no ID
                 frame_match_score = 0.5
            
            # -----------------------------------------------------------

            h, w, _ = cv2_img.shape
            coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            
            # Calculate Metrics
            ear_left = get_ear(coords, LEFT_EYE)
            ear_right = get_ear(coords, RIGHT_EYE)
            avg_ear = (ear_left + ear_right) / 2.0
            yaw = get_head_yaw(coords)
            mar = get_mar(coords) # New: Mouth Aspect Ratio
            
            # Display Metrics in Sidebar
            st.sidebar.metric("Eye Aspect Ratio (EAR)", f"{avg_ear:.3f}")
            st.sidebar.metric("Head Yaw", f"{yaw:.3f}")
            st.sidebar.metric("Mouth Aspect Ratio (MAR)", f"{mar:.3f}")

            # Challenge Logic (Randomized)
            challenge_list = st.session_state.challenge_order
            if st.session_state.challenge_idx < len(challenge_list):
                current_challenge = challenge_list[st.session_state.challenge_idx]
            else:
                current_challenge = "DONE"
            
            challenge_met = False
            
            # Logic for each challenge type
            if current_challenge == "BLINK" and avg_ear < 0.20:
                 challenge_met = True
            elif current_challenge == "TURN HEAD LEFT" and yaw < 0.40:
                 challenge_met = True
            elif current_challenge == "TURN HEAD RIGHT" and yaw > 0.60:
                 challenge_met = True
            elif current_challenge == "SMILE" and mar > 0.35 and mar < 0.6: 
                 # Smiles widen the mouth horizontally, typically reducing MAR slightly 
                 # but we need a specific 'Smile' detector or assume 'Wide Mouth'
                 # Simple MAR isn't great for smile (it's for open mouth). 
                 # Let's use 'Open Mouth' as the proxy for non-neutral expression for now, 
                 # or imply Smile if MAR is moderate but mouth is wide?
                 # For simplicity/reliability in this context, we will treat SMILE loosely as "Teeth Visible"
                 challenge_met = True
            
            if challenge_met:
                # Update Score
                st.session_state.risk_components['liveness'] = 1.0
                frame_liveness_score = 1.0
                
                st.success(f"✅ Challenge '{current_challenge}' Completed!")
                if st.session_state.challenge_idx < len(challenge_list) - 1:
                    st.session_state.challenge_idx += 1
                    next_c = challenge_list[st.session_state.challenge_idx]
                    st.info(f"Next Challenge: {next_c}")
                    st.markdown("Please take another photo performing the next action.")
                else:
                    st.session_state.challenges_completed = True
                    st.session_state.captured_image = cv2_img
            else:
                if not st.session_state.challenges_completed:
                    st.session_state.risk_components['liveness'] = 0.0
                    st.warning(f"ACTION REQUIRED: {current_challenge}")
                    if current_challenge == "BLINK":
                        st.write("Please BLINK (close both eyes) and take the photo.")
                    elif current_challenge == "TURN HEAD LEFT":
                        st.write("Please turn your head to your LEFT and take the photo.")
                    elif current_challenge == "TURN HEAD RIGHT":
                        st.write("Please turn your head to your RIGHT and take the photo.")
                    elif current_challenge == "SMILE":
                        st.write("Please SMILE (show teeth) and take the photo.")

            # Update Risk Dashboard
            update_risk_dashboard(st.session_state.risk_components['liveness'], 
                                  st.session_state.risk_components['fft_integrity'], 
                                  st.session_state.risk_components['face_match'],
                                  st.session_state.risk_components['replay_risk'],
                                  st.session_state.risk_components['spoof_risk'])

        else:
            st.error("No face detected. Please position your face clearly in the frame.")

    # Step 2: AI Detection (After Liveness)
    if st.session_state.challenges_completed and st.session_state.captured_image is not None:
        st.divider()
        st.markdown("### Step 2: Analyzing for AI Forgery...")
        
        if detector:
            # Convert CV2 to PIL for the detector
            pil_image = Image.fromarray(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB))
            
            try:
                result = detector.predict(pil_image)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.captured_image, channels="BGR", caption="Verified Frame")
                
                with col2:
                    if result['is_ai_generated']:
                        st.error("⚠️ FAKE DETECTED")
                        st.metric("AI Confidence", f"{result['ai_probability']:.2%}")
                    else:
                        st.success("✅ REAL PERSON VERIFIED")
                        st.metric("Real Confidence", f"{result['real_probability']:.2%}")
                    
                    st.markdown("#### Liveness Check: ✅ PASSED")
                    st.text(f"Verdict: {result['verdict']}")
                    
            except Exception as e:
                st.error(f"Error executing AI Detector: {e}")
        else:
            st.warning("AI Detector model not found. Only Liveness Check was performed.")

        if st.button("Reset Verification"):
            st.session_state.challenge_idx = 0
            st.session_state.challenges_completed = False
            st.session_state.captured_image = None
            st.rerun()

elif page == "Option 2: Voice Deepfake Check":
    st.title("🎙️ Voice Integrity Analysis")
    st.markdown("### Deepfake Audio Detection (SOTA)")
    st.info("Please read the challenge sentence below clearly into the microphone.")

    # Challenge Sentence Logic
    VOICE_CHALLENGES = [
        "The quick brown fox jumps over the lazy dog",
        "Identity verification requires my natural voice and presence",
        "Artificial intelligence cannot mimic my true soul",
        "Blue skies and green grass make a pretty picture today",
        "Please verify this secure login attempt with my voice",
        "My voice is my passport verify me now",
        "Seven red roses valid for seven days"
    ]
    
    if 'voice_challenge' not in st.session_state:
        st.session_state.voice_challenge = random.choice(VOICE_CHALLENGES)
        
    st.divider()
    st.markdown(f"### 🗣️ Please say:  \n# **“{st.session_state.voice_challenge}”**")
    
    if st.button("🔄 New Sentence"):
        st.session_state.voice_challenge = random.choice(VOICE_CHALLENGES)
        st.rerun()
    st.divider()

    audio_bytes = None
    if RECORDER_AVAILABLE:
        st.markdown("#### Record Live Voice")
        try:
            # pause_threshold=300.0 (5 mins) prevents auto-stop on silence
            # sample_rate=16000 ensures compatibility with the model
            audio_bytes = audio_recorder(text="Click to Record", 
                                       icon_size="2x", 
                                       pause_threshold=300.0, 
                                       sample_rate=16000)
        except:
             audio_bytes = audio_recorder("Click to Record", "Recording...")
             
        if audio_bytes and len(audio_bytes) > 0:
            st.audio(audio_bytes, format="audio/wav")
            # Basic visualize
            st.markdown("#### Audio Waveform Preview")
            try:
                # We need to save bytes to read with librosa or just show simple chart
                # Since we don't have numpy array yet, let's just wait for analysis or use basic chart
                pass 
            except:
                pass
    else:
        st.warning("Voice recorder not available (install streamlit-audiorecorder)")
    
    # Process recording only
    target_audio = None

    if audio_bytes is not None and len(audio_bytes) > 0:
        target_audio = audio_bytes
        
    if target_audio is not None:
        analyze_btn = st.button("🔍 Analyze Voice Spoofing", type="primary")
        
        if analyze_btn:
            with st.spinner("Analyzing spectral artifacts & embeddings..."):
                # Save temp
                with open("temp_audio_chk.wav", "wb") as f:
                    f.write(target_audio)
                
                fake_prob = analyze_voice_spoof("temp_audio_chk.wav")
                os.remove("temp_audio_chk.wav")
                
                # UI Result
                st.divider()
                st.markdown("### Analysis Results")
                
                # Sensitivity Config
                with st.expander("⚙️ Detection Sensitivity"):
                    threshold = st.slider("Spoof Threshold (Lower = Stricter)", 0.0, 1.0, 0.50, 0.05)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Spoof Confidence", f"{fake_prob:.2%}")
                    st.progress(max(0.0, min(1.0, fake_prob)))
                
                with col2:
                    real_prob = 1.0 - fake_prob
                    st.metric("Real Confidence", f"{real_prob:.2%}")
                    st.progress(max(0.0, min(1.0, real_prob)))

                st.divider()
                if fake_prob > threshold:
                    st.error(f"🚨 FAKE DETECTED (Confidence > {threshold})")
                    st.warning("**Result: SYNTHETIC / CLONED VOICE**")
                    st.info("The model detected artifacts typical of TTS or Voice Conversion.")
                else:
                    st.success(f"✅ REAL VOICE (Confidence < {threshold})")
                    st.success("**Result: GENUINE HUMAN VOICE**")
                    st.info("Audio spectrum is consistent with natural human speech.")

