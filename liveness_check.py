import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark Indices for EAR (Eyes) and Head Pose
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_indices):
    # Vertical distances
    v1 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    v2 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    # Horizontal distance
    h = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (v1 + v2) / (2.0 * h)

def get_head_yaw(landmarks):
    # Simplified Yaw: Ratio of nose distance to face edges
    # Landmark 1: Nose tip, 454: Right edge, 234: Left edge
    nose = landmarks[1][0]
    right_edge = landmarks[454][0]
    left_edge = landmarks[234][0]
    
    # Calculate relative position of nose tip
    face_width = right_edge - left_edge
    relative_nose_pos = (nose - left_edge) / face_width
    return relative_nose_pos # 0.5 is centered, <0.4 is Left, >0.6 is Right

# Try opening camera 0
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    print("Trying camera 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        print("\nPossible solutions:")
        print("1. Ensure your webcam is connected.")
        print("2. If running on a server/cloud, you must provide a video file path.")
        
        # Fallback to video file input
        video_path = input("\nEnter path to video file (or Press Enter to exit): ").strip()
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                exit()
        else:
            exit()

challenge = "BLINK" # Initial challenge
challenge_met = False
start_time = time.time()
challenge_sequence = ["BLINK", "TURN LEFT", "TURN RIGHT"]
current_challenge_index = 0

print("Starting verification... Press ESC to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: 
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            
            # 1. EAR Calculation
            ear_left = get_ear(coords, LEFT_EYE)
            ear_right = get_ear(coords, RIGHT_EYE)
            avg_ear = (ear_left + ear_right) / 2.0
            
            # 2. Head Yaw Calculation
            yaw = get_head_yaw(coords)

            # 3. Challenge Logic
            challenge = challenge_sequence[current_challenge_index]
            
            if challenge == "BLINK" and avg_ear < 0.20:
                challenge_met = True
            elif challenge == "TURN LEFT" and yaw < 0.35: # Flipped image logic: Left becomes Right visually if not careful, but yaw logic depends on coordinates
                # In mirrored image: 
                # Left edge of image is x=0.
                # If I turn my head left, my nose moves to my left (image left).
                # Previous logic: relative_nose_pos < 0.4 is Left.
                challenge_met = True
            elif challenge == "TURN RIGHT" and yaw > 0.65:
                challenge_met = True
                
            if challenge_met:
                print(f"Challenge '{challenge}' Met!")
                time.sleep(1) # Pause briefly to show success
                challenge_met = False
                current_challenge_index += 1
                if current_challenge_index >= len(challenge_sequence):
                    print("ALL CHALLENGES COMPLETED - VERIFICATION SUCCESSFUL")
                    # Here you would typically integrate with the AI detector
                    # E.g. save the current frame and run AI detection
                    # cv2.imwrite("verification_frame.jpg", image)
                    # And then exit or restart
                    current_challenge_index = 0 # Loop for demo purposes
                    # break # Uncomment to stop after success

    # UI Overlay
    challenge = challenge_sequence[current_challenge_index]
    color = (0, 0, 255) # Red for pending
    
    status = f"DO THIS: {challenge}"
    cv2.putText(image, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('KYC Liveness Challenge', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
