import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    """Converts MediaPipe results into a flat numpy array of exactly 1662 values."""
    # 1. Pose Landmarks (33 points * 4 values: x, y, z, visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # 2. Face Landmarks (468 points * 3 values: x, y, z) = 1404
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # 3. Left Hand (21 points * 3 values: x, y, z) = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # 4. Right Hand (21 points * 3 values: x, y, z) = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Total: 132 + 1404 + 63 + 63 = 1662
    return np.concatenate([pose, face, lh, rh])