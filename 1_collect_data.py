import cv2
import os
import numpy as np
from utils import mp_holistic, extract_landmarks

# SETTINGS: Define what signs you want to teach the AI
actions = np.array(['hello', 'thanks', 'help_urgent'])
no_sequences = 30 # Number of videos per sign
sequence_length = 30 # Number of frames per video

# Create folders
for action in actions:
    os.makedirs(os.path.join('data', action), exist_ok=True)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                # Extract and save
                keypoints = extract_landmarks(results)
                npy_path = os.path.join('data', action, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)

                cv2.putText(frame, f'RECORDING: {action} Sequence {sequence}', (15,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()