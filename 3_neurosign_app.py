import cv2
import numpy as np
import tensorflow as tf
from utils import mp_holistic, extract_landmarks, mp_drawing
import pyttsx3

# 1. Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150) 

# 2. Load your trained model
model = tf.keras.models.load_model('neurosign_model.h5')
actions = np.array(['hello', 'thanks', 'help_urgent'])

# Variables for the "Sliding Window"
sequence = []
sentence = []
predictions = []
threshold = 0.7  # Only speak if confidence is > 70%

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Draw landmarks on screen for a "pro" look
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Logic: Prediction
        keypoints = extract_landmarks(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only the last 30 frames

        if len(sequence) == 30:
            # AI makes a prediction on the 30-frame sequence
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            current_action = actions[np.argmax(res)]
            
            # Logic: Smoothing & Speaking
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0:
                    if current_action != sentence[-1]:
                        sentence.append(current_action)
                        engine.say(current_action)
                        engine.runAndWait()
                else:
                    sentence.append(current_action)

            if len(sentence) > 5: sentence = sentence[-5:]

        # UI: Show the translated text on screen
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('NeuroSign AI - Live Translator', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()