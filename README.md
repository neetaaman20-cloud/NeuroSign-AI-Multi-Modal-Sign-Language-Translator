##🤟 NeuroSign AI: Multi-Modal Sign Language Translator
NeuroSign AI is an "exceptional" accessibility tool that bridges the gap between sign language and spoken word. Unlike standard image classifiers, NeuroSign uses Spatial-Temporal Deep Learning to translate dynamic gestures into real-time speech while incorporating facial landmarks for emotional context.

🌟 Key Features
Real-Time Gesture Recognition: Uses a sliding window of 30 frames to analyze movement sequences rather than static hand shapes.

Affective Computing: Incorporates 468 facial landmarks to provide a more nuanced translation (e.g., distinguishing between a statement and a question).

Sign-to-Voice: Integrated with pyttsx3 to provide immediate auditory feedback for translated signs.

Edge Optimized: Designed to run efficiently on local hardware (tested on MacBook Air) using optimized NumPy coordinate extraction.

🛠️ Tech Stack
Component	Technology
Language	Python 3.x
Deep Learning	TensorFlow / Keras
Computer Vision	OpenCV, MediaPipe
Sequence Processing	Long Short-Term Memory (LSTM) Networks
Speech Synthesis	pyttsx3

🧠 How It Works
NeuroSign AI treats sign language as a "dance" of data points. It follows a three-stage pipeline:

Coordinate Extraction: MediaPipe tracks 21 points on each hand and 468 points on the face.

Temporal Analysis: An LSTM (Long Short-Term Memory) neural network processes the sequence of these coordinates over time.

Model=CNN 
FeatureExtraction
​	
 →LSTM 
TemporalPattern
​	
 →Dense 
Classification
​	
 
Inference & Voice: If the model confidence exceeds a threshold (70%), the predicted action is appended to a "sentence" and converted to speech.

📂 Project Structure

NeuroSign_AI/
├── 1_collect_data.py   # Script to record custom sign sequences
├── 2_train_model.py    # TensorFlow LSTM model training script
├── 3_neurosign_app.py  # The final real-time translation application
├── utils.py            # Landmark extraction and helper functions
├── neurosign_model.h5  # The pre-trained weights (after training)
└── data/               # Local storage for recorded .npy coordinates

1. Installation
Clone the repository and install the dependencies:

git clone https://github.com/neetamaan20-cloud/NeuroSign_AI.git
cd NeuroSign_AI
pip install tensorflow opencv-python mediapipe pyttsx3 scikit-learn

2. Record Your Data

Run the collection script to teach the AI your signs:

python 1_collect_data.py

3. Train and Run

Train the model on your recorded sequences and launch the app:

python 2_train_model.py
python 3_neurosign_app.py

