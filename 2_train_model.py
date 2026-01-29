import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Setup - Only load what actually exists in your data folder
data_path = 'data'
actions = np.array([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
print(f"📂 Found folders for: {actions}")

# 2. Load Data
for action in actions:
    action_path = os.path.join(data_path, action)
    for sequence in range(30):
        window = []
        for frame_num in range(30):
            file_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                window.append(np.load(file_path))
        if len(window) == 30:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3. Build Model - Using dynamic number of actions
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Dynamic output

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("🚀 Training starting...")
model.fit(X_train, y_train, epochs=200)
model.save('neurosign_model.h5')
print("🎯 Success! 'neurosign_model.h5' is ready.")