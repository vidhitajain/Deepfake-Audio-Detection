import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa

# Load dataset
df = pd.read_csv(r"C:\\Users\\Asus\\Downloads\\DeepFakeAudio\\DeepFakeAudio\\DATASET-balanced.csv")

# Encode labels
label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['LABEL'])  # FAKE -> 0, REAL -> 1

# Separate features and labels
X = df.drop(columns=['LABEL'])  # Features
y = df['LABEL']  # Target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# Function to predict audio authenticity
def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)  # Ensure 26 MFCC features
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
    if mfccs_processed.shape[1] != 26:
        raise ValueError("Incorrect number of features extracted. Expected 26.")
    mfccs_scaled = scaler.transform(mfccs_processed)
    prediction = model.predict(mfccs_scaled)
    return "REAL" if prediction[0][0] > 0.5 else "FAKE"

# Example usage
audio_file = r"C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio\REAL\trump-original.wav"
try:
    result = predict_audio(audio_file)
    print(f"The audio is classified as: {result}")
except ValueError as e:
    print(f"Error processing audio: {e}")