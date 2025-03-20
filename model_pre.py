import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import librosa
import os
import tempfile

# --- Configuration ---
DATASET_PATH = r"C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio\dataset"  # Replace with your actual dataset path
MAX_LENGTH = 175  # Adjust based on dataset's longest MFCC sequence
N_MFCC = 40  # Number of MFCCs to extract
REAL_AUDIO_DIR = os.path.join(DATASET_PATH, "real")
FAKE_AUDIO_DIR = os.path.join(DATASET_PATH, "fake")

# --- Feature Extraction ---
def extract_features(audio_path, max_length=MAX_LENGTH, n_mfcc=N_MFCC):
    try:
        audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Pad/truncate to ensure fixed-length feature extraction
        if mfccs.shape[1] < max_length:
            padded_mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            padded_mfccs = mfccs[:, :max_length]

        return padded_mfccs.T
    except Exception as e:
        st.error(f"Error processing {audio_path}: {e}")
        return None

# --- Load Dataset ---
def load_data(dataset_path):
    real_files = [os.path.join(REAL_AUDIO_DIR, f) for f in os.listdir(REAL_AUDIO_DIR) if f.endswith(('.wav', '.mp3'))]
    fake_files = [os.path.join(FAKE_AUDIO_DIR, f) for f in os.listdir(FAKE_AUDIO_DIR) if f.endswith(('.wav', '.mp3'))]

    real_labels = [0] * len(real_files)
    fake_labels = [1] * len(fake_files)

    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    features, processed_labels = [], []
    for path, label in zip(all_files, all_labels):
        feature = extract_features(path)
        if feature is not None:
            features.append(feature)
            processed_labels.append(label)

    if not features:
        st.error("No valid audio files found! Check your dataset path.")
        return None, None

    return np.array(features), np.array(processed_labels)

# --- CNN Model ---
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Streamlit UI ---
st.title("Deepfake Audio Detection")

# Train Model Button
if st.button("Train Model"):
    features, labels = load_data(DATASET_PATH)

    if features is not None and labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_cnn_model(input_shape)

        with st.spinner("Training model..."):
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        st.write("Model Trained!")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Classification Report:", classification_report(y_test, y_pred))

        st.session_state.trained_model = model  # Store model for later use
        st.success("Model training complete.")

# Upload File for Detection
if 'trained_model' in st.session_state:
    uploaded_file = st.file_uploader("Upload an audio file for deepfake detection", type=['wav', 'mp3'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.getvalue())
            temp_audio_path = temp_audio.name  # Store temp file path

        # Extract Features from Uploaded Audio
        feature = extract_features(temp_audio_path)

        if feature is not None:
            feature = np.expand_dims(feature, axis=0)  # Add batch dimension
            prediction = st.session_state.trained_model.predict(feature)[0][0]

            # st.write(f"Prediction Score: {prediction:.4f}")  # Show raw probability

            if prediction < 0.5:
                st.write(" **Deepfake Detected** ")
            else:
                st.write(" **Real Audio** ")

        os.remove(temp_audio_path)  # Remove temp file after use
