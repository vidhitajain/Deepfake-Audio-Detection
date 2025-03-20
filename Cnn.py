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
MAX_LENGTH = 173  # Adjust based on your audio data
real_audio_dir = r'C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio\dataset\real'
fake_audio_dir = r'C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio\dataset\fake'

# st.write(f"DATASET_PATH: {DATASET_PATH}")  # Debugging Prints
# st.write(f"real_audio_dir: {real_audio_dir}")
# st.write(f"fake_audio_dir: {fake_audio_dir}")

# --- Feature Extraction and Data Loading ---
def extract_features(audio_path, max_length=MAX_LENGTH):
    try:
        audio, sr = librosa.load(audio_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        padded_mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)
        return padded_mfccs.T
    except Exception as e:
        st.error(f"Error processing {audio_path}: {e}")
        return None

def load_data(dataset_path):
    real_audio_dir = os.path.join(dataset_path, 'real')
    fake_audio_dir = os.path.join(dataset_path, 'fake')

    if not os.path.exists(real_audio_dir) or not os.path.exists(fake_audio_dir):
        st.error("Real or fake audio directories not found. Please check your dataset path.")
        return None, None

    real_files = [os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) if f.endswith(('.wav', '.mp3'))]
    fake_files = [os.path.join(fake_audio_dir, f) for f in os.listdir(fake_audio_dir) if f.endswith(('.wav', '.mp3'))]

    real_labels = [0] * len(real_files)
    fake_labels = [1] * len(fake_files)

    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    features = []
    processed_labels = []
    for path, label in zip(all_files, all_labels):
        feature = extract_features(path)
        if feature is not None:
            features.append(feature)
            processed_labels.append(label)

    if not features:
        st.error("No audio files were successfully loaded. Please check your audio files.")
        return None, None

    return np.array(features), np.array(processed_labels)

# --- Model Creation ---
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Streamlit App ---
st.title("Deepfake Audio Detection")

if st.button("Train Model"):
    features, labels = load_data(DATASET_PATH)

    if features is not None and labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_cnn_model(input_shape)

        with st.spinner("Training model..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)  # verbose=0 to suppress training output.

        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        st.write("Model Trained!")
        # st.write("Accuracy:", accuracy_score(y_test, y_pred))
        # st.write("Classification Report:\n", classification_report(y_test, y_pred))

        st.session_state.trained_model = model  # Store model for later use.
        st.success("Model training complete.")

if 'trained_model' in st.session_state:
    uploaded_file = st.file_uploader("Upload an audio file for deepfake detection", type=['wav', 'mp3'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.getvalue())
            temp_audio_path = temp_audio.name  # Store the temp file path

        feature = extract_features(temp_audio_path)

        if feature is not None:
            feature = np.expand_dims(feature, axis=0)  # Add batch dimension
            prediction = st.session_state.trained_model.predict(feature)


            if prediction[0][0] > 0.5:
                st.write("Prediction: Deepfake Audio")
            else:
                st.write("Prediction: Real Audio")

        os.remove(temp_audio_path)  # Remove temporary file after use
