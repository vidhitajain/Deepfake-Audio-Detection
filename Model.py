# import streamlit as st
# import numpy as np
# import librosa
# import tensorflow as tf
# import librosa.display
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_modelstr
# import os

# # Load the pre-trained model
# model_path = "audio_classification_model.h5"
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     st.error("Model file not found. Please train and save the model first.")
#     st.stop()

# # Function to preprocess the audio
# def preprocess_audio(file_path, max_length=150):
#     audio, sr = librosa.load(file_path, sr=22050)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Adjusted to match model input
#     mfccs = np.pad(mfccs, ((0, 0), (0, max(0, max_length - mfccs.shape[1]))), mode='constant')
#     mfccs = mfccs[:, :max_length]
#     return np.expand_dims(mfccs.T, axis=-1)  # Ensure shape compatibility

# # Streamlit UI
# st.title("Deepfake Audio Detection")
# st.write("Upload an audio file to check if it is Real or Fake.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# if uploaded_file is not None:
#     file_path = "temp_audio.wav"
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Preprocess and predict
#     sample = preprocess_audio(file_path)
#     sample = np.expand_dims(sample, axis=0)
    
#     try:
#         prediction_prob = model.predict(sample)[0][0]
#     except ValueError as e:
#         st.error(f"Model input shape mismatch: {e}")
#         st.stop()
    
#     real_confidence = prediction_prob * 100
#     fake_confidence = (1 - prediction_prob) * 100
    
#     st.audio(file_path, format='audio/wav')
    
#     # Display results with enhanced clarity
#     if prediction_prob > 0.75:
#         st.success(f"The audio is classified as **Real** with {real_confidence:.2f}% confidence.")
#     elif prediction_prob < 0.25:
#         st.error(f"The audio is classified as **Fake** with {fake_confidence:.2f}% confidence.")
#     else:
#         st.warning(f"The classification is uncertain. Confidence - Real: {real_confidence:.2f}%, Fake: {fake_confidence:.2f}%.")
    
#     # Plot waveform
#     audio, sr = librosa.load(file_path, sr=22050)
#     fig, ax = plt.subplots(figsize=(10, 4))
#     librosa.display.waveshow(audio, sr=sr, alpha=0.6, ax=ax)
#     ax.set_title("Audio Waveform")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude")
#     st.pyplot(fig)
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load the pre-trained model
model_path = "audio_classification_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Ensure MFCC feature shape matches the model's expected input
N_MFCC = 13  # Adjust this based on your model
MAX_LENGTH = 150  # Adjust this based on your training data

# Function to preprocess the audio
def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    
    # Ensure consistent input shape (time steps, features)
    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, MAX_LENGTH - mfccs.shape[1]))), mode='constant')
    mfccs = mfccs[:, :MAX_LENGTH]  # Truncate if needed
    
    return np.expand_dims(mfccs.T, axis=0)  # Shape: (1, MAX_LENGTH, N_MFCC)

# Streamlit UI
st.title("Deepfake Audio Detection")
st.write("Upload an audio file to check if it is Real or Fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess and predict
    sample = preprocess_audio(file_path)
    prediction_prob = model.predict(sample)[0][0]
    
    real_confidence = prediction_prob * 100
    fake_confidence = (1 - prediction_prob) * 100
    
    st.audio(file_path, format='audio/wav')
    
    # Display results with enhanced clarity
    if prediction_prob > 0.60:
        st.success(f"The audio is classified as **Real** with {real_confidence:.2f}% confidence.")
    elif prediction_prob < 0.25:
        st.error(f"The audio is classified as **Fake** with {fake_confidence:.2f}% confidence.")
    else:
        st.warning(f"The classification is uncertain. Confidence - Real: {real_confidence:.2f}%, Fake: {fake_confidence:.2f}%.")
    
    # Plot waveform
    audio, sr = librosa.load(file_path, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, alpha=0.6, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


