# import streamlit as st
# import numpy as np
# import librosa
# import tensorflow as tf
# import librosa.display
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import LabelEncoder
# import os

# # Load or define the model
# model_path = "audio_classification_model.h5"
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     # Build a new model if not found
#     model = Sequential([
#         LSTM(64, input_shape=(100, 13), return_sequences=False),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     # Save the model for future use
#     model.save(model_path)

# # Function to preprocess the audio
# def preprocess_audio(file_path, max_length=100):
#     audio, sr = librosa.load(file_path, sr=22050)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     if mfccs.shape[1] < max_length:
#         mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
#     else:
#         mfccs = mfccs[:, :max_length]
#     return mfccs.T

# # Streamlit UI
# st.title("Deepfake Audio Detection")
# st.write("Upload an audio file to check if it is Real or Fake.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# if uploaded_file is not None:
#     # Save the uploaded file
#     file_path = "temp_audio.wav"
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Preprocess and predict
#     sample = preprocess_audio(file_path)
#     sample = np.expand_dims(sample, axis=0)
#     prediction_prob = model.predict(sample)[0][0]
    
#     # Confidence scores
#     real_confidence = prediction_prob * 100
#     fake_confidence = (1 - prediction_prob) * 100
    
#     st.audio(file_path, format='audio/wav')
    
#     # Display the results
#     st.write(f"Confidence that the audio is **Real**: {real_confidence:.2f}%")
#     st.write(f"Confidence that the audio is **Fake**: {fake_confidence:.2f}%")
    
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
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the improved pre-trained model
model_path = "DeepFakeAudio/audio_classification_model_bilstm1.h5"  # Updated model filename
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Feature extraction parameters
N_MELS = 128  # Using Mel Spectrogram instead of MFCC
MAX_LENGTH = 200  # Adjust for time step consistency

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Ensure consistent shape (pad or truncate)
    mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, max(0, MAX_LENGTH - mel_spectrogram_db.shape[1]))), mode='constant')
    mel_spectrogram_db = mel_spectrogram_db[:, :MAX_LENGTH]
    
    return np.expand_dims(mel_spectrogram_db.T, axis=0)  # Shape: (1, MAX_LENGTH, N_MELS)

# Streamlit UI
st.title("Deepfake Audio Detection (CNN + BiLSTM)")
st.write("Upload an audio file to check if it is Real or Fake.")

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
    
    # Display results
    if prediction_prob > 0.75:
        st.success(f"The audio is classified as **Real** with {real_confidence:.2f}% confidence.")
    elif prediction_prob < 0.25:
        st.error(f"The audio is classified as **Fake** with {fake_confidence:.2f}% confidence.")
    else:
        st.warning(f"The classification is uncertain. Confidence - Real: {real_confidence:.2f}%, Fake: {fake_confidence:.2f}%.")
    
    # Plot Mel Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS), ref=np.max), sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

