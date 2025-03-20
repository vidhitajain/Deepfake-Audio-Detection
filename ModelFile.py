# # import numpy as np
# # import tensorflow as tf
# # import librosa
# # import librosa.display
# # import os
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# # # Set Parameters
# # N_MFCC = 13  # Ensure this matches your preprocessing
# # MAX_LENGTH = 150  # Ensure same as in your inference script

# # def preprocess_audio(file_path):
# #     audio, sr = librosa.load(file_path, sr=22050)
# #     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    
# #     # Ensure consistent input shape
# #     mfccs = np.pad(mfccs, ((0, 0), (0, max(0, MAX_LENGTH - mfccs.shape[1]))), mode='constant')
# #     mfccs = mfccs[:, :MAX_LENGTH]
    
# #     return np.expand_dims(mfccs.T, axis=-1)  # Shape: (MAX_LENGTH, N_MFCC, 1)

# # DATASET_PATH = r"C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio/dataset"


# # X, y = [], []
# # for label, category in enumerate(["real", "fake"]):  # Label: Real=0, Fake=1
# #     folder = os.path.join(DATASET_PATH, category)
# #     for file in os.listdir(folder):
# #         if file.endswith(".wav"):  # Ensure only WAV files are processed
# #             file_path = os.path.join(folder, file)
# #             feature = preprocess_audio(file_path)
# #             X.append(feature)
# #             y.append(label)

# # X = np.array(X)  # Convert to NumPy array
# # y = tf.keras.utils.to_categorical(y, num_classes=2)  # One-hot encoding


# # model = Sequential([
# #     Conv2D(32, (3, 3), activation='relu', input_shape=(MAX_LENGTH, N_MFCC, 1)),
# #     MaxPooling2D((2, 2)),
# #     Conv2D(64, (3, 3), activation='relu'),
# #     MaxPooling2D((2, 2)),
# #     Flatten(),
# #     Dense(128, activation='relu'),
# #     Dropout(0.3),
# #     Dense(2, activation='softmax')  # 2 Classes: Real/Fake
# # ])

# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # model.summary()

# # model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# # model.save("audio_classification_model.h5")
# # print("Model saved successfully!")

# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import tensorflow as tf
# import os
# from tempfile import NamedTemporaryFile
# from sklearn.metrics import accuracy_score

# # Load trained model
# model = tf.keras.models.load_model("audio_model.h5")

# # Function to preprocess audio for prediction
# def preprocess_audio(filepath, max_length=100, n_mfcc=20):
#     audio, sr = librosa.load(filepath, sr=22050)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')
#     mfcc = mfcc[:, :max_length]  # Trim or pad to fixed length
    
#     return mfcc.T  # **Remove extra dimension**0

# def plot_audio_features(audio, sr, mfcc):
#     fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
#     # Plot waveform
#     librosa.display.waveshow(audio, sr=sr, ax=ax[0])
#     ax[0].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
    
#     # Plot MFCCs
#     img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax[1])
#     fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
#     ax[1].set(title='MFCC', xlabel='Time (s)', ylabel='MFCC Coefficients')
    
#     st.pyplot(fig)


# # Streamlit UI
# st.title("🔊 DeepFake Audio Detection")
# st.write("Upload an audio file to check if it's real or fake.")

# # model = tf.keras.models.load_model("audio_model.h5")

# # # Make predictions
# # y_pred = model.predict(X_test)

# # # Convert probabilities to binary labels
# # y_pred = (y_pred > 0.5).astype("int").flatten()

# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # File uploader
# uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

# if uploaded_file is not None:
#     # Save file temporarily
#     temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
#     temp_audio.write(uploaded_file.read())
#     temp_audio.close()

#     st.audio(temp_audio.name, format='audio/wav')

#     # Process and predict
#     input_features = preprocess_audio(temp_audio.name)
#     input_features = input_features.reshape(1, 100, 20)
#     prediction = model.predict(input_features)

#     # Display result
#     probability = prediction[0][0]
#     # st.write(f"### Prediction Probability: {probability:.4f}")

#     if probability > 0.5:
#         st.error("🚨 **This audio is likely FAKE!**")
#     else:
#         st.success("✅ **This audio is likely REAL!**")

#      # Plot waveform and MFCC
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
#     plot_audio_features(audio, sr, mfcc)

#     # Cleanup temporary file
#     os.remove(temp_audio.name)

import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

# Load trained model
model = tf.keras.models.load_model("audio_model.h5")

# Function to preprocess audio for prediction
def preprocess_audio(filepath, max_length=100, n_mfcc=20):
    audio, sr = librosa.load(filepath, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')
    mfcc = mfcc[:, :max_length]  # Trim or pad to fixed length
    return mfcc.T, audio, sr

# Function to plot waveform and MFCC
def plot_audio_features(audio, sr, mfcc):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
    
    # Plot MFCCs
    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set(title='MFCC', xlabel='Time (s)', ylabel='MFCC Coefficients')
    
    st.pyplot(fig)

# Streamlit UI
st.title("🔊 DeepFake Audio Detection")
st.write("Upload an audio file to")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save file temporarily
    temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(uploaded_file.read())
    temp_audio.close()
    
    # Process audio
    input_features, audio, sr = preprocess_audio(temp_audio.name)
    input_features = input_features.reshape(1, 100, 20)
    
    # Display audio player
    st.audio(temp_audio.name, format='audio/wav')
    
    # Predict
    prediction = model.predict(input_features)
    probability = prediction[0][0]
    
    if probability > 0.5:
        st.error("🚨 **This audio is likely FAKE!**")
    else:
        st.success("✅ **This audio is likely REAL!**")
    
    # Plot waveform and MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    plot_audio_features(audio, sr, mfcc)
    
    # Cleanup temporary file
    os.remove(temp_audio.name)
