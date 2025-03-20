import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


def preprocess_audio(filepath, max_length=100, n_mfcc=20):
    audio, sr = librosa.load(filepath, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')
    mfcc = mfcc[:, :max_length]  # Trim or pad to fixed length
    
    return mfcc.T  # **Remove extra dimension**

# Load dataset
def load_data(real_path, fake_path):
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.wav')]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith('.wav')]

    X, y = [], []
    
    for file in real_files:
        X.append(preprocess_audio(file))
        y.append(0)  # Real audio label

    for file in fake_files:
        X.append(preprocess_audio(file))
        y.append(1)  # Fake audio label

    X = np.array(X).reshape(len(X), 100, 20)  # Ensure shape is (num_samples, 100, 20)
    y = np.array(y)

    return X, y

# Paths to dataset
DATASET_PATH = r"C:\Users\Asus\Downloads\DeepFakeAudio\DeepFakeAudio\dataset"
REAL_AUDIO_PATH = os.path.join(DATASET_PATH, "real")
FAKE_AUDIO_PATH = os.path.join(DATASET_PATH, "fake")

X, y = load_data(REAL_AUDIO_PATH, FAKE_AUDIO_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(100, 20)),  # 100 time-steps, 20 MFCC features
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=16, callbacks=[early_stopping, reduce_lr])

# Save trained model
model.save("audio_model.h5")
print("Model saved successfully!")

model = tf.keras.models.load_model("audio_model.h5")

# Make Predictions
y_pred = model.predict(X_test)

# Convert probabilities to binary labels (0 or 1)
y_pred = (y_pred > 0.5).astype("int").flatten()

# Calculate and Print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")


