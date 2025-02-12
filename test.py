import os
import numpy as np
import pyaudio
import wave
import joblib
import librosa
import librosa.display
import cudf
import cupy as cp
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split, GridSearchCV
from scipy.signal import butter, lfilter
from scipy.spatial.distance import euclidean

# Path to dataset
DATASET_PATH = r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0"

# ----- Audio Preprocessing -----
def record_audio(duration=3, fs=44100, filename="temp.wav"):
    """Records audio from the microphone and saves it as a WAV file."""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
        print("🎤 Recording...")

        frames = [stream.read(1024) for _ in range(0, int(fs / 1024 * duration))]

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        with wave.open(filename, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        return audio.astype(np.float32), fs
    except Exception as e:
        print(f"Recording error: {e}")
        return np.array([]), fs

# ----- Feature Extraction -----
def butter_lowpass_filter(data, cutoff=3000, fs=44100, order=5):
    """Applies a low-pass filter to remove high-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def extract_features(audio, fs):
    """Extracts advanced audio features for authentication."""
    try:
        audio = butter_lowpass_filter(audio)
        audio, _ = librosa.effects.trim(audio, top_db=40)
        n_fft = 1024 if len(audio) >= 512 else 512
        mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=audio, sr=fs, n_fft=n_fft)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=fs, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=fs)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=n_fft)
        delta_mfccs = librosa.feature.delta(mfccs)
        features = np.hstack([
            np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1), 
            np.mean(tonnetz, axis=1), np.mean(mel_spec, axis=1), np.mean(delta_mfccs, axis=1)
        ])
        return cp.array(features)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return cp.zeros(169)

# ----- Model Training -----
def train_voice_model():
    """Trains an SVM voice authentication model using GPU."""
    user_samples = []
    for file in os.listdir(DATASET_PATH):
        if file.endswith(".wav"):
            file_path = os.path.join(DATASET_PATH, file)
            with wave.open(file_path, "rb") as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                fs = wf.getframerate()
            user_samples.append(extract_features(audio, fs))
    
    impostor_samples = [cp.random.randn(169) for _ in range(len(user_samples))]
    X = cp.vstack((user_samples, impostor_samples))
    y = cp.array([1] * len(user_samples) + [0] * len(impostor_samples))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"✅ Training Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
    joblib.dump(model, "voice_auth_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return model, scaler

# ----- Authentication -----
def authenticate_voice():
    """Authenticates a user using GPU-accelerated SVM."""
    try:
        model = joblib.load("voice_auth_model.pkl")
        scaler = joblib.load("scaler.pkl")
        audio, fs = record_audio(3)
        features = extract_features(audio, fs)
        if cp.count_nonzero(features) == 0:
            print("❌ Invalid audio input, please try again!")
            return
        features = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][int(prediction)] * 100
        if prediction == 1 and probability >= 90:
            print(f"✅ Access Granted (Confidence: {probability:.2f}%)")
        else:
            print(f"❌ Access Denied (Confidence: {probability:.2f}%)")
    except Exception as e:
        print(f"Authentication error: {e}")

if __name__ == "__main__":
    model, scaler = train_voice_model()
    authenticate_voice()
    os.remove("temp.wav")
