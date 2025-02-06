import numpy as np
import pyaudio
import wave
import joblib
import librosa
import librosa.display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter

# ----- Audio Preprocessing -----
def record_audio(duration=3, fs=44100, filename="temp.wav"):
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

        # Read WAV file without scipy.io.wavfile
        with wave.open(filename, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        return audio.astype(np.float32), fs
    except Exception as e:
        print(f"Recording error: {e}")
        return np.array([]), fs

def butter_lowpass_filter(data, cutoff=3000, fs=44100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# ----- Feature Extraction -----
def extract_features(audio, fs):
    try:
        # Noise Reduction
        audio = butter_lowpass_filter(audio)

        # Voice Activity Detection (VAD)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Extract 13 MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(13)

# ----- Model Training -----
def train_voice_model():
    user_samples = [extract_features(record_audio(3)[0], 44100) for _ in range(10)]
    impostor_samples = [extract_features(np.random.randn(44100 * 3), 44100) for _ in range(10)]

    X = np.vstack((user_samples, impostor_samples))
    y = np.array([1] * len(user_samples) + [0] * len(impostor_samples))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print(f"✅ Training Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    # Save model and scaler
    joblib.dump(model, "voice_auth_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler

# ----- Authentication -----
def authenticate_voice():
    try:
        model = joblib.load("voice_auth_model.pkl")
        scaler = joblib.load("scaler.pkl")

        audio, fs = record_audio(3)
        features = extract_features(audio, fs)
        features = scaler.transform([features])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction] * 100

        if prediction == 1 and probability >= 90:
            print(f"✅ Access Granted (Confidence: {probability:.2f}%)")
        else:
            print(f"❌ Access Denied (Confidence: {probability:.2f}%)")
    except Exception as e:
        print(f"Authentication error: {e}")

if __name__ == "__main__":
    model, scaler = train_voice_model()
    authenticate_voice()
