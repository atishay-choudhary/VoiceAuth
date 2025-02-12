import os
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

def butter_lowpass_filter(data, cutoff=3000, fs=44100, order=5):
    """Applies a low-pass filter to remove high-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# ----- Feature Extraction -----
def zero_pad_audio(audio, target_length=1024):
    """Zero-pads the audio signal to the target length."""
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    return audio

def safe_n_fft(audio_length, default_n_fft=1024):
    """Determines a safe n_fft based on audio length."""
    return min(default_n_fft, audio_length) if audio_length >= 512 else 512

def extract_features(audio, fs):
    """Extracts advanced audio features for better authentication."""
    try:
        audio = butter_lowpass_filter(audio)

        # Voice Activity Detection (VAD)
        audio, _ = librosa.effects.trim(audio, top_db=40)

        # Zero-pad the audio if it's too short
        audio = zero_pad_audio(audio)

        # Dynamically set n_fft
        n_fft = safe_n_fft(len(audio))

        # Extract Features
        mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=audio, sr=fs, n_fft=n_fft)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=fs, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=fs)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=n_fft)
        delta_mfccs = librosa.feature.delta(mfccs)

        # Combine all features
        features = np.hstack([
            np.mean(mfccs, axis=1), 
            np.mean(chroma, axis=1), 
            np.mean(spectral_contrast, axis=1), 
            np.mean(tonnetz, axis=1),
            np.mean(mel_spec, axis=1), 
            np.mean(delta_mfccs, axis=1)
        ])

        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(169)  # Consistent shape for ML model

def load_dataset(dataset_path):
    """Loads user voice samples and returns features along with their filenames."""
    user_samples = []
    filenames = []

    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)

            try:
                with wave.open(file_path, "rb") as wf:
                    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                    fs = wf.getframerate()
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue  # Skip problematic files

            features = extract_features(audio, fs)
            user_samples.append(features)
            filenames.append(file_path)

    return user_samples, filenames

# ----- Model Training -----
def train_voice_model():
    """Trains an SVM voice authentication model using stored user samples."""
    user_samples, filenames = load_dataset(DATASET_PATH)

    # Generate impostor samples
    impostor_samples = [extract_features(np.random.randn(44100 * 3), 44100) for _ in range(len(user_samples))]

    X = np.vstack((user_samples, impostor_samples))
    y = np.array([1] * len(user_samples) + [0] * len(impostor_samples))

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
    joblib.dump(filenames, "user_filenames.pkl")  # Save filenames for authentication

    return model, scaler

# ----- Authentication -----
def authenticate_voice():
    """Authenticates a user and retrieves their closest matching voice sample."""
    try:
        model = joblib.load("voice_auth_model.pkl")
        scaler = joblib.load("scaler.pkl")
        filenames = joblib.load("user_filenames.pkl")

        audio, fs = record_audio(3)
        features = extract_features(audio, fs)

        if np.count_nonzero(features) == 0:
            print("❌ Invalid audio input, please try again!")
            return

        features = scaler.transform([features])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction] * 100

        if prediction == 1 and probability >= 90:
            # Find the closest matching voice sample
            user_samples, filenames = load_dataset(DATASET_PATH)
            closest_sample = min(zip(user_samples, filenames), key=lambda x: euclidean(features.flatten(), x[0]))

            print(f"✅ Access Granted (Confidence: {probability:.2f}%)")
            print(f"🔊 Matched Voice Sample: {closest_sample[1]}")
            
            if closest_sample[1] in ["AD_1.wav","AD_2.wav","AD_3.wav","AD_4.wav","AD_5.wav"]:
                print("Greeting......")
                Greetings1=r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\AD_5.wav"
                # Play greeting (Specify your file path)
                os.system(f"start {Greetings1}")    
            elif closest_sample[1] in ["APS_1.wav","APS_2.wav","APS_3.wav","APS_4.wav","APS_5.wav"]:
                print("Greeting......")
                Greetings2=r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\APS_5.wav"
                # Play greeting (Specify your file path)
                os.system(f"start {Greetings2}")  
            elif closest_sample[1] in ["ATS_1.wav","ATS_2.wav","ATS_3.wav","ATS_4.wav","ATS_5.wav"]:
                print("Greeting......")
                Greetings3=r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\ATS_5.wav"
                # Play greeting (Specify your file path)
                os.system(f"start {Greetings3}")  
            elif closest_sample[1] in ["SJI_1.wav","SJI_2.wav","SJI_3.wav","SJI_4.wav","SJI_5.wav"]:
                print("Greeting......")
                Greetings4=r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\SJI_5.wav"   
                # Play greeting (Specify your file path)
                os.system(f"start {Greetings4}")  
            elif closest_sample[1] in ["PD_1.wav","PD_2.wav","PD_3.wav","PD_4.wav","PD_5.wav"]:
                print("Greeting......") 
                Greetings4=r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\PD_5.wav"   
                # Play greeting (Specify your file path)
                os.system(f"start {Greetings4}")  
        else:
            Greetings0 = r"C:\Users\atish\Documents\VoiceAuth\Audios\user_0\PD.wav"
            print(f"❌ Access Denied (Confidence: {probability:.2f}%)")
            os.system(f"start {Greetings0}")

    except Exception as e:
        print(f"Authentication error: {e}")

if __name__ == "__main__":
    model, scaler = train_voice_model()
    authenticate_voice()

    os.remove("temp.wav")