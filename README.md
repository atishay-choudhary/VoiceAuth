# VoiceAuth: Real-Time Voice Authentication with SVM

A Python prototype that authenticates a speaker from a live microphone recording. It extracts acoustic features from a 3-second clip, scales them, and classifies them with a tuned Support Vector Machine.


# How it works
microphone (PyAudio, 3 s, 44.1 kHz)
   -> low-pass filter (3 kHz Butterworth)
   -> silence trimming (voice activity)
   -> feature extraction (179 dimensions)
   -> StandardScaler
   -> SVM (GridSearchCV over C and kernel)
   -> accept only if predicted "user" with confidence >= 90%


# Features (179-dimensional vector, time-averaged)
|Feature | Dims |
|---|---|
| MFCC | 13 |
| Delta MFCC | 13 |
| Chroma | 12 |
| Spectral contrast | 7 |
| Tonnetz (harmonic component) | 6 |
| Mel spectrogram | 128 |

### Model

- `sklearn.svm.SVC` with probability estimates
- Grid search over `C` in {0.1, 1, 10} and kernel in {linear, rbf}, 3-fold cross-validation
- `StandardScaler` fitted on training data only
- A GPU variant using RAPIDS cuML lives in `test.py` and `test.ipynb`

## Repository layout

| Path | Purpose |
|---|---|
| `AuthDs.py` | Main script: loads recordings, trains, then authenticates live |
| `auth.py`, `authos.py` | Earlier and simpler variants (MFCC only) |
| `greet.py`, `greet2.py` | Authentication with a spoken greeting on success |
| `test.py`, `test.ipynb` | GPU (cuML) experiments |
| `pkl.py`, `check_pickle_integrity.py` | Model save/load and integrity checks |
| `Audios/user_0/` | Enrollment recordings (WAV) |
| `Docs/` | Design documents, including embedded (Raspberry Pi) and AirDraw integration concepts |

## Setup

```bash
python -m venv .venv
pip install numpy scipy scikit-learn joblib librosa pyaudio
```

On Windows, PyAudio may need a prebuilt wheel. Set `DATASET_PATH` at the top of `AuthDs.py` to your local `Audios/user_0` folder (the default is a hardcoded Windows path).

## Usage

```bash
python AuthDs.py     # trains from Audios/user_0, then asks for a live 3-second recording
```

Output is `Access Granted` or `Access Denied` with the model's confidence.

## Limitations

- **Impostor class is synthetic.** Training uses random Gaussian noise as the "not the user" class, not recordings of other speakers. The reported accuracy therefore shows that the model separates a human voice from noise. It is **not** a measure of speaker-verification performance.
- **Single enrolled speaker**, about 26 short recordings.
- **No standard biometric metrics** (FAR, FRR, EER) yet.
- **No liveness or replay-attack protection**, so a recording of the user's voice could pass.
- Model files are stored as pickles, which should only ever be loaded from trusted sources.

## Roadmap

- [ ] Train and evaluate against real impostor speakers (for example, a LibriSpeech subset) with a speaker-disjoint split
- [ ] Report FAR, FRR, and EER, and choose the decision threshold from the EER point
- [ ] Add speaker embeddings (for example, x-vectors) and compare against the MFCC + SVM baseline
- [ ] Add replay and liveness checks
- [ ] Port inference to a Raspberry Pi (design notes are in `Docs/`)

## Author

Atishay Choudhary
