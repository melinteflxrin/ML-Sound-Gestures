
# Sound Gesture Control for Spotify - Machine Learning

Control Spotify to skip songs with sound gestures (like double claps) using machine learning, real-time audio detection, and the Spotify Web API.

---

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [How It Works](#how-it-works)
- [Script Usage & Order](#script-usage--order)
- [Customizing Gestures](#customizing-gestures)

---

## Description
This project uses your microphone and machine learning to recognize sound gestures (like double claps) to pause or skip songs on Spotify.

## Features
- Real-time audio gesture recognition
- Machine learning (SVM + MFCC features)
- Spotify Web API integration (skip, pause, etc.)
- Live waveform GUI with gesture feedback
- Easily extensible for new gestures

## Requirements
- Spotify Premium account
- [Spotify Developer App](https://developer.spotify.com/dashboard/applications) (for API credentials)
- Microphone

**Python packages:**
- numpy, sounddevice, librosa, joblib, spotipy, python-dotenv, tkinter, matplotlib, scikit-learn

Install all dependencies with:
```
pip install -r requirements.txt
```

## Setup
1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/ML-Sound-Gestures.git
   cd ML-Sound-Gestures
   ```
2. **Create a Spotify Developer App:**
   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications)
   - Create a new app and set the redirect URI to `http://127.0.0.1:8888/callback`
   - Copy your Client ID and Client Secret
3. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in your credentials:
     ```
     SPOTIPY_CLIENT_ID=your_client_id_here
     SPOTIPY_CLIENT_SECRET=your_client_secret_here
     SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
     ```
4. **Prepare your audio data:**
   - Place your double clap samples in `data/raw/double_clap/`
   - Place negative samples (not double claps) in `data/raw/negative/`
   - (Optional) Add more gesture folders for future expansion

## How It Works
1. **Audio Normalization:** Normalize all audio files for consistent feature extraction.
2. **Feature Extraction:** Extracts MFCC features from your audio files.
3. **Model Training:** Trains an SVM classifier to recognize gestures.
4. **Real-Time Detection:** Listens to your microphone, predicts gestures, and controls Spotify.


## Script Usage & Order

**1. Normalize your audio data:**
[`src/normalize.py`](src/normalize.py)
```
python src/normalize.py
```
- This will normalize all audio files and save them in `data/normalized/`.

**2. Extract features from your normalized audio data:**
[`src/extract_features.py`](src/extract_features.py)
```
python src/extract_features.py
```
- This will process your normalized audio files and save `X.npy` and `y.npy` in `data/processed/`.

**3. Train the machine learning model:**
[`src/train_svm.py`](src/train_svm.py)
```
python src/train_svm.py
```
- This will train the SVM and save `svm_model.joblib` in `data/processed/`.

**4. Test Spotify connection (optional):**
[`src/connect_spotify.py`](src/connect_spotify.py)
```
python src/connect_spotify.py
```
- This will skip a song to verify your Spotify API setup.

**5. Run the application:**
  - **GUI version:** [`src/listen_gui.py`](src/listen_gui.py)
    ```
    python src/listen_gui.py
    ```
    - Shows a live waveform of the audio input
  - **Console version:** [`src/listen_console.py`](src/listen_console.py)
    ```
    python src/listen_console.py
    ``` 

## Customizing & Extending Gestures
- To add new gestures, update your dataset (add new folders in `data/raw/`), normalize, extract features, and retrain the model.
- Edit `sound_gesture` in [`src/connect_spotify.py`](src/connect_spotify.py) to map new gestures to Spotify actions.
