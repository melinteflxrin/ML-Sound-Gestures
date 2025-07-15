import os
import time
import numpy as np
import sounddevice as sd
import librosa
import joblib
from connect_spotify import sound_gesture
from extract_features import extract_mfcc


START_TIME = time.time()
WARMUP_SECONDS = 3  # ignore predictions for the first 3 seconds 


# load trained SVM model
clf = joblib.load('data/processed/svm_model.joblib')


# audio settings
DURATION = 2  # seconds per chunk
SR = 22050    # sample rate (librosa default)


def audio_callback(indata, frames, time_info, status):
    audio = indata[:, 0]
    features = extract_mfcc_from_audio(audio, sr=SR)
    gesture = predict_gesture(features)
    if gesture:
        sound_gesture(gesture)
        time.sleep(2)  # cooldown to avoid multiple triggers


def extract_mfcc_from_audio(audio, sr=SR, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def predict_gesture(features):
    # map model output to gesture names
    label = clf.predict([features])[0]
    if label == 1:
        return "double_clap"
    elif label == 2:
        return "snap"
    else:
        return None


print("Listening for sound gestures. Press Ctrl+C to stop.")
with sd.InputStream(channels=1, samplerate=SR, callback=audio_callback, blocksize=int(SR * DURATION)):
    while True:
        time.sleep(0.1)