
import os
import time
import numpy as np
import sounddevice as sd
import librosa
import joblib
from connect_spotify import sound_gesture
from extract_features import extract_mfcc_from_audio
from scipy.signal import find_peaks


# load trained SVM model
clf = joblib.load('data/processed/svm_model.joblib')


# audio settings
SR = 22050    # sample rate
DURATION = 1.5  # seconds for gesture detection window
COOLDOWN = 2  # seconds
WARMUP_SECONDS = 2


# rolling buffer for gesture detection
gesture_buffer = np.zeros(int(SR * DURATION))
gesture_samples = 0
COOLDOWN_ACTIVE = False
COOLDOWN_END = 0
START_TIME = time.time()


def predict_gesture(features):
    label = clf.predict([features])[0]
    if label == 1:
        return "double_clap"
    elif label == 2:
        return "snap"
    else:
        return None


def audio_callback(indata, frames, time_info, status):
    global gesture_buffer, gesture_samples, COOLDOWN_ACTIVE, COOLDOWN_END
    audio = indata[:, 0]
    gesture_buffer = np.roll(gesture_buffer, -len(audio))
    gesture_buffer[-len(audio):] = audio
    gesture_samples += len(audio)

    now = time.time()
    if now - START_TIME < WARMUP_SECONDS:
        return
    if np.count_nonzero(gesture_buffer) == 0:
        return
    if gesture_samples < int(SR * DURATION):
        return
    if COOLDOWN_ACTIVE and now < COOLDOWN_END:
        return
    gesture_samples = 0  # reset counter after detection

    # gesture detection on rolling buffer
    peaks, _ = find_peaks(np.abs(gesture_buffer), height=0.2, distance=int(0.08 * SR))
    if len(peaks) < 2:
        return  # not enough claps detected

    # normalize for feature extraction
    norm_gesture = gesture_buffer.copy()
    max_val = np.max(np.abs(norm_gesture))
    if max_val > 0:
        norm_gesture = norm_gesture / max_val
    features = extract_mfcc_from_audio(norm_gesture, SR, n_mfcc=13)
    # original max amplitude as a feature
    orig_max_amp = np.max(np.abs(gesture_buffer))
    features = np.concatenate([features, [orig_max_amp]])
    gesture = predict_gesture(features)
    if gesture:
        print(f"Detected: {gesture}")
        sound_gesture(gesture)
        if gesture == "double_clap":
            COOLDOWN_ACTIVE = True
            COOLDOWN_END = now + COOLDOWN

    # reset cooldown if time passed
    if COOLDOWN_ACTIVE and now >= COOLDOWN_END:
        COOLDOWN_ACTIVE = False


print("Listening for sound gestures. Press Ctrl+C to stop.")
with sd.InputStream(channels=1, samplerate=SR, callback=audio_callback, blocksize=int(SR * 0.1)):
    while True:
        time.sleep(0.1)