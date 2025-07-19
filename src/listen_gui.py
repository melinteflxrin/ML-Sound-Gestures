import os
import time
import numpy as np
import sounddevice as sd

# list all available audio devices
print("Available audio devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"Input {i}: {dev['name']}")

# set to index of microphone
MIC_DEVICE_INDEX = 1  # none for device default mic
import librosa
import joblib
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from connect_spotify import sound_gesture
from extract_features import extract_mfcc_from_audio
from scipy.signal import find_peaks


# load trained SVM model
clf = joblib.load('data/processed/svm_model.joblib')


# audio settings
SR = 22050    # sample rate
DURATION = 1.5  # seconds for gesture detection window
DISPLAY_CHUNK = 0.1  # seconds for waveform update


# GUI setup
root = tk.Tk()
root.title("Sound Gesture Spotify Controller")
root.geometry("600x400")

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

status_label = ttk.Label(main_frame, text="Listening...", font=("Arial", 18))
status_label.pack(pady=10)


# matplotlib for waveform
fig, ax = plt.subplots(figsize=(6, 2))
canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

waveform_line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, SR * DURATION)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()


# state
last_gesture = None
last_trigger_time = 0
COOLDOWN = 2  # seconds
COOLDOWN_ACTIVE = False
COOLDOWN_END = 0


# for warmup
START_TIME = time.time()
WARMUP_SECONDS = 2


# buffer for waveform (for display)
audio_buffer = np.zeros(int(SR * DURATION))

# rolling buffer for gesture detection
gesture_buffer = np.zeros(int(SR * DURATION))
# counter to track when to run gesture detection
gesture_samples = 0


def update_waveform(audio):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -len(audio))
    audio_buffer[-len(audio):] = audio
    waveform_line.set_data(np.arange(len(audio_buffer)), audio_buffer)
    canvas.draw()


def predict_gesture(features):
    # Get prediction probabilities for confidence checking
    proba = clf.predict_proba([features])[0]
    label = clf.predict([features])[0]
    confidence = np.max(proba)
    
    # Only return gesture if confidence is high enough
    if confidence < 0.7:  # Require 70% confidence
        return None
        
    if label == 1:
        return "double_clap"
    elif label == 2:
        return "snap"
    else:
        return None


def audio_callback(indata, frames, time_info, status):
    global last_gesture, last_trigger_time
    audio = indata[:, 0]
    # update waveform display
    norm_audio = audio.copy()
    max_val = np.max(np.abs(norm_audio))
    if max_val > 0:
        norm_audio = norm_audio / max_val
    update_waveform(norm_audio)

    # update rolling buffer for gesture detection
    global gesture_buffer, gesture_samples
    gesture_buffer = np.roll(gesture_buffer, -len(audio))
    gesture_buffer[-len(audio):] = audio
    gesture_samples += len(audio)

    # only run gesture detection if buffer is full, 
    # not in warmup, and not in cooldown
    global COOLDOWN_ACTIVE, COOLDOWN_END
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
    
    # step 1: ML to classify the audio first
    norm_gesture = gesture_buffer.copy()
    max_val = np.max(np.abs(norm_gesture))
    if max_val > 0:
        norm_gesture = norm_gesture / max_val
    features = extract_mfcc_from_audio(norm_gesture, SR, n_mfcc=13)
    # original max amplitude as a feature
    orig_max_amp = np.max(np.abs(gesture_buffer))
    features = np.concatenate([features, [orig_max_amp]])
    gesture = predict_gesture(features)
    
    # step 2: verify with appropriate checks based on gesture type
    if gesture == "double_clap":
        # For double claps, verify with peak detection
        peaks, properties = find_peaks(np.abs(gesture_buffer), height=0.2, distance=int(0.08 * SR))
        if len(peaks) < 2:
            return  # ML said double_clap but no double peaks found
        
        # peaks should be spaced
        peak_intervals = np.diff(peaks)
        if len(peak_intervals) > 0:
            min_interval = int(0.05 * SR)  # minimum 50ms between claps
            max_interval = int(0.5 * SR)   # maximum 500ms between claps
            if not all(min_interval <= interval <= max_interval for interval in peak_intervals):
                return  # peaks not properly spaced for double clap
    
    elif gesture == "snap":
        # For finger snaps, verify there's at least one clear peak
        peaks, properties = find_peaks(np.abs(gesture_buffer), height=0.15, distance=int(0.05 * SR))
        if len(peaks) < 1:
            return  # ML said snap but no clear peak found
    
    if gesture:
        last_gesture = gesture
        last_trigger_time = now
        sound_gesture(gesture)
        status_label.config(text=f"Detected: {gesture}", background="lightgreen")
        root.after(1500, lambda: status_label.config(text="Listening...", background="SystemButtonFace"))
        # start cooldown after any gesture
        COOLDOWN_ACTIVE = True
        COOLDOWN_END = now + COOLDOWN
    # reset cooldown if time passed
    if COOLDOWN_ACTIVE and now >= COOLDOWN_END:
        COOLDOWN_ACTIVE = False


def on_closing():
    stream.stop()
    stream.close()
    root.destroy()
    os._exit(0)  # force exit to ensure all threads/processes are killed


root.protocol("WM_DELETE_WINDOW", on_closing)


# start audio stream in a thread
stream = sd.InputStream(
    channels=1,
    samplerate=SR,
    callback=audio_callback,
    blocksize=int(SR * DISPLAY_CHUNK),
    device=MIC_DEVICE_INDEX
)
stream.start()

root.mainloop()
