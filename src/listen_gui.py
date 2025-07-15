import os
import time
import numpy as np
import sounddevice as sd
import librosa
import joblib
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from connect_spotify import sound_gesture
from extract_features import extract_mfcc


# load trained SVM model
clf = joblib.load('data/processed/svm_model.joblib')


# audio settings
DURATION = 2  # seconds per chunk
SR = 22050    # sample rate


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


# for warmup
START_TIME = time.time()
WARMUP_SECONDS = 2


# buffer for waveform
audio_buffer = np.zeros(SR * DURATION)

def update_waveform(audio):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -len(audio))
    audio_buffer[-len(audio):] = audio
    waveform_line.set_data(np.arange(len(audio_buffer)), audio_buffer)
    canvas.draw()


def extract_mfcc_from_audio(audio, sr=SR, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def predict_gesture(features):
    label = clf.predict([features])[0]
    if label == 1:
        return "double_clap"
    elif label == 2:
        return "snap"
    else:
        return None


def audio_callback(indata, frames, time_info, status):
    global last_gesture, last_trigger_time
    audio = indata[:, 0]
    update_waveform(audio)
    if time.time() - START_TIME < WARMUP_SECONDS:
        return
    features = extract_mfcc_from_audio(audio, sr=SR)
    gesture = predict_gesture(features)
    now = time.time()
    if gesture and (now - last_trigger_time > COOLDOWN):
        last_gesture = gesture
        last_trigger_time = now
        sound_gesture(gesture)
        status_label.config(text=f"Detected: {gesture}", background="lightgreen")
        root.after(1500, lambda: status_label.config(text="Listening...", background="SystemButtonFace"))


def on_closing():
    stream.stop()
    stream.close()
    root.destroy()
    os._exit(0)  # force exit to ensure all threads/processes are killed


root.protocol("WM_DELETE_WINDOW", on_closing)


# start audio stream in a thread
stream = sd.InputStream(channels=1, samplerate=SR, callback=audio_callback, blocksize=int(SR * DURATION))
stream.start()

root.mainloop()
