import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import sounddevice as sd
import joblib
from collections import deque
from pathlib import Path
from connect_spotify import sound_gesture
from extract_features import DoubleClap


class SoundGestureApp:
    """GUI app"""
    
    def __init__(self, root):
        self.root = root
        self.is_running = False
        self.audio_stream = None
        self.detection_thread = None
        
        # audio detection parameters
        self.SR = 22050
        self.WINDOW_SIZE = 1.5  # seconds
        self.HOP_SIZE = 0.5     # seconds (how often to check)
        self.confidence_threshold = 0.75
        self.cooldown_time = 2.0  # seconds
        self.last_detection_time = 0
        
        # audio buffer
        self.audio_buffer = deque(maxlen=int(self.SR * self.WINDOW_SIZE))
        
        # models and feature extractor
        self.scaler = None
        self.ensemble_model = None
        self.feature_extractor = None
        self.working_device = None
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """setup user interface"""
        self.root.title("Sound Gesture Detection for Spotify")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # title
        title_label = ttk.Label(main_frame, text="Sound Gesture Detection", 
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(main_frame, text="Control Spotify with sound gestures", 
                                  font=("Arial", 11), foreground="gray")
        subtitle_label.pack(pady=(0, 20))
        
        # status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="15")
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=("Arial", 12, "bold"), foreground="blue")
        self.status_label.pack()
        
        # settings
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings", padding="15")
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.75)
        self.confidence_scale = ttk.Scale(conf_frame, from_=0.5, to=0.95, 
                                         variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        self.confidence_label = ttk.Label(conf_frame, text="0.75")
        self.confidence_label.pack(side=tk.RIGHT, padx=(5, 10))
        
        # update confidence label when scale changes
        self.confidence_var.trace('w', self.update_confidence_label)
        
        # cooldown
        cooldown_frame = ttk.Frame(settings_frame)
        cooldown_frame.pack(fill=tk.X)
        ttk.Label(cooldown_frame, text="Cooldown Time (seconds):").pack(side=tk.LEFT)
        self.cooldown_var = tk.DoubleVar(value=2.0)
        self.cooldown_scale = ttk.Scale(cooldown_frame, from_=1.0, to=5.0, 
                                       variable=self.cooldown_var, orient=tk.HORIZONTAL)
        self.cooldown_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        self.cooldown_label = ttk.Label(cooldown_frame, text="2.0s")
        self.cooldown_label.pack(side=tk.RIGHT, padx=(5, 10))
        
        # update cooldown label
        self.cooldown_var.trace('w', self.update_cooldown_label)
        
        # control buttons
        button_frame = ttk.LabelFrame(main_frame, text="Controls", padding="15")
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", 
                                      command=self.start_detection, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, pady=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state="disabled")
        self.stop_button.pack(fill=tk.X)
        
        # activity log
        log_frame = ttk.LabelFrame(main_frame, text="Detection Activity", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.activity_text = tk.Text(log_container, height=12, wrap=tk.WORD, 
                                   font=("Consolas", 9), state=tk.DISABLED,
                                   bg="#f8f8f8", fg="black")
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, 
                                command=self.activity_text.yview)
        self.activity_text.configure(yscrollcommand=scrollbar.set)
        
        self.activity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # instructions
        instructions_frame = ttk.Frame(main_frame)
        instructions_frame.pack(fill=tk.X, pady=(15, 0))
        
        instructions = ttk.Label(instructions_frame, 
                               text="Make a double clap gesture to skip Spotify tracks",
                               font=("Arial", 9), foreground="blue")
        instructions.pack()
        
        # activity log
        self.log_activity("Sound Gesture Detection Ready")
        self.log_activity("=" * 50)
        self.log_activity("Checklist:")
        self.log_activity("  - Make sure Spotify is running")
        self.log_activity("  - Ensure your microphone works") 
        self.log_activity("  - Adjust settings if needed")
        self.log_activity("")
        self.log_activity("Click 'Start Detection' to begin!")
        
    def update_confidence_label(self, *args):
        """update confidence threshold label"""
        value = self.confidence_var.get()
        self.confidence_label.config(text=f"{value:.2f}")
        if hasattr(self, 'confidence_threshold'):
            self.confidence_threshold = value
    
    def update_cooldown_label(self, *args):
        """update cooldown time label"""
        value = self.cooldown_var.get()
        self.cooldown_label.config(text=f"{value:.1f}s")
        if hasattr(self, 'cooldown_time'):
            self.cooldown_time = value
    
    def log_activity(self, message, tag=None):
        """add a message to the activity log"""
        self.activity_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.activity_text.insert(tk.END, formatted_message)
        
        # color coding for different message types
        if tag or any(keyword in message for keyword in ["ERROR", "FAIL", "error", "failed"]):
            # error messages in red
            start_idx = self.activity_text.index(f"{tk.END}-1c linestart")
            end_idx = self.activity_text.index(f"{tk.END}-1c lineend")
            self.activity_text.tag_add("error", start_idx, end_idx)
            self.activity_text.tag_config("error", foreground="red")
        elif any(keyword in message for keyword in ["Models loaded successfully", "DOUBLE CLAP DETECTED", "successful", "Using device"]):
            # success messages in green
            start_idx = self.activity_text.index(f"{tk.END}-1c linestart")
            end_idx = self.activity_text.index(f"{tk.END}-1c lineend")
            self.activity_text.tag_add("success", start_idx, end_idx)
            self.activity_text.tag_config("success", foreground="green")
        elif any(keyword in message for keyword in ["CLAP Detection", "QUIET Detection"]):
            # detection messages in blue
            start_idx = self.activity_text.index(f"{tk.END}-1c linestart")
            end_idx = self.activity_text.index(f"{tk.END}-1c lineend")
            self.activity_text.tag_add("detection", start_idx, end_idx)
            self.activity_text.tag_config("detection", foreground="blue")
        
        self.activity_text.config(state=tk.DISABLED)
        self.activity_text.see(tk.END)  # auto scroll to bottom
    
    def clear_activity_log(self):
        """clear the activity log"""
        self.activity_text.config(state=tk.NORMAL)
        self.activity_text.delete(1.0, tk.END)
        self.activity_text.config(state=tk.DISABLED)
    
    def update_status(self, status, color="black"):
        """update the status display"""
        self.status_var.set(status)
        self.status_label.config(foreground=color)
    
    def load_models(self):
        """load the trained models and feature extractor"""
        try:
            model_dir = Path(__file__).parent.parent / "data" / "processed"
            
            self.scaler = joblib.load(model_dir / "scaler.joblib")
            self.ensemble_model = joblib.load(model_dir / "model.joblib")
            self.feature_extractor = DoubleClap(sr=self.SR)
            
            self.log_activity("Models loaded successfully!")
            self.update_status("Models Loaded", "green")
            
        except FileNotFoundError as e:
            self.log_activity(f"ERROR: Could not load models - {e}")
            self.log_activity("Make sure models are trained first:")
            self.log_activity("   1. Run: python src/extract_features.py")
            self.log_activity("   2. Run: python src/train_model.py")
            self.update_status("Models Missing", "red")
            
        except Exception as e:
            self.log_activity(f"ERROR: Failed to load models - {e}")
            self.update_status("Error", "red")
    
    def find_audio_device(self):
        """find working audio input device"""
        self.log_activity("Searching for working audio device...")
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                try:
                    # test this device
                    test_stream = sd.InputStream(
                        device=i, 
                        channels=1, 
                        samplerate=self.SR, 
                        blocksize=1024
                    )
                    test_stream.start()
                    test_stream.stop()
                    test_stream.close()
                    
                    self.working_device = i
                    self.log_activity(f"Using device {i}: {device['name']}")
                    return True
                    
                except Exception as e:
                    self.log_activity(f"Device {i} failed: {device['name']}")
                    continue
        
        self.log_activity("ERROR: No working audio device found!")
        self.log_activity("Try closing other audio applications")
        return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """audio callback for real-time processing"""
        if status:
            self.log_activity(f"Audio status: {status}")
        
        audio_chunk = indata[:, 0]  # mono audio
        self.audio_buffer.extend(audio_chunk)
    
    def detect_double_clap(self, audio_window):
        """detect double clap in audio window"""
        try:
            # extract features
            features = self.feature_extractor.extract_all_features(audio_window)
            
            # scale features and predict
            features_scaled = self.scaler.transform([features])
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            prediction = self.ensemble_model.predict(features_scaled)[0]
            
            # get confidence for positive class (double clap)
            confidence = probabilities[1] if len(probabilities) > 1 else 0.5
            
            return confidence, prediction == 1
            
        except Exception as e:
            self.log_activity(f"Detection error: {e}")
            return None
    
    def detection_loop(self):
        """detection loop running in background thread"""
        self.log_activity("Detection loop started")
        
        try:
            while self.is_running:
                # wait for hop time
                time.sleep(self.HOP_SIZE)
                
                # update settings from GUI
                self.confidence_threshold = self.confidence_var.get()
                self.cooldown_time = self.cooldown_var.get()
                
                # check if we have enough audio
                if len(self.audio_buffer) >= int(self.SR * self.WINDOW_SIZE):
                    # skip if in cooldown
                    if time.time() - self.last_detection_time < self.cooldown_time:
                        continue
                    
                    # get audio window and detect
                    audio_window = np.array(list(self.audio_buffer))
                    result = self.detect_double_clap(audio_window)
                    
                    if result:
                        confidence, is_double_clap = result
                        
                        # log detection attempt
                        status = "CLAP" if is_double_clap else "QUIET"
                        self.log_activity(f"{status} Detection: {confidence:.3f} confidence")
                        
                        # trigger action if above threshold
                        if is_double_clap and confidence >= self.confidence_threshold:
                            self.last_detection_time = time.time()
                            self.log_activity(f"DOUBLE CLAP DETECTED! Skipping track...")
                            
                            # call Spotify action
                            try:
                                sound_gesture("double_clap")
                                self.log_activity("Spotify action successful")
                            except Exception as e:
                                self.log_activity(f"Spotify action failed: {e}")
                                self.log_activity("Make sure Spotify is running and connected")
                
        except Exception as e:
            self.log_activity(f"Detection loop error: {e}")
        finally:
            self.log_activity("Detection loop ended")
    
    def start_detection(self):
        """start the sound gesture detection"""
        if self.is_running:
            return
        
        # check if models are loaded
        if not self.scaler or not self.ensemble_model:
            messagebox.showerror("Error", "Models not loaded! Please check that the models are trained.")
            return
        
        try:
            self.clear_activity_log()
            self.log_activity("Starting sound gesture detection...")
            
            # find working audio device
            if not self.find_audio_device():
                self.update_status("Device Error", "red")
                return
            
            # start audio stream
            self.log_activity("Starting audio stream...")
            self.audio_stream = sd.InputStream(
                device=self.working_device,
                channels=1,
                samplerate=self.SR,
                callback=self.audio_callback,
                blocksize=int(self.SR * 0.1)  # 100ms blocks
            )
            
            self.audio_stream.start()
            self.is_running = True
            
            # start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            # update UI
            self.update_status("Listening...", "green")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            self.log_activity("Detection started successfully!")
            self.log_activity(f"Confidence threshold: {self.confidence_threshold:.2f}")
            self.log_activity(f"Cooldown time: {self.cooldown_time:.1f}s")
            self.log_activity("=" * 50)
            self.log_activity("Listening for double claps...")
            
        except Exception as e:
            self.log_activity(f"ERROR: Failed to start detection - {e}")
            self.update_status("Error", "red")
            messagebox.showerror("Error", f"Failed to start detection:\n{e}")
    
    def stop_detection(self):
        """stop the sound gesture detection"""
        if not self.is_running:
            return
        
        try:
            self.log_activity("Stopping detection...")
            self.is_running = False
            
            # stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            
            # update UI
            self.update_status("Stopped", "red")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            self.log_activity("Detection stopped successfully")
            self.log_activity("Ready to start again!")
            
        except Exception as e:
            self.log_activity(f"ERROR: Failed to stop detection - {e}")
            messagebox.showerror("Error", f"Error stopping detection:\n{e}")
    
    def on_closing(self):
        """handle window closing"""
        if self.is_running:
            result = messagebox.askyesno("Confirm Exit", 
                                       "Sound gesture detection is running.\nDo you want to stop it and exit?")
            if result:
                self.stop_detection()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """main entry point"""
    root = tk.Tk()
    
    # try to set a nice theme
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clam' in available_themes:
            style.theme_use('clam')
    except:
        pass
    
    # create the app
    app = SoundGestureApp(root)
    
    # handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # center the window
    root.eval('tk::PlaceWindow . center')
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
