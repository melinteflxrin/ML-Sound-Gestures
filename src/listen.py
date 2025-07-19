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
from extract_features import DoubleClap
from collections import deque
import threading
from queue import Queue


class RealTimeDoubleClap:
    def __init__(self, model_dir='data/processed'):
        # Audio settings
        self.SR = 22050
        self.WINDOW_SIZE = 1.5  # seconds
        self.HOP_SIZE = 0.1     # seconds (how often to check)
        self.DISPLAY_CHUNK = 0.05  # seconds for smooth waveform
        
        # Load models
        self.load_models(model_dir)
        
        # Feature extractor
        self.feature_extractor = DoubleClap(sr=self.SR)
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(self.SR * self.WINDOW_SIZE))
        self.display_buffer = np.zeros(int(self.SR * 2.0))  # 2 seconds for display
        
        # Detection parameters
        self.confidence_threshold = 0.75
        self.cooldown_time = 2.0  # seconds
        self.last_detection_time = 0
        
        # State management
        self.detection_queue = Queue()
        self.is_listening = False
        self.warmup_time = 3.0  # seconds
        self.start_time = time.time()
        
        # Performance monitoring
        self.detection_history = deque(maxlen=100)
        self.false_positive_filter = False
        
        # Setup GUI
        self.setup_gui()
        
    def load_models(self, model_dir):
        """Load the trained ensemble model and scaler"""
        try:
            self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
            self.ensemble_model = joblib.load(f'{model_dir}/model.joblib')
            print("Advanced models loaded successfully!")
        except FileNotFoundError:
            raise FileNotFoundError("No trained models found! Please train a model first.")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Advanced Double Clap Detection")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status section
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Initializing...", font=("Arial", 16))
        self.status_label.pack(side=tk.LEFT)
        
        self.confidence_label = ttk.Label(status_frame, text="Confidence: -", font=("Arial", 12))
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings", padding=5)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5)
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        confidence_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, 
                                   variable=self.confidence_var, length=200)
        confidence_scale.grid(row=0, column=1, padx=5)
        
        confidence_label = ttk.Label(settings_frame, textvariable=self.confidence_var)
        confidence_label.grid(row=0, column=2, padx=5)
        
        # False positive filter
        self.fp_filter_var = tk.BooleanVar(value=self.false_positive_filter)
        fp_check = ttk.Checkbutton(settings_frame, text="Enhanced False Positive Filter", 
                                  variable=self.fp_filter_var)
        fp_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Waveform display
        self.setup_waveform_display(main_frame)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(main_frame, text="Detection Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="No detections yet")
        self.stats_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Listening", 
                                      command=self.toggle_listening)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Test Detection", 
                  command=self.test_detection).pack(side=tk.LEFT, padx=5)
        
        # Setup close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_waveform_display(self, parent):
        """Setup matplotlib waveform display"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 4))
        
        # Raw waveform
        self.waveform_line, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, 2.0)
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title('Real-time Audio')
        self.ax1.grid(True, alpha=0.3)
        
        # Detection confidence over time
        self.confidence_line, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.ax2.axhline(y=self.confidence_threshold, color='g', linestyle='--', alpha=0.7)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlim(0, 50)  # Last 50 detections
        self.ax2.set_ylabel('Confidence')
        self.ax2.set_xlabel('Detection #')
        self.ax2.set_title('Detection Confidence History')
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for real-time processing"""
        if not self.is_listening:
            return
            
        audio_chunk = indata[:, 0]  # Mono audio
        
        # Add to buffers
        self.audio_buffer.extend(audio_chunk)
        
        # Update display buffer
        self.display_buffer = np.roll(self.display_buffer, -len(audio_chunk))
        self.display_buffer[-len(audio_chunk):] = audio_chunk
        
        # Update waveform display
        self.update_waveform()
        
        # Check if we have enough audio for detection
        if len(self.audio_buffer) >= int(self.SR * self.WINDOW_SIZE):
            # Add to detection queue (non-blocking)
            if self.detection_queue.qsize() < 3:  # Prevent queue overflow
                audio_window = np.array(list(self.audio_buffer))
                self.detection_queue.put(audio_window)
    
    def update_waveform(self):
        """Update the waveform display"""
        time_axis = np.linspace(0, len(self.display_buffer) / self.SR, len(self.display_buffer))
        self.waveform_line.set_data(time_axis, self.display_buffer)
        
        # Update confidence history
        if len(self.detection_history) > 0:
            confidences = [d['confidence'] for d in self.detection_history]
            detection_nums = list(range(len(confidences)))
            self.confidence_line.set_data(detection_nums, confidences)
            
            # Update threshold line
            self.ax2.lines[1].set_ydata([self.confidence_var.get()] * 2)
        
        self.canvas.draw_idle()
    
    def detection_worker(self):
        """Worker thread for audio detection"""
        while self.is_listening:
            try:
                # Get audio from queue (with timeout)
                audio_window = self.detection_queue.get(timeout=0.1)
                
                # Skip if in warmup period
                if time.time() - self.start_time < self.warmup_time:
                    continue
                
                # Skip if in cooldown
                if time.time() - self.last_detection_time < self.cooldown_time:
                    continue
                
                # Extract features and predict
                prediction_result = self.detect_double_clap(audio_window)
                
                if prediction_result:
                    confidence, is_double_clap = prediction_result
                    
                    # Update confidence threshold from GUI
                    current_threshold = self.confidence_var.get()
                    
                    # Record detection attempt
                    detection_record = {
                        'timestamp': time.time(),
                        'confidence': confidence,
                        'predicted_class': is_double_clap,
                        'triggered': False
                    }
                    
                    # Apply enhanced filtering if enabled
                    should_trigger = self.should_trigger_action(
                        confidence, is_double_clap, current_threshold
                    )
                    
                    if should_trigger:
                        detection_record['triggered'] = True
                        self.trigger_detection(confidence)
                    
                    self.detection_history.append(detection_record)
                    self.update_gui_status(confidence, should_trigger)
                    
            except:
                continue  # Queue timeout or other error
    
    def detect_double_clap(self, audio_window):
        """
        Detect double clap in audio window
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(audio_window)
            
            # Handle case where we don't have the advanced scaler
            if self.scaler is not None:
                features_scaled = self.scaler.transform([features])
                probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
                prediction = self.ensemble_model.predict(features_scaled)[0]
            else:
                # Fallback to basic model
                probabilities = [0.5, 0.5]  # Dummy probabilities
                prediction = self.ensemble_model.predict([features])[0]
            
            # Get confidence for positive class
            confidence = probabilities[1] if len(probabilities) > 1 else 0.5
            
            return confidence, prediction == 1
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def should_trigger_action(self, confidence, is_double_clap, threshold):
        """
        Enhanced decision making for triggering actions
        """
        if not is_double_clap:
            return False
        
        if confidence < threshold:
            return False
        
        # Enhanced false positive filtering
        if self.fp_filter_var.get():
            # Check recent detection history
            recent_detections = [d for d in self.detection_history 
                               if time.time() - d['timestamp'] < 5.0]
            
            # Too many recent detections might indicate noise
            if len([d for d in recent_detections if d['triggered']]) > 3:
                return False
            
            # Check confidence trend
            if len(recent_detections) > 3:
                recent_confidences = [d['confidence'] for d in recent_detections[-3:]]
                if np.std(recent_confidences) > 0.2:  # High variance = unstable
                    return False
        
        return True
    
    def trigger_detection(self, confidence):
        """
        Trigger the Spotify action
        """
        self.last_detection_time = time.time()
        
        # Call Spotify action
        try:
            sound_gesture("double_clap")
            print(f"Double clap detected! Confidence: {confidence:.3f}")
        except Exception as e:
            print(f"Spotify action failed: {e}")
    
    def update_gui_status(self, confidence, triggered):
        """
        Update GUI with detection results
        """
        # Update status
        if triggered:
            self.status_label.config(text="🎵 DOUBLE CLAP DETECTED!", foreground="green")
            self.root.after(2000, lambda: self.status_label.config(
                text="Listening...", foreground="black"))
        
        # Update confidence
        self.confidence_label.config(text=f"Confidence: {confidence:.3f}")
        
        # Update statistics
        total_detections = len(self.detection_history)
        triggered_detections = len([d for d in self.detection_history if d['triggered']])
        
        if total_detections > 0:
            avg_confidence = np.mean([d['confidence'] for d in self.detection_history])
            self.stats_label.config(
                text=f"Total: {total_detections} | Triggered: {triggered_detections} | "
                     f"Avg Confidence: {avg_confidence:.3f}"
            )
    
    def toggle_listening(self):
        """
        Start/stop listening
        """
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """
        Start audio capture and detection
        """
        self.is_listening = True
        self.start_time = time.time()
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.SR,
            callback=self.audio_callback,
            blocksize=int(self.SR * self.DISPLAY_CHUNK),
            device=None  # Use default microphone
        )
        
        # Start detection worker thread
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        self.stream.start()
        
        self.start_button.config(text="Stop Listening")
        self.status_label.config(text=f"Warming up... ({self.warmup_time}s)", foreground="orange")
        
        # Set status to listening after warmup
        self.root.after(int(self.warmup_time * 1000), 
                       lambda: self.status_label.config(text="Listening...", foreground="blue"))
        
        print("Started listening for double claps...")
    
    def stop_listening(self):
        """
        Stop audio capture and detection
        """
        self.is_listening = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.start_button.config(text="Start Listening")
        self.status_label.config(text="Stopped", foreground="red")
        
        print("Stopped listening")
    
    def test_detection(self):
        """
        Test detection with current audio buffer
        """
        if len(self.audio_buffer) >= int(self.SR * self.WINDOW_SIZE):
            audio_window = np.array(list(self.audio_buffer))
            result = self.detect_double_clap(audio_window)
            
            if result:
                confidence, is_double_clap = result
                status = "DOUBLE CLAP" if is_double_clap else "NOT DOUBLE CLAP"
                print(f"Test result: {status}, Confidence: {confidence:.3f}")
                
                # Show in GUI temporarily
                original_text = self.status_label.cget("text")
                self.status_label.config(text=f"Test: {status} ({confidence:.3f})")
                self.root.after(3000, lambda: self.status_label.config(text=original_text))
    
    def on_closing(self):
        """
        Handle window closing
        """
        self.stop_listening()
        self.root.destroy()
    
    def run(self):
        """
        Start the GUI main loop
        """
        print("Advanced Double Clap Detection GUI ready!")
        print("Click 'Start Listening' to begin detection.")
        self.root.mainloop()


if __name__ == "__main__":
    try:
        detector = RealTimeDoubleClap()
        detector.run()
    except Exception as e:
        print(f"Error starting detector: {e}")
        print("Please ensure you have trained models available!")
