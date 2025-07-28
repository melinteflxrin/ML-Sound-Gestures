import os
import time
import numpy as np
import sounddevice as sd
import joblib
from connect_spotify import sound_gesture
from extract_features import DoubleClap
from collections import deque


class SimpleDoubleClap:
    def __init__(self, model_dir='data/processed'):
        # audio settings
        self.SR = 22050
        self.WINDOW_SIZE = 1.5  # seconds
        self.HOP_SIZE = 0.5     # seconds (how often to check)
        
        # load models
        self.load_models(model_dir)
        
        # feature extractor
        self.feature_extractor = DoubleClap(sr=self.SR)
        
        # audio buffer
        self.audio_buffer = deque(maxlen=int(self.SR * self.WINDOW_SIZE))
        
        # detection parameters
        self.confidence_threshold = 0.75
        self.cooldown_time = 2.0  # seconds
        self.last_detection_time = 0
        
    def load_models(self, model_dir):
        """load the trained ensemble model and scaler"""
        try:
            self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
            self.ensemble_model = joblib.load(f'{model_dir}/model.joblib')
            print("Models loaded successfully!")
        except FileNotFoundError:
            raise FileNotFoundError("No trained models found! Please train a model first.")
    
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
            print(f"Detection error: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """audio callback for real-time processing"""
        audio_chunk = indata[:, 0]  # mono audio
        self.audio_buffer.extend(audio_chunk)
    
    def run(self):
        """Start listening for double claps"""
        print("Simple Double Clap Detection")
        print("=" * 40)
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Cooldown time: {self.cooldown_time}s")
        print("Starting audio capture...")
        print("Make a double clap to test detection!")
        print("Press Ctrl+C to stop")
        print("=" * 40)
        
        # find a working audio device
        devices = sd.query_devices()
        working_device = None
        
        print("Searching for working audio device...")
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
                    working_device = i
                    print(f"✓ Using device {i}: {device['name']}")
                    break
                except Exception as e:
                    print(f"✗ Device {i} failed: {device['name']}")
                    continue
        
        if working_device is None:
            raise RuntimeError("No working audio input device found! Try closing other audio applications.")
        
        # start audio stream with working device
        with sd.InputStream(
            device=working_device,
            channels=1,
            samplerate=self.SR,
            callback=self.audio_callback,
            blocksize=int(self.SR * 0.1)  # 100ms blocks
        ):
            try:
                while True:
                    # wait for hop time
                    time.sleep(self.HOP_SIZE)
                    
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
                            
                            # print detection attempt (for debugging)
                            status = "CLAP" if is_double_clap else "QUIET"
                            print(f"{status} Detection: {confidence:.3f} confidence", end="")
                            
                            # trigger action if above threshold
                            if is_double_clap and confidence >= self.confidence_threshold:
                                self.last_detection_time = time.time()
                                print(" DOUBLE CLAP DETECTED! Skipping track...")
                                
                                # call Spotify action
                                try:
                                    sound_gesture("double_clap")
                                except Exception as e:
                                    print(f"Spotify action failed: {e}")
                            else:
                                print()  # new line for low confidence detections
                        
            except KeyboardInterrupt:
                print("\nStopping detection...")


if __name__ == "__main__":
    try:
        detector = SimpleDoubleClap()
        detector.run()
    except Exception as e:
        print(f"Error starting detector: {e}")
        print("Please ensure you have trained models available!")
        print("Run: python src/extract_features.py")
        print("Then: python src/train_model.py")
