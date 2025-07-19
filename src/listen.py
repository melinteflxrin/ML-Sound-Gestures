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
        # Audio settings
        self.SR = 22050
        self.WINDOW_SIZE = 1.5  # seconds
        self.HOP_SIZE = 0.5     # seconds (how often to check)
        
        # Load models
        self.load_models(model_dir)
        
        # Feature extractor
        self.feature_extractor = DoubleClap(sr=self.SR)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.SR * self.WINDOW_SIZE))
        
        # Detection parameters
        self.confidence_threshold = 0.75
        self.cooldown_time = 2.0  # seconds
        self.last_detection_time = 0
        
    def load_models(self, model_dir):
        """Load the trained ensemble model and scaler"""
        try:
            self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
            self.ensemble_model = joblib.load(f'{model_dir}/model.joblib')
            print("✅ Models loaded successfully!")
        except FileNotFoundError:
            raise FileNotFoundError("❌ No trained models found! Please train a model first.")
    
    def detect_double_clap(self, audio_window):
        """Detect double clap in audio window"""
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(audio_window)
            
            # Scale features and predict
            features_scaled = self.scaler.transform([features])
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            prediction = self.ensemble_model.predict(features_scaled)[0]
            
            # Get confidence for positive class (double clap)
            confidence = probabilities[1] if len(probabilities) > 1 else 0.5
            
            return confidence, prediction == 1
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for real-time processing"""
        audio_chunk = indata[:, 0]  # Mono audio
        self.audio_buffer.extend(audio_chunk)
    
    def run(self):
        """Start listening for double claps"""
        print("🎵 Simple Double Clap Detection")
        print("=" * 40)
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"⏱️  Cooldown time: {self.cooldown_time}s")
        print("🎤 Starting audio capture...")
        print("👏 Make a double clap to test detection!")
        print("Press Ctrl+C to stop")
        print("=" * 40)
        
        # Start audio stream
        with sd.InputStream(
            channels=1,
            samplerate=self.SR,
            callback=self.audio_callback,
            blocksize=int(self.SR * 0.1)  # 100ms blocks
        ):
            try:
                while True:
                    # Wait for hop time
                    time.sleep(self.HOP_SIZE)
                    
                    # Check if we have enough audio
                    if len(self.audio_buffer) >= int(self.SR * self.WINDOW_SIZE):
                        # Skip if in cooldown
                        if time.time() - self.last_detection_time < self.cooldown_time:
                            continue
                        
                        # Get audio window and detect
                        audio_window = np.array(list(self.audio_buffer))
                        result = self.detect_double_clap(audio_window)
                        
                        if result:
                            confidence, is_double_clap = result
                            
                            # Print detection attempt (for debugging)
                            status = "👏" if is_double_clap else "🔇"
                            print(f"{status} Detection: {confidence:.3f} confidence", end="")
                            
                            # Trigger action if above threshold
                            if is_double_clap and confidence >= self.confidence_threshold:
                                self.last_detection_time = time.time()
                                print(" ✅ DOUBLE CLAP DETECTED! Skipping track...")
                                
                                # Call Spotify action
                                try:
                                    sound_gesture("double_clap")
                                except Exception as e:
                                    print(f"❌ Spotify action failed: {e}")
                            else:
                                print()  # New line for low confidence detections
                        
            except KeyboardInterrupt:
                print("\n🛑 Stopping detection...")
                print("👋 Goodbye!")


if __name__ == "__main__":
    try:
        detector = SimpleDoubleClap()
        detector.run()
    except Exception as e:
        print(f"❌ Error starting detector: {e}")
        print("💡 Please ensure you have trained models available!")
        print("📋 Run: python src/extract_features.py")
        print("📋 Then: python src/train_model.py")
