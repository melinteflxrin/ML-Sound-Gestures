import librosa
import numpy as np
import os
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


class DoubleClap:
    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        
    def preprocess_audio(self, y, target_length=1.5):
        """
        preprocess audio with noise reduction and normalization
        """
        # Remove DC offset
        y = y - np.mean(y)
        
        # Apply high-pass filter to remove low-frequency noise
        b, a = butter(3, 100, btype='high', fs=self.sr)
        y = filtfilt(b, a, y)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Trim or pad to target length
        target_samples = int(target_length * self.sr)
        if len(y) > target_samples:
            # Find the most energetic part
            energy = np.convolve(y**2, np.ones(target_samples), mode='valid')
            start_idx = np.argmax(energy)
            y = y[start_idx:start_idx + target_samples]
        else:
            # Pad with zeros
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
            
        return y
    
    def extract_temporal_features(self, y):
        """
        extract temporal features for double clap detection
        """
        features = []
        
        # 1. Peak-based features
        peaks, properties = find_peaks(np.abs(y), height=0.1, distance=int(0.05 * self.sr))
        
        # Number of significant peaks
        features.append(len(peaks))
        
        # Peak spacing analysis (key for double claps)
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / self.sr  # Convert to seconds
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.min(intervals),
                np.max(intervals),
                len(intervals[intervals < 0.5])  # Peaks within 500ms (double clap range)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 2. Energy envelope features
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.01 * self.sr)    # 10ms hop
        
        # Calculate RMS energy envelope
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Energy envelope characteristics
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms),
            skew(rms),
            kurtosis(rms)
        ])
        
        # 3. Zero crossing rate patterns
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.max(zcr)
        ])
        
        # 4. Spectral features for transient detection
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        
        features.extend([
            np.mean(spec_centroid),
            np.std(spec_centroid),
            np.mean(spec_rolloff),
            np.std(spec_rolloff),
            np.mean(spec_bandwidth),
            np.std(spec_bandwidth)
        ])
        
        return np.array(features)
    
    def extract_spectral_features(self, y):
        """
        Extract spectral features optimized for clap sounds
        """
        # 1. MFCC features (fewer coefficients, focused on transients)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=8)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Statistical features for MFCC coefficients
        mfcc_stats = []
        for coeff in [mfcc, mfcc_delta, mfcc_delta2]:
            mfcc_stats.extend(np.mean(coeff, axis=1))
            mfcc_stats.extend(np.std(coeff, axis=1))
            mfcc_stats.extend(np.max(coeff, axis=1))
            mfcc_stats.extend(np.min(coeff, axis=1))
        
        # 2. Spectral contrast (good for percussive sounds)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        contrast_stats = np.concatenate([np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1)])
        
        # 3. Chroma features (good for distinguishing from music)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_stats = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
        
        # Combine all spectral features
        return np.concatenate([mfcc_stats, contrast_stats, chroma_stats])
    
    def extract_onset_features(self, y):
        """
        Extract onset detection features - crucial for clap detection
        """
        # Onset strength
        onset_frames = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_frames)), sr=self.sr)
        
        # Detect onsets
        onsets = librosa.onset.onset_detect(y=y, sr=self.sr, units='time')
        
        features = [
            len(onsets),  # Number of onsets
            np.mean(onset_frames),  # Mean onset strength
            np.std(onset_frames),   # Std onset strength
            np.max(onset_frames),   # Max onset strength
        ]
        
        # Onset interval analysis (key for double claps)
        if len(onsets) >= 2:
            intervals = np.diff(onsets)
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                len(intervals[intervals < 0.5])  # Onsets within 500ms
            ])
        else:
            features.extend([0, 0, 0])
            
        return np.array(features)
    
    def extract_all_features(self, y):
        """
        Combine all feature extraction methods
        """
        # Preprocess the audio
        y_clean = self.preprocess_audio(y)
        
        # Extract different types of features
        temporal_features = self.extract_temporal_features(y_clean)
        spectral_features = self.extract_spectral_features(y_clean)
        onset_features = self.extract_onset_features(y_clean)
        
        # Combine all features
        all_features = np.concatenate([temporal_features, spectral_features, onset_features])
        
        return all_features
    
    def extract_from_file(self, file_path):
        """
        Extract features from audio file
        """
        y, sr = librosa.load(file_path, sr=self.sr)
        return self.extract_all_features(y)
    
    def visualize_audio(self, y, title="Audio Analysis"):
        """
        Visualize audio for debugging
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Waveform
        time = np.linspace(0, len(y)/self.sr, len(y))
        axes[0].plot(time, y)
        axes[0].set_title('Waveform')
        axes[0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = axes[1].imshow(D, aspect='auto', origin='lower', 
                           extent=[0, len(y)/self.sr, 0, self.sr/2])
        axes[1].set_title('Spectrogram')
        axes[1].set_ylabel('Frequency (Hz)')
        
        # Onset strength
        onset_frames = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_frames)), sr=self.sr)
        axes[2].plot(onset_times, onset_frames)
        axes[2].set_title('Onset Strength')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Onset Strength')
        
        plt.tight_layout()
        plt.suptitle(title)
        plt.show()


def process_dataset(raw_folder, output_folder, extractor):
    """
    Process entire dataset and extract features
    """
    os.makedirs(output_folder, exist_ok=True)
    
    X, y = [], []
    
    # Define folder mappings: (folder_name, label)
    folder_mappings = [
        ("double_clap", 1),
        ("negative", 0)
    ]
    
    # Process each folder type
    for folder_name, label in folder_mappings:
        folder_path = os.path.join(raw_folder, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            if file.endswith(('.wav', '.mp3')):
                try:
                    features = extractor.extract_from_file(os.path.join(folder_path, file))
                    X.append(features)
                    y.append(label)
                    print(f"Processed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Double claps: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
    
    # Save processed data
    np.save(os.path.join(output_folder, "X.npy"), X)
    np.save(os.path.join(output_folder, "y.npy"), y)
    
    return X, y


if __name__ == "__main__":
    # Initialize feature extractor
    extractor = DoubleClap(sr=22050)
    
    # Process the dataset
    raw_folder = "data/raw"
    output_folder = "data/processed"
    
    X, y = process_dataset(raw_folder, output_folder, extractor)
    
    print("Feature extraction completed!")
    print(f"Total samples: {len(X)}")
    if len(X) > 0:
        print(f"Feature dimension: {X.shape[1]}")
    else:
        print("No samples were processed.")
