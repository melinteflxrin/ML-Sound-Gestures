import librosa
import numpy as np
import os
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis


class DoubleClap:
    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        
    def preprocess_audio(self, y, target_length=1.5):
        """
        preprocess audio with noise reduction and normalization
        """
        # remove DC offset
        y = y - np.mean(y)
        
        # high pass filter
        b, a = butter(3, 100, btype='high', fs=self.sr)
        y = filtfilt(b, a, y)
        
        # normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # trim or pad to target length
        target_samples = int(target_length * self.sr)
        if len(y) > target_samples:
            # find loudest part
            energy = np.convolve(y**2, np.ones(target_samples), mode='valid')
            start_idx = np.argmax(energy)
            y = y[start_idx:start_idx + target_samples]
        else:
            # pad with zeros
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
            
        return y
    
    def extract_temporal_features(self, y):
        """
        extract temporal features for double clap detection
        """
        features = []
        
        # 1. peak based features
        peaks, properties = find_peaks(np.abs(y), height=0.1, distance=int(0.05 * self.sr))
        
        # number of significant peaks
        features.append(len(peaks))
        
        # peak spacing 
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / self.sr  # convert to seconds
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.min(intervals),
                np.max(intervals),
                len(intervals[intervals < 0.5])  # peaks within 500ms (double clap range)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 2. energy envelope features
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.01 * self.sr)    # 10ms hop
        
        #  RMS energy envelope
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # energy envelope characteristics - "shape" of the clap
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms),
            skew(rms),
            kurtosis(rms)
        ])
        
        # 3. zero crossing rate patterns - how often the audio signal crosses the zero line
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr),
            np.max(zcr)
        ])
        
        # 4. spectral features for transient detection - frequency spectrum
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
        extract spectral features optimized for clap sounds
        """
        # 1. MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=8)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # stat features for MFCC coefficients
        mfcc_stats = []
        for coeff in [mfcc, mfcc_delta, mfcc_delta2]:
            mfcc_stats.extend(np.mean(coeff, axis=1))
            mfcc_stats.extend(np.std(coeff, axis=1))
            mfcc_stats.extend(np.max(coeff, axis=1))
            mfcc_stats.extend(np.min(coeff, axis=1))
        
        # 2. spectral contrast - good for percussion - how "spiky" vs "smooth" the frequency is - 
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        contrast_stats = np.concatenate([np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1)])
        
        # 3. chroma features - good for distinguishing from music - analyze the musical pitch content
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_stats = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
        
        # combine all spectral features
        return np.concatenate([mfcc_stats, contrast_stats, chroma_stats])
    
    def extract_onset_features(self, y):
        """
        extract onset detection features - very important for clap detection
        = the attack of the sound
        """
        # onset strength
        onset_frames = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_frames)), sr=self.sr)
        
        # detect onsets
        onsets = librosa.onset.onset_detect(y=y, sr=self.sr, units='time')
        
        features = [
            len(onsets),  # number of onsets
            np.mean(onset_frames),  # mean onset strength
            np.std(onset_frames),   # std onset 
            np.max(onset_frames),   # max onset 
        ]
        
        # onset interval analysis
        if len(onsets) >= 2:
            intervals = np.diff(onsets)
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                len(intervals[intervals < 0.5])  # onsets within 500ms
            ])
        else:
            features.extend([0, 0, 0])
            
        return np.array(features)
    
    def extract_all_features(self, y):
        """
        combine all feature extraction methods
        """
        # preprocess the audio
        y_clean = self.preprocess_audio(y)
        
        # extract features
        temporal_features = self.extract_temporal_features(y_clean)
        spectral_features = self.extract_spectral_features(y_clean)
        onset_features = self.extract_onset_features(y_clean)
        
        # combine all features
        all_features = np.concatenate([temporal_features, spectral_features, onset_features])
        
        return all_features
    
    def extract_from_file(self, file_path):
        """
        extract features from audio file
        """
        y, sr = librosa.load(file_path, sr=self.sr)
        return self.extract_all_features(y)


def process_dataset(raw_folder, output_folder, extractor):
    """
    process whole dataset and extract features
    """
    os.makedirs(output_folder, exist_ok=True)
    
    X, y = [], []
    
    # (folder_name, label)
    folder_mappings = [
        ("double_clap", 1),
        ("negative", 0)
    ]
    
    # process each folder type
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
    
    # convert to numpy arrays
    X, y = np.array(X), np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Double claps: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
    
    # save processed data
    np.save(os.path.join(output_folder, "X.npy"), X)
    np.save(os.path.join(output_folder, "y.npy"), y)
    
    return X, y


if __name__ == "__main__":
    # initialize feature extractor
    extractor = DoubleClap(sr=22050)
    
    # process dataset
    raw_folder = "data/raw"
    output_folder = "data/processed"
    
    X, y = process_dataset(raw_folder, output_folder, extractor)
    
    print("Feature extraction completed!")
    print(f"Total samples: {len(X)}")
    if len(X) > 0:
        print(f"Feature dimension: {X.shape[1]}")
    else:
        print("No samples were processed.")
