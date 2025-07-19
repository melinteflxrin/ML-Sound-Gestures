import librosa
import numpy as np
import os


def extract_mfcc_from_audio(y, sr, n_mfcc=13):
    """
    extract features from a numpy audio array: mean/std of MFCCs/deltas, zero-crossing rate, 
    spectral contrast, and flattened MFCC frames
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # mean and std
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)
    # flatten first 10 frames of MFCC
    mfcc_flat = mfcc[:, :10].flatten() if mfcc.shape[1] >= 10 else mfcc.flatten()
    # zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    # spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    spec_contrast_std = np.std(spec_contrast, axis=1)
    # concatenate all features
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        mfcc_flat,
        [zcr_mean, zcr_std],
        spec_contrast_mean, spec_contrast_std
    ])
    return features


def extract_mfcc(file_path, n_mfcc=13):
    """
    extract features from a file path (for training/processing)
    """
    y, sr = librosa.load(file_path, sr=None)
    features = extract_mfcc_from_audio(y, sr, n_mfcc=n_mfcc)
    # added original max amplitude as a feature
    orig_max_amp = np.max(np.abs(y))
    features = np.concatenate([features, [orig_max_amp]])
    return features


def process_folder(folder_path, label):
    """
    process all audio files in a folder and return features and labels
    """
    X = [] # features
    y = [] # labels
    for file in os.listdir(folder_path):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(label)
    return X, y


def main():
    double_clap_folder = "data/normalized/double_clap"
    finger_snap_folder = "data/normalized/finger_snap"
    negative_folder = "data/normalized/negative"

    X_clap, y_clap = process_folder(double_clap_folder, 1)
    X_snap, y_snap = process_folder(finger_snap_folder, 2)
    X_neg, y_neg = process_folder(negative_folder, 0)

    # combine features and labels
    X = np.array(X_clap + X_snap + X_neg)
    y = np.array(y_clap + y_snap + y_neg)

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)

    # save processed data
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)


if __name__ == "__main__":
    main()
