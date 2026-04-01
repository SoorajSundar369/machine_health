"""
=============================================================
  MACHINE HEALTH MONITOR  —  Step 2: Train Classifiers
=============================================================
Run this SECOND (after generate_data.py has created the data/).

What this script does:
  1. Loads every WAV file from data/normal, data/mechanical,
     data/electrical, data/wear
  2. Cleans each file with the noise filter
  3. Extracts a 43-number feature vector from each file
  4. Trains TWO classifiers:
       - Binary model    : Is the machine HEALTHY or FAULTY?
       - Multiclass model: If faulty, WHAT TYPE of fault?
  5. Saves both models to models/

Usage:
    python train.py
"""

import os
import sys
import numpy as np
import librosa
import joblib
from sklearn.ensemble  import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import noise_filter   # our noise removal module (noise_filter.py)

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
N_MFCC     = 40          # number of MFCC coefficients to extract

# Must match the folder names in generate_data.py
CLASSES    = ["normal", "mechanical", "electrical", "wear"]


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(audio, sr):
    """
    Converts a raw audio clip into a fixed-size feature vector of 43 numbers.

    Why? Machine learning models cannot directly understand audio waveforms.
    We need to turn the sound into meaningful numbers that describe the sound.

    Features extracted:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  MFCCs (40 values)          — describe the overall 'texture' and    │
    │                               tonal quality of the sound.           │
    │                               Think of it as the sound's fingerprint│
    │                                                                     │
    │  Spectral Centroid (1 val)  — the 'brightness' of the sound.       │
    │                               High = bright/harsh. Low = dull/deep. │
    │                                                                     │
    │  Spectral Rolloff  (1 val)  — frequency below which 85% of energy  │
    │                               is concentrated. Indicates bandwidth. │
    │                                                                     │
    │  Zero-Crossing Rate (1 val) — how often the signal crosses zero.   │
    │                               High = noisy/percussive sounds.       │
    └─────────────────────────────────────────────────────────────────────┘
    Total: 40 + 1 + 1 + 1 = 43 features per audio clip.

    Parameters
    ----------
    audio : numpy array  — the cleaned audio signal
    sr    : int          — sample rate

    Returns
    -------
    numpy array of shape (43,)
    """
    # 40 MFCC coefficients, averaged across all time frames → shape (40,)
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Single-value features (each is the mean across all time frames)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio,  sr=sr)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

    # Combine everything into one flat array: 40 + 1 + 1 + 1 = 43 values
    return np.concatenate([mfcc_mean, [centroid, rolloff, zcr]])


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset():
    """
    Walks through the data/ folder and loads every WAV file.

    For each file:
      - Load audio with librosa
      - Apply noise_filter.clean_signal() to remove background noise
      - Extract the 43-feature vector
      - Record the class label (folder name)

    Returns
    -------
    X      : numpy array (n_files, 43) — one feature vector per file
    y      : numpy array (n_files,)    — class label strings
    counts : dict                      — number of files loaded per class
    """
    X, y, counts = [], [], {}

    for class_name in CLASSES:
        folder = os.path.join(DATA_DIR, class_name)

        if not os.path.exists(folder):
            print(f"  [!]  Folder missing: {folder}  — skipping")
            continue

        wav_files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        counts[class_name] = len(wav_files)

        for filename in wav_files:
            filepath = os.path.join(folder, filename)
            try:
                audio, sr = librosa.load(filepath, sr=None)
                audio     = noise_filter.clean_signal(audio, sr)
                feat      = extract_features(audio, sr)
                X.append(feat)
                y.append(class_name)
            except Exception as err:
                print(f"  [!]  Could not process {filepath}: {err}")

    return np.array(X, dtype=np.float32), np.array(y), counts


# ─── ASCII Bar Chart ──────────────────────────────────────────────────────────

def print_class_chart(counts):
    """
    Prints a simple ASCII bar chart so you can see at a glance
    how many training samples exist per class.

    Example output:
      normal       ████████████████████  30
      mechanical   ████████████████████  30
      electrical   ████████████████████  30
      wear         ████████████████████  30
    """
    if not counts:
        return
    max_count = max(counts.values())
    BAR_MAX   = 28   # maximum bar length in characters

    print("\n  Training data loaded:")
    print("  " + "─" * 46)
    for name in CLASSES:
        count   = counts.get(name, 0)
        bar_len = int(count / max_count * BAR_MAX) if max_count > 0 else 0
        bar     = "█" * bar_len
        print(f"  {name:<12}  {bar:<{BAR_MAX}}  {count}")
    print("  " + "─" * 46)


# ─── Main Training ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 54)
    print("   MACHINE HEALTH MONITOR  —  Model Training")
    print("=" * 54)

    # ── Load and process all audio files ──────────────────────────────────
    print("\n  Loading audio files and extracting features...")
    print("  (This may take a minute — processing", end=" ", flush=True)

    X, y_labels, counts = load_dataset()

    print(f"{len(X)} files)\n")

    if len(X) == 0:
        print("  ERROR: No audio files found.")
        print("  Please run  python generate_data.py  first!\n")
        sys.exit(1)

    print(f"  Feature vector size : {X.shape[1]} values per clip")
    print_class_chart(counts)

    # ── Encode class names as integers ────────────────────────────────────
    # LabelEncoder turns "normal" → 0, "mechanical" → 1, etc.
    # The exact mapping depends on alphabetical order of class names.
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)   # integer labels

    print(f"\n  Class encoding: ", end="")
    for i, name in enumerate(le.classes_):
        print(f"{name}={i}", end="  ")
    print()

    # ── Binary labels: 0 = normal/healthy, 1 = any fault ─────────────────
    normal_idx = list(le.classes_).index("normal")
    y_binary   = (y != normal_idx).astype(int)
    binary_pos = np.sum(y_binary)
    print(f"  Binary split    :  {len(y) - binary_pos} healthy  /  {binary_pos} faulty")

    # ── Train multiclass model (4 classes) ────────────────────────────────
    print("\n  [1/2] Training multiclass classifier (normal / mechanical / electrical / wear)...")
    clf_multi = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1        # use all CPU cores
    )
    clf_multi.fit(X, y)
    acc_multi = cross_val_score(clf_multi, X, y, cv=5).mean()
    print(f"        5-fold cross-validation accuracy: {acc_multi * 100:.1f}%")

    # ── Train binary model (healthy vs faulty) ────────────────────────────
    print("\n  [2/2] Training binary classifier (healthy vs faulty)...")
    clf_binary = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf_binary.fit(X, y_binary)
    acc_binary = cross_val_score(clf_binary, X, y_binary, cv=5).mean()
    print(f"        5-fold cross-validation accuracy: {acc_binary * 100:.1f}%")

    # ── Save models to disk ───────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf_binary,  os.path.join(MODEL_DIR, "model_binary.pkl"))
    joblib.dump(clf_multi,   os.path.join(MODEL_DIR, "model_multiclass.pkl"))
    joblib.dump(le,          os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print(f"\n  Models saved to  {MODEL_DIR}/")
    print(f"    model_binary.pkl       — healthy vs faulty")
    print(f"    model_multiclass.pkl   — fault type classifier")
    print(f"    label_encoder.pkl      — class name ↔ integer mapping")

    print()
    print("=" * 54)
    print()
    print("  DONE! Next step:")
    print("    python predict.py")
    print()
