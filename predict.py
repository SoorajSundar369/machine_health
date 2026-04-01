"""
=============================================================
  MACHINE HEALTH MONITOR  —  Step 3: Predict & Display
=============================================================
Run this LAST (after generate_data.py and train.py).

What this script does:
  - Loads a WAV file
  - Cleans the audio (noise removal)
  - Extracts features
  - Runs binary classifier  : is the machine healthy or faulty?
  - If faulty, runs multiclass classifier : what kind of fault?
  - Calculates a health score out of 100
  - Prints a formatted, colour-coded health report

Usage:
    python predict.py                       (runs 3 built-in demo files)
    python predict.py data/wear/wear_010.wav   (predict any single file)
"""

import os
import sys
import re
import numpy as np
import librosa
import joblib

import noise_filter   # our noise removal module

# ─── Enable ANSI colours on Windows 10+ ─────────────────────────────────────
os.system("")   # this one-liner activates colour support in Windows terminal

# ─── ANSI colour / style codes ───────────────────────────────────────────────
# These are invisible control characters that tell the terminal to change colour.
# They work on Windows 10+, macOS, and Linux.
GREEN   = "\033[92m"    # bright green
RED     = "\033[91m"    # bright red
YELLOW  = "\033[93m"    # bright yellow
CYAN    = "\033[96m"    # bright cyan
WHITE   = "\033[97m"    # bright white
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"     # back to default colour

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_DIR = "models"
N_MFCC    = 40   # must match train.py exactly

# Health score weight per class.
# The final health score is the weighted average using prediction probabilities.
# Example: 94% mechanical → score ≈ 0.94×20 + 0.06×(other weights) ≈ 22
HEALTH_WEIGHT = {
    "normal":     95,   # healthy machine: high score
    "mechanical": 20,   # bearing/mechanical fault: critical
    "electrical": 35,   # electrical fault: moderate-critical
    "wear":       18,   # wear & tear: very low
}

# What to do for each fault type
ACTIONS = {
    "normal":     "No action needed — machine is running well.",
    "mechanical": "Stop machine immediately. Inspect bearings and rotating parts.",
    "electrical": "Shut down and inspect wiring, motor windings, and grounding.",
    "wear":       "Schedule overhaul soon. Plan for part replacement.",
}

# Short action keywords for the score card
ACTION_SHORT = {
    "normal":     "Continue normal operation",
    "mechanical": "Immediate maintenance required",
    "electrical": "Electrical inspection required",
    "wear":       "Plan for part replacement",
}

# Severity label per class (for display)
SEVERITY = {
    "normal":     "NONE",
    "mechanical": "CRITICAL",
    "electrical": "HIGH",
    "wear":       "MODERATE",
}

# Status display colour per class
CLASS_COLOR = {
    "normal":     GREEN,
    "mechanical": RED,
    "electrical": RED,
    "wear":       YELLOW,
}


# ─── ANSI-aware string padding ────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def visible_len(text):
    """Returns the visible length of a string, ignoring ANSI escape codes."""
    return len(_ANSI_RE.sub("", text))

def pad_right(text, width):
    """
    Right-pads a string to a given VISIBLE width.
    Necessary because ANSI codes have zero visual width but Python counts them.
    """
    extra = width - visible_len(text)
    return text + " " * max(0, extra)


# ─── Box drawing helpers ──────────────────────────────────────────────────────

INNER = 52   # visible width of content inside the box (between ║ borders)

def box_top():
    return "  ╔" + "═" * (INNER + 2) + "╗"

def box_sep():
    return "  ╠" + "═" * (INNER + 2) + "╣"

def box_bot():
    return "  ╚" + "═" * (INNER + 2) + "╝"

def box_blank():
    return "  ║" + " " * (INNER + 2) + "║"

def box_row(content):
    """Wraps content in a box row, padding to the correct visual width."""
    padded = pad_right(content, INNER)
    return "  ║ " + padded + " ║"

def box_title(title):
    """Centres a title string inside the box row."""
    # Build visible-centred title without ANSI problems
    spaces_total = INNER - visible_len(title)
    left  = spaces_total // 2
    right = spaces_total - left
    return "  ║" + " " * (left + 1) + title + " " * (right + 1) + "║"


# ─── Health bar ───────────────────────────────────────────────────────────────

def health_bar(score, bar_width=34):
    """
    Returns a visual health bar like:  [████████░░░░░░░░░░░░░░░░░░░░░░] 23 / 100

    Colour:
      Green  — score 70–100  (healthy)
      Yellow — score 40–69   (degraded)
      Red    — score 0–39    (critical)
    """
    filled = int(score / 100 * bar_width)
    empty  = bar_width - filled

    if score >= 70:
        bar_color = GREEN
    elif score >= 40:
        bar_color = YELLOW
    else:
        bar_color = RED

    filled_str = bar_color + "█" * filled + RESET
    empty_str  = DIM      + "░" * empty  + RESET
    label      = f"{score} / 100"

    return "[" + filled_str + empty_str + "] " + label


# ─── Feature Extraction  (must match train.py exactly) ───────────────────────

def extract_features(audio, sr):
    """
    Converts an audio clip into a 43-number feature vector.
    This is the SAME function as in train.py — they must stay identical.
    """
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio,  sr=sr)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    return np.concatenate([mfcc_mean, [centroid, rolloff, zcr]])


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_models():
    """
    Loads the two classifiers and the label encoder from the models/ folder.
    If any file is missing, prints a helpful error and exits.
    """
    needed = {
        "Binary classifier":    os.path.join(MODEL_DIR, "model_binary.pkl"),
        "Multiclass classifier": os.path.join(MODEL_DIR, "model_multiclass.pkl"),
        "Label encoder":        os.path.join(MODEL_DIR, "label_encoder.pkl"),
    }
    for name, path in needed.items():
        if not os.path.exists(path):
            print(f"\n  ERROR: {name} not found at: {path}")
            print("  Please run  python train.py  first!\n")
            sys.exit(1)

    clf_binary = joblib.load(needed["Binary classifier"])
    clf_multi  = joblib.load(needed["Multiclass classifier"])
    le         = joblib.load(needed["Label encoder"])
    return clf_binary, clf_multi, le


# ─── Main Prediction Function ─────────────────────────────────────────────────

def predict_sound(filepath):
    """
    Analyses a machine sound file and prints a full health report.

    Steps:
      1. Load the WAV file
      2. Apply noise filter to clean the audio
      3. Extract 43 audio features
      4. Run binary classifier  → is it healthy or faulty?
      5. Run multiclass classifier → what type of fault is it?
      6. Calculate health score (weighted average of class scores × probabilities)
      7. Print the formatted report

    Parameters
    ----------
    filepath : str  — path to a WAV file

    Returns
    -------
    dict with keys: status, fault_type, confidence, health_score, action
    Returns None if the file cannot be found or loaded.
    """

    # ── Load models (cached automatically by Python if called multiple times) ─
    clf_bin, clf_multi, le = load_models()

    # ── Check file exists ─────────────────────────────────────────────────────
    if not os.path.exists(filepath):
        print(f"\n  ERROR: File not found — {filepath}")
        print("  Tip: run  python generate_data.py  first to create test files.\n")
        return None

    filename = os.path.basename(filepath)

    # ── Processing steps with progress output ────────────────────────────────
    print()
    print(f"  File      : {filename}")
    print(f"  {DIM}{'─'*46}{RESET}")

    print(f"  Loading audio...       ", end="", flush=True)
    try:
        audio, sr = librosa.load(filepath, sr=None)
    except Exception as e:
        print(f"FAILED\n  Error: {e}\n")
        return None
    print("done")

    print(f"  Removing noise...      ", end="", flush=True)
    audio_clean = noise_filter.clean_signal(audio, sr)
    print("done")

    print(f"  Extracting features... ", end="", flush=True)
    feat = extract_features(audio_clean, sr).reshape(1, -1)
    print("done")

    print(f"  Running classifier...  ", end="", flush=True)

    # Binary: 0 = healthy, 1 = faulty
    is_faulty  = bool(clf_bin.predict(feat)[0])
    bin_proba  = clf_bin.predict_proba(feat)[0]     # [P(healthy), P(faulty)]

    # Multiclass: get the predicted class integer and all class probabilities
    multi_pred  = int(clf_multi.predict(feat)[0])
    multi_proba = clf_multi.predict_proba(feat)[0]  # probabilities for each class
    fault_name  = le.inverse_transform([multi_pred])[0]   # e.g. "mechanical"
    confidence  = int(round(max(multi_proba) * 100))

    print("done")

    # ── Health score: weighted average across all class probabilities ──────────
    # Each class has a preset "health weight" (normal=95, mechanical=20, etc.)
    # We multiply each class's probability by its weight and sum them up.
    # This means even if we're not 100% sure, the score reflects uncertainty.
    class_names  = list(le.classes_)
    health_score = int(round(
        sum(multi_proba[i] * HEALTH_WEIGHT[class_names[i]] for i in range(len(class_names)))
    ))

    # ── Determine display values ───────────────────────────────────────────────
    if is_faulty:
        status      = "FAULTY"
        fault_label = fault_name.capitalize()
        severity    = SEVERITY.get(fault_name, "HIGH")
        color       = CLASS_COLOR.get(fault_name, RED)
        action      = ACTION_SHORT.get(fault_name, "Inspect machine")
        full_action = ACTIONS.get(fault_name, "Inspect machine")
    else:
        status      = "HEALTHY"
        fault_label = "None"
        severity    = "NONE"
        color       = GREEN
        action      = ACTION_SHORT["normal"]
        full_action = ACTIONS["normal"]
        # Override health score to be high if binary says healthy
        health_score = max(health_score, 75)

    # ── Print the report box ──────────────────────────────────────────────────
    print()
    print(box_top())
    print(box_title(BOLD + WHITE + "  MACHINE HEALTH REPORT  " + RESET))
    print(box_sep())
    print(box_blank())

    # Status row  (coloured)
    status_display = color + BOLD + f"  {status}" + RESET
    print(box_row(f"  STATUS        :  {status_display}"))

    # Confidence row
    conf_display = f"{BOLD}{confidence}%{RESET}"
    print(box_row(f"  CONFIDENCE    :  {conf_display}"))

    # Fault type row
    if is_faulty:
        fault_display = color + fault_label + RESET
    else:
        fault_display = GREEN + "None (Healthy)" + RESET
    print(box_row(f"  FAULT TYPE    :  {fault_display}"))

    # Severity row
    if severity != "NONE":
        sev_display = color + BOLD + severity + RESET
    else:
        sev_display = GREEN + severity + RESET
    print(box_row(f"  SEVERITY      :  {sev_display}"))

    print(box_blank())
    print(box_sep())
    print(box_blank())

    # Health score number
    score_label = color + BOLD + f"{health_score} / 100" + RESET
    print(box_row(f"  HEALTH SCORE  :  {score_label}"))

    # Health bar
    bar = health_bar(health_score, bar_width=34)
    print(box_row(f"  {bar}"))

    print(box_blank())
    print(box_sep())
    print(box_blank())

    # Action row
    act_display = BOLD + action + RESET
    print(box_row(f"  ACTION        :  {act_display}"))

    print(box_blank())
    print(box_bot())

    # Full action description below the box
    print(f"\n  {DIM}Recommendation: {full_action}{RESET}")
    print()

    return {
        "file":         filepath,
        "status":       status,
        "fault_type":   fault_name,
        "confidence":   confidence,
        "health_score": health_score,
        "action":       full_action,
    }


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # If a filepath argument is given, predict just that file
    if len(sys.argv) > 1:
        predict_sound(sys.argv[1])
        sys.exit(0)

    # Otherwise run 3 demo predictions
    print()
    print("=" * 58)
    print(f"   {BOLD}MACHINE HEALTH MONITOR{RESET}  —  Demo Mode")
    print("=" * 58)
    print(f"  {DIM}Running 3 test predictions (normal / mechanical / wear){RESET}")
    print("=" * 58)

    demo_files = [
        ("Normal Machine",     os.path.join("data", "normal",     "normal_005.wav")),
        ("Mechanical Fault",   os.path.join("data", "mechanical", "mechanical_007.wav")),
        ("Wear & Tear",        os.path.join("data", "wear",       "wear_003.wav")),
    ]

    for label, filepath in demo_files:
        print()
        print(f"  {'─'*54}")
        print(f"  TEST: {BOLD}{label}{RESET}")
        print(f"  {'─'*54}")
        result = predict_sound(filepath)
        if result:
            input(f"  {DIM}Press Enter for the next prediction...{RESET}\n")

    print("=" * 58)
    print(f"  {GREEN}{BOLD}Demo complete.{RESET}")
    print(f"  To analyse any file: {CYAN}python predict.py <path/to/file.wav>{RESET}")
    print("=" * 58)
    print()
