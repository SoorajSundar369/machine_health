"""
=============================================================
  MACHINE HEALTH MONITOR  —  Step 1: Generate Training Data
=============================================================
Run this FIRST before anything else.

Creates 30 synthetic audio samples for each of 4 machine states:
  data/normal/      — healthy machine hum
  data/mechanical/  — bearing defect (374 Hz + impulses)
  data/electrical/  — electrical buzz (120 Hz hum + harmonics)
  data/wear/        — wear & tear (progressively increasing noise)

Usage:
    python generate_data.py
"""

import numpy as np
import soundfile as sf
import os

# ─── Configuration ────────────────────────────────────────────────────────────
SR                = 22050          # sample rate: 22050 samples per second
DURATION          = 2.0            # each clip is 2 seconds long
N                 = int(SR * DURATION)   # total samples per clip = 44100
SAMPLES_PER_CLASS = 30             # generate 30 files per category
OUTPUT_DIR        = "data"


# ─── Helper ───────────────────────────────────────────────────────────────────

def normalize(signal):
    """
    Scales the audio so its loudest peak is at 90% volume.
    This prevents distortion (clipping) and keeps all files at a consistent level.
    """
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.9
    return signal.astype(np.float32)


# ─── Sound Generators ─────────────────────────────────────────────────────────

def make_normal(seed):
    """
    Generates a HEALTHY machine sound.

    What it sounds like: a steady, smooth motor hum.
    How it's built:
      - Strong sine wave at 50 Hz  (main motor frequency)
      - Weaker sine wave at 100 Hz (second harmonic, natural in motors)
      - Tiny random noise (just 1% — almost silent background)

    A healthy machine is predictable and clean.
    """
    np.random.seed(seed)
    t = np.linspace(0, DURATION, N)
    signal = (
        0.70 * np.sin(2 * np.pi * 50  * t) +   # main hum at 50 Hz
        0.25 * np.sin(2 * np.pi * 100 * t) +   # harmonic at 100 Hz
        0.01 * np.random.randn(N)               # very faint noise floor
    )
    return normalize(signal)


def make_mechanical(seed):
    """
    Generates a MECHANICAL FAULT sound (e.g., a worn bearing or shaft imbalance).

    What it sounds like: a hum with a rattling or clicking overlay.
    How it's built:
      - Normal motor hum (50 + 100 Hz)
      - Extra vibration at 374 Hz  — the classic "bearing defect frequency"
      - Periodic impact impulses every 1/20th second (like ball bearings hitting a crack)
      - Elevated noise floor (8% random noise)

    The 374 Hz component and the regular impacts are the giveaway signature.
    """
    np.random.seed(seed)
    t      = np.linspace(0, DURATION, N)
    signal = (
        0.50 * np.sin(2 * np.pi * 50  * t) +
        0.20 * np.sin(2 * np.pi * 100 * t) +
        0.40 * np.sin(2 * np.pi * 374 * t) +   # bearing defect frequency
        0.08 * np.random.randn(N)               # elevated noise
    )

    # Add periodic impact impulses (ball-pass impacts in a cracked bearing)
    rng           = np.random.default_rng(seed)
    impact_period = int(SR / 20)        # one impact every 50 ms (20 Hz)
    impact_width  = int(0.003 * SR)     # each impact lasts 3 milliseconds
    for start in range(0, N, impact_period):
        end = min(start + impact_width, N)
        signal[start:end] += 0.50 * rng.standard_normal(end - start)

    return normalize(signal)


def make_electrical(seed):
    """
    Generates an ELECTRICAL FAULT sound (e.g., loose wiring, failing motor windings).

    What it sounds like: a buzzing hum, like a fluorescent light buzzing loudly.
    How it's built:
      - Normal motor hum (50 Hz)
      - Strong 120 Hz buzz  — electrical interference from AC power (2× mains)
      - Harmonics at 240 Hz  — 2nd harmonic of the buzz
      - 60 Hz component  — direct mains pickup
      - Amplitude-modulated buzz  — the buzz pulses in strength (arcing signature)

    The 120 Hz dominance and the pulsing hum are the electrical fault signatures.
    """
    np.random.seed(seed)
    t      = np.linspace(0, DURATION, N)
    signal = (
        0.35 * np.sin(2 * np.pi * 50  * t) +   # motor fundamental
        0.55 * np.sin(2 * np.pi * 120 * t) +   # 120 Hz electrical buzz
        0.30 * np.sin(2 * np.pi * 240 * t) +   # 2nd harmonic
        0.15 * np.sin(2 * np.pi * 60  * t) +   # 60 Hz mains pickup
        0.05 * np.random.randn(N)
    )

    # Amplitude-modulated buzz: the 120 Hz tone pulses with the 60 Hz cycle
    # This simulates arcing or intermittent electrical contact
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * 60 * t)
    signal   += 0.25 * np.sin(2 * np.pi * 120 * t) * modulator

    return normalize(signal)


def make_wear(seed):
    """
    Generates a WEAR & TEAR sound (e.g., worn-out gear teeth or degraded bearings).

    What it sounds like: starts quieter, then gets progressively noisier and rougher.
    How it's built:
      - Weakened motor hum (only 30% strength — the machine is losing power)
      - A noise envelope that linearly grows from 10% to 100% over 2 seconds
      - This simulates a surface that's grinding worse and worse as it rotates

    The increasing-noise envelope is the unique signature of wear & tear.
    """
    np.random.seed(seed)
    t        = np.linspace(0, DURATION, N)
    envelope = np.linspace(0.10, 1.00, N)   # noise grows linearly over time
    signal   = (
        0.30 * np.sin(2 * np.pi * 50  * t) +   # weakened motor hum
        0.10 * np.sin(2 * np.pi * 100 * t) +   # faint harmonic
        envelope * 0.50 * np.random.randn(N)    # progressively increasing noise
    )
    return normalize(signal)


# ─── Main ─────────────────────────────────────────────────────────────────────

# Map: folder name → generator function
CLASSES = {
    "normal":     make_normal,
    "mechanical": make_mechanical,
    "electrical": make_electrical,
    "wear":       make_wear,
}


if __name__ == "__main__":
    print()
    print("=" * 54)
    print("   MACHINE HEALTH MONITOR  —  Data Generation")
    print("=" * 54)
    print(f"\n  Generating {SAMPLES_PER_CLASS} samples × {len(CLASSES)} classes")
    print(f"  Sample rate : {SR} Hz")
    print(f"  Duration    : {DURATION}s per clip")
    print(f"  Output dir  : {OUTPUT_DIR}/\n")

    total_files = 0

    for class_name, generator_fn in CLASSES.items():
        folder = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(folder, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            filename = os.path.join(folder, f"{class_name}_{i:03d}.wav")
            audio    = generator_fn(seed=i)
            sf.write(filename, audio, SR)

        total_files += SAMPLES_PER_CLASS
        print(f"  [OK]  {class_name:<12}  {SAMPLES_PER_CLASS} files  →  {folder}/")

    print()
    print(f"  Total: {total_files} audio files created.")
    print("=" * 54)
    print()
    print("  DONE! Next step:")
    print("    python train.py")
    print()
