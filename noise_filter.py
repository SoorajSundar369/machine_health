"""
=============================================================
  MACHINE HEALTH MONITOR  —  Noise Filter Module
=============================================================
This module removes background noise from audio signals.

Technique used: SPECTRAL SUBTRACTION
  1. Look at the first 0.5 seconds of the recording (assumed quiet / pre-machine)
  2. Measure what frequencies are present — that's the background noise profile
  3. Subtract that noise profile from the entire recording
  4. Return the cleaned audio

This file is used by both train.py and predict.py.
You don't need to run this file directly.
"""

import numpy as np
import librosa


def clean_signal(audio, sr):
    """
    Removes stationary background noise from an audio signal.

    Parameters
    ----------
    audio : numpy array
        The raw audio signal (1D array of float values).
    sr : int
        Sample rate of the audio (e.g. 22050 means 22050 samples per second).

    Returns
    -------
    numpy array
        The cleaned audio signal, same length as input, as float32.

    How it works (plain English):
    ─────────────────────────────
    Imagine the audio is a mix of the machine sound (signal) and background
    noise (fans, traffic, other machines). The first 0.5 seconds are likely
    just noise before the machine kicks in. We measure that noise, then
    subtract it from the whole recording — like erasing the background.

    Steps:
    1. Convert both the noise sample and full audio to the frequency domain
       using STFT (Short-Time Fourier Transform). This shows how strong each
       frequency is at each moment in time.
    2. Average the noise frequencies across the 0.5s sample → noise profile.
    3. Subtract 2× the noise profile from the full signal (over-subtract a bit
       to be aggressive about cleaning).
    4. Clip to a minimum of 1% of original magnitude so we don't create silence.
    5. Convert back to time-domain audio.
    """

    # ── Guard: audio must be long enough for a noise sample ──────────────
    noise_length = int(0.5 * sr)   # 0.5 seconds worth of samples
    if len(audio) < noise_length:
        # Too short to estimate noise — return audio unchanged
        return audio.astype(np.float32)

    # ── Step 1: Isolate the noise sample (first 0.5 seconds) ─────────────
    noise_segment = audio[:noise_length]

    # ── Step 2: Convert to frequency domain with STFT ────────────────────
    # n_fft=1024 : chunk size (balances frequency resolution vs time resolution)
    # hop_length=256 : how much we slide the window each step
    stft_full  = librosa.stft(audio,          n_fft=1024, hop_length=256)
    stft_noise = librosa.stft(noise_segment,  n_fft=1024, hop_length=256)

    # Separate magnitude (strength) and phase (timing) of each frequency
    magnitude  = np.abs(stft_full)    # how strong each frequency is
    phase      = np.angle(stft_full)  # the timing component (needed to rebuild audio)

    # ── Step 3: Estimate noise profile ───────────────────────────────────
    # Average across all time frames → one noise level per frequency bin
    # keepdims=True keeps the shape so broadcasting works in subtraction
    noise_profile = np.mean(np.abs(stft_noise), axis=1, keepdims=True)

    # ── Step 4: Subtract noise from signal ───────────────────────────────
    # Over-subtract by 2× (aggressive cleaning)
    # np.maximum(..., 0.01 * magnitude) floors at 1% to avoid silent gaps
    clean_magnitude = np.maximum(
        magnitude - 2.0 * noise_profile,
        0.01 * magnitude
    )

    # ── Step 5: Reconstruct the time-domain audio ─────────────────────────
    # Combine cleaned magnitude with original phase, then inverse STFT
    clean_stft = clean_magnitude * np.exp(1j * phase)
    cleaned    = librosa.istft(clean_stft, hop_length=256, length=len(audio))

    # ── Normalize volume ──────────────────────────────────────────────────
    peak = np.max(np.abs(cleaned))
    if peak > 0:
        cleaned = cleaned / peak * 0.9

    return cleaned.astype(np.float32)
