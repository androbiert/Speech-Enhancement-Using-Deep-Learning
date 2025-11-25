# src/preprocessing.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from typing import List, Tuple
from src.utils import load_audio  # normalized version, used for visualization if needed

# -------------------------
# EDA FUNCTIONS
# -------------------------

def explore_folder(folder: str, sr: int = 48000) -> Tuple[List[str], List[float], List[float]]:
    """
    Explore audio folder WITHOUT normalization (raw audio)
    Returns: file names, durations (s), peak amplitudes
    """
    files = sorted(os.listdir(folder))
    durations, peaks = [], []

    for f in files:
        path = os.path.join(folder, f)
        audio, _ = librosa.load(path, sr=sr)  # raw audio, no normalization
        durations.append(len(audio) / sr)
        
        
        peaks.append(np.max(np.abs(audio)))
    print(peaks)

    print(f"Folder: {folder}")
    print(f"Number of files: {len(files)}")
    print(f"Duration (s): min={min(durations):.2f}, max={max(durations):.2f}, mean={np.mean(durations):.2f}, std={np.std(durations):.2f}")
    print(f"Peak amplitude: min={min(peaks):.2f}, max={max(peaks):.2f}, mean={np.mean(peaks):.2f}, std={np.std(peaks):.2f}")

    return files, durations, peaks

def plot_duration_distribution(durations_clean: List[float], durations_noisy: List[float]):
    plt.figure(figsize=(10,4))
    sns.histplot(durations_noisy, color='r', label='Noisy', kde=True)
    sns.histplot(durations_clean, color='g', label='Clean', kde=True)
    plt.title("Audio Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.legend()
    plt.show()

def plot_peak_distribution(peaks_clean: List[float], peaks_noisy: List[float]):
    plt.figure(figsize=(10,4))
    sns.histplot(peaks_noisy, color='b', label='Noisy', kde=True, alpha=0.8)
    
    sns.histplot(peaks_clean, color='g', label='Clean', kde=True)
    plt.title("Peak Amplitude Distribution")
    plt.xlabel("Amplitude")
    plt.legend()
    plt.show()

def plot_waveform(noisy_audio: np.ndarray, clean_audio: np.ndarray, title="Waveform Sample"):
    plt.figure(figsize=(12,4))
    plt.plot(noisy_audio, label='Noisy')
    plt.plot(clean_audio, label='Clean', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_spectrogram(audio: np.ndarray, sr: int = 48000, title="Spectrogram"):
    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def plot_random_samples(noisy_folder: str, clean_folder: str, sr: int = 48000, n_samples: int = 3):
    """
    Randomly pick n_samples from the dataset and plot waveform + spectrogram
    """
    noisy_files = sorted(os.listdir(noisy_folder))
    clean_files = sorted(os.listdir(clean_folder))

    for _ in range(n_samples):
        idx = random.randint(0, len(noisy_files)-1)
        noisy_audio, _ = librosa.load(os.path.join(noisy_folder, noisy_files[idx]), sr=sr)
        clean_audio, _ = librosa.load(os.path.join(clean_folder, clean_files[idx]), sr=sr)
        print(f"Sample: {noisy_files[idx]} / {clean_files[idx]}")
        plot_waveform(noisy_audio, clean_audio)
        plot_spectrogram(noisy_audio, title="Noisy Spectrogram")
        plot_spectrogram(clean_audio, title="Clean Spectrogram")
