import os
import torch
import os,sys
PROJECT_ROOT = os.path.abspath("..")  
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import librosa
import soundfile as sf
import numpy as np
from pystoi import stoi

from torch.utils.data import DataLoader
from pystoi import stoi
from src.dataset import SpeechEnhancementDataset
from src.model import Conv1DEnhancer, ResNetEnhancer , UNet1D, DCCRN
import matplotlib.pyplot as plt

def plot_results(clean, noisy, enhanced, sr, idx, output_dir):
    """
    Plot waveform and spectrogram for one sample.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))

    titles = ["Clean", "Noisy", "Enhanced"]
    signals = [clean, noisy, enhanced]

    for i in range(3):
        # Waveform
        axs[i, 0].plot(signals[i])
        axs[i, 0].set_title(f"{titles[i]} Waveform")
        axs[i, 0].set_xlabel("Time")
        axs[i, 0].set_ylabel("Amplitude")

        # Spectrogram
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(signals[i])), ref=np.max)
        img = axs[i, 1].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axs[i, 1].set_title(f"{titles[i]} Spectrogram")
        fig.colorbar(img, ax=axs[i, 1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_{idx+1}.png"))
    plt.close(fig)


def evaluate_model(
    model_type="DCCRN",
    checkpoint_path="../outputs/checkpoints/DCCRN_epoch50.pth",
    data_dir="../data",
    output_dir="../outputs/predictions",
    batch_size=1,
    sample_rate=48000,
    target_len=None,
    device=None
):
    """
    Evaluate a trained speech enhancement model on the test dataset.
    Computes STOI and visualizes results (waveform + spectrogram).
    """

    # -----------------------------
    # 1. Setup
    # -----------------------------
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy_test_dir = os.path.join(data_dir, "N_TS")
    clean_test_dir = os.path.join(data_dir, "CL_TS")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 2. Load Dataset
    # -----------------------------
    test_dataset = SpeechEnhancementDataset(
        noisy_dir=noisy_test_dir,
        clean_dir=clean_test_dir,
        target_len=target_len or sample_rate * 4,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 3. Load Model
    # -----------------------------
    if model_type == "Conv1D":
        model = Conv1DEnhancer()
    elif model_type == "ResNet":
        model = ResNetEnhancer()
    elif model_type == 'UNet':
        model = UNet1D()
    elif model_type == "DCCRN":
        model = DCCRN()
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose 'Conv1D', 'ResNet', or 'DCCRN'")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f" Loaded {model_type} from {checkpoint_path} on {device}")

    # -----------------------------
    # 4. Evaluation Loop
    # -----------------------------
    stoi_scores = []

    for i, (noisy, clean) in enumerate(test_loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            enhanced = model(noisy)

        enhanced_np = enhanced.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()
        noisy_np = noisy.squeeze().cpu().numpy()


        output_path = os.path.join(output_dir, f"enhanced_{i+1}.wav")
        sf.write(output_path, enhanced_np, sample_rate)

        # Compute STOI
        try:
            stoi_score = stoi(clean_np, enhanced_np, sample_rate, extended=False)
        except Exception as e:
            print(f" STOI error for sample {i+1}: {e}")
            stoi_score = None

        stoi_scores.append(stoi_score)
        print(f" Sample {i+1}: STOI={stoi_score:.3f}" if stoi_score else f"Sample {i+1}: skipped")


        plot_results(clean_np, noisy_np, enhanced_np, sample_rate, i, output_dir)

    # -----------------------------
    # 5. Average STOI
    # -----------------------------
    valid_scores = [s for s in stoi_scores if s is not None]
    avg_stoi = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    print(f"\n Average STOI: {avg_stoi:.3f}")
    return avg_stoi


if __name__ == "__main__":
    evaluate_model(
        model_type="Conv1D",

        checkpoint_path="../outputs/checkpoints/Conv1D_epoch10.pth",
        data_dir="../data",
        output_dir="../outputs/predictions",
        batch_size=1,
        sample_rate=48000,
    )
