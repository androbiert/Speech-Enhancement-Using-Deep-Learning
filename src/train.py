import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import argparse
from tqdm import tqdm 

# make project root importable
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import SpeechEnhancementDataset
from src.model import Conv1DEnhancer, UNet1D, ResNetEnhancer, DCCRN

try:
    from src.utils import waveform_to_magphase, waveform_to_stft_realimag
except ImportError:
    print("Warning: src.utils not found. Using placeholder utility functions.")

    def waveform_to_magphase(waveform, n_fft, hop_length):
        # Dummy implementation: returns a tensor of ones matching a plausible spectrogram shape
        # You'll need to replace this with actual STFT magnitude calculation
        batch_size, channels, time_len = waveform.shape
        freq_bins = n_fft // 2 + 1
        num_frames = (time_len - n_fft) // hop_length + 1
        return torch.ones(batch_size, 1, freq_bins, num_frames, device=waveform.device)

    def waveform_to_stft_realimag(waveform, n_fft, hop_length):
       
        batch_size, channels, time_len = waveform.shape
        freq_bins = n_fft // 2 + 1
        num_frames = (time_len - n_fft) // hop_length + 1
        return torch.ones(batch_size, 2, freq_bins, num_frames, device=waveform.device)


warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ---------------------------
# Hyperparams (you can change)
# ---------------------------
DATA_DIR = "../data"
TRAIN_NOISE_DIR = os.path.join(DATA_DIR, "N_TR")
TRAIN_CLEAN_DIR = os.path.join(DATA_DIR, "CL_TR")
VAL_NOISE_DIR = os.path.join(DATA_DIR, "N_TS")
VAL_CLEAN_DIR = os.path.join(DATA_DIR, "CL_TS")

CHECKPOINT_DIR = "./outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 5
SR = 16000
TARGET_LEN = SR * 4  # 4 seconds
LEARNING_RATE = 1e-4
N_FFT = 512
HOP_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train_model(model_type="Conv1D", epochs=EPOCHS):
    train_dataset = SpeechEnhancementDataset(TRAIN_NOISE_DIR, TRAIN_CLEAN_DIR, target_len=TARGET_LEN, sample_rate=SR)
    val_dataset = SpeechEnhancementDataset(VAL_NOISE_DIR, VAL_CLEAN_DIR, target_len=TARGET_LEN, sample_rate=SR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    if model_type == "Conv1D":
        model = Conv1DEnhancer()
        time_domain = True
    elif model_type == "UNet":
        model = UNet1D(base_channels=64)
        time_domain = True
    elif model_type == "ResNet":
        model = ResNetEnhancer()
        time_domain = False  
    elif model_type == "DCCRN":
        model = DCCRN(n_fft=N_FFT, hop_length=HOP_LENGTH, base_channels=16)
        time_domain = False 
    else:
        raise ValueError("model_type must be one of Conv1D, UNet, ResNet, DCCRN")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training {model_type} for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        for noisy, clean in train_loop:
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()

            if model_type == "DCCRN":
       
                clean_stft = waveform_to_stft_realimag(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
                # forward
                output = model(noisy)  # [B,2,F,T]
                
                min_t = min(output.shape[-1], clean_stft.shape[-1])
                loss = criterion(output[..., :min_t], clean_stft[..., :min_t])
            elif model_type == "ResNet":
                noisy_mag = waveform_to_magphase(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)
                clean_mag = waveform_to_magphase(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
                output = model(noisy_mag)
                min_f = min(output.shape[-2], clean_mag.shape[-2])
                min_t = min(output.shape[-1], clean_mag.shape[-1])
                loss = criterion(output[..., :min_f, :min_t], clean_mag[..., :min_f, :min_t])
            else:
                # Conv1D/UNet: time-domain
                output = model(noisy)
                min_t = min(output.shape[-1], clean.shape[-1])
                loss = criterion(output[..., :min_t], clean[..., :min_t])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item()) 

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
   
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch} Val  ", leave=False)
        with torch.no_grad():
            for noisy, clean in val_loop:
                noisy = noisy.to(device)
                clean = clean.to(device)

                if model_type == "DCCRN":
                    clean_stft = waveform_to_stft_realimag(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    output = model(noisy)
                    min_t = min(output.shape[-1], clean_stft.shape[-1])
                    val_loss += criterion(output[..., :min_t], clean_stft[..., :min_t]).item()
                elif model_type == "ResNet":
                    noisy_mag = waveform_to_magphase(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    clean_mag = waveform_to_magphase(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
                    output = model(noisy_mag)
                    min_f = min(output.shape[-2], clean_mag.shape[-2])
                    min_t = min(output.shape[-1], clean_mag.shape[-1])
                    val_loss += criterion(output[..., :min_f, :min_t], clean_mag[..., :min_f, :min_t]).item()
                else:
                    output = model(noisy)
                    min_t = min(output.shape[-1], clean.shape[-1])
                    val_loss += criterion(output[..., :min_t], clean[..., :min_t]).item()
                val_loop.set_postfix(loss=val_loss / (val_loop.n + 1))

        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DCCRN", choices=["Conv1D","UNet","ResNet","DCCRN"])
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_model(model_type=args.model, epochs=args.epochs)