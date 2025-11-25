# src/utils.py
import torch
import numpy as np
import soundfile as sf

def waveform_to_magphase(waveform, n_fft=512, hop_length=256, device=None):
    """
    waveform: [B,1,T] tensor
    returns magnitude [B,1,F,T_frames] and phase complex tensor [B,F,T,2] if needed
    """
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, center=True)
    mag = torch.abs(stft).unsqueeze(1)  # [B,1,F,T]
    return mag

def waveform_to_stft_realimag(waveform, n_fft=512, hop_length=256, device=None):
    """
    returns real-imag permuted to [B, 2, F, T]
    """
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform.squeeze(1), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False, center=True)
    # [B, F, T, 2] -> permute
    stft = stft.permute(0, 3, 1, 2)  # [B, 2, F, T]
    return stft

def save_wav(waveform, path, sample_rate=48000):
    """
    waveform: torch tensor [1, T] or [T]
    """
    if isinstance(waveform, torch.Tensor):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = waveform
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    sf.write(path, arr, sample_rate)
