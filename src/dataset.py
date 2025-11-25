import os
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import soundfile as sf
import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import soundfile as sf

class SpeechEnhancementDataset(Dataset):
    """
    Returns fixed-length waveforms (mono) as tensors with shape [1, target_len].
    If file shorter than target_len, pads with zeros; if longer, truncates.
    """
    def __init__(self, noisy_dir, clean_dir, target_len=None, sample_rate=48000, ext_whitelist=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.target_len = target_len
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if (ext_whitelist is None or os.path.splitext(f)[1] in ext_whitelist)])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if (ext_whitelist is None or os.path.splitext(f)[1] in ext_whitelist)])
        assert len(self.noisy_files) == len(self.clean_files), f"Mismatch noisy/clean counts: {len(self.noisy_files)} vs {len(self.clean_files)}"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy, sr1 = sf.read(noisy_path)
        clean, sr2 = sf.read(clean_path)

        if sr1 != self.sample_rate:
            # Fix 2: Use keyword arguments for librosa.resample
            noisy = librosa.resample(y=noisy.astype('float32'), orig_sr=sr1, target_sr=self.sample_rate)
        if sr2 != self.sample_rate:
            # Fix 2: Use keyword arguments for librosa.resample
            clean = librosa.resample(y=clean.astype('float32'), orig_sr=sr2, target_sr=self.sample_rate)

        # mono
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        # convert to float32
        noisy = noisy.astype('float32')
        clean = clean.astype('float32')

        if self.target_len is not None:
            noisy = self._fix_length(noisy, self.target_len)
            clean = self._fix_length(clean, self.target_len)

        # shape -> [1, T]
        noisy = torch.from_numpy(noisy).float().unsqueeze(0)
        clean = torch.from_numpy(clean).float().unsqueeze(0)

        return noisy, clean

    # Fix 1: Indent _fix_length to be a method of the class
    def _fix_length(self, wav, target_len):
        cur = len(wav)
        if cur < target_len:
            pad = target_len - cur
            wav = np.pad(wav, (0, pad), mode='constant', constant_values=0)
        else:
            wav = wav[:target_len]
        return wav