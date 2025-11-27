# 🗣️ Speech Enhancement Using Deep Learning
### Clean and Noisy Parallel Speech Dataset — Valentini-Botinhao

---

## 📘 Project Overview

This project aims to **enhance noisy speech signals** using deep learning models.  
It is based on the **Valentini-Botinhao Noisy Speech Dataset**, which contains parallel pairs of clean and noisy audio files recorded from multiple speakers at 48 kHz.  

Due to the large size of the dataset (over **71,000 audio files** and **21 GB** of data), a **subset** was extracted for faster experimentation:  
- **Training set:** 100 clean + 100 noisy files  
- **Test set:** 10 clean + 10 noisy files  

The main goal is to build, train, and evaluate a **speech enhancement model** capable of learning to remove background noise from human speech recordings.

---

## 📊 Dataset Information

**Source:** [Kaggle – Valentini Noisy Speech Dataset](https://www.kaggle.com/datasets/muhmagdy/valentini-noisy)

**Full Dataset Size:**  
- 71,000+ files  
- ~21 GB total  
- Sampling rate: 48 kHz  
- Speakers: 28 and 56 (two configurations)

**Data Composition:**
- `clean_trainset_28spk_wav` – Clean training speech  
- `noisy_trainset_28spk_wav` – Corresponding noisy training data  
- `clean_testset_wav` – Clean test data  
- `noisy_testset_wav` – Noisy test data  

Each noisy file corresponds **exactly** to a clean version (e.g., `p1_1.wav` clean ↔ `p1_1.wav` noisy).

**Noises Used:**
- Speech-shaped noise  
- Babble noise  
- Environmental noises from the **DEMAND database**  
- Additional details in:
  - Valentini-Botinhao et al., *Interspeech 2016*  
  - Valentini-Botinhao et al., *SSW 2016*  

---


---

## 🧠 Model Description

Two model architectures were tested:
1. **Simple CNN Autoencoder** – Baseline model for denoising.  
2. **ResNet-based Network** – Transfer learning model adapted for audio enhancement.
3. **DCCRN (Deep Complex Convolutional Recurrent Network)**

· Combines complex-valued convolutions with LSTM layers

· Operates in the time-frequency domain, learning both magnitude and phase

· State-of-the-art for speech enhancement tasks

Both models learn to map a **noisy waveform** to its corresponding **clean waveform** using MSE loss.

---

## 🧩 Requirements

```bash
pip install torch torchvision torchaudio
pip install librosa numpy matplotlib tqdm
``` 
## 🚀 How to Run

### Prepare the data:

``` bash
python scripts/prepare_data.py
```

### Train the model:

``` bash
python scripts/train_model.py
```

### Evaluate the model:
``` bash
python scripts/evaluate_model.py
``` 


# THANKYOU !
