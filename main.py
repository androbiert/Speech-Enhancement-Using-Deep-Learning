import argparse
import os
import torch
import sys
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import train_model
from src.evaluate import evaluate_model

# --- Data and Model Directories ---
DATA_DIR = "../data"
TRAIN_NOISE_DIR = os.path.join(DATA_DIR, "N_TR")
TRAIN_CLEAN_DIR = os.path.join(DATA_DIR, "CL_TR")
VAL_NOISE_DIR = os.path.join(DATA_DIR, "N_TS")
VAL_CLEAN_DIR = os.path.join(DATA_DIR, "CL_TS")

CHECKPOINT_DIR = "../outputs/checkpoints"
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "DCCRN_epoch5.pth")
OUTPUT_DIR = "../outputs/eval"


def main():
    parser = argparse.ArgumentParser(description=" Voice Filtering Project Main App")

    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True,
                        help="Select whether to train or evaluate the model")

    parser.add_argument("--model", type=str, default="DCCRN",
                        help="Model to use (DCCRN, ResNetEnhancer, etc.)")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device (cuda or cpu)")

    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint for evaluation")

    args = parser.parse_args()

    print(f"\n Mode: {args.mode}")
    print(f" Model: {args.model}")
    print(f" Device: {args.device}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Data Dir: {DATA_DIR}")
    print("-" * 50)

    # ---- Training Mode ----
    if args.mode == "train":
        train_model(
            model_type=args.model,
            noisy_dir=TRAIN_NOISE_DIR,
            clean_dir=TRAIN_CLEAN_DIR,
            val_noisy_dir=VAL_NOISE_DIR,
            val_clean_dir=VAL_CLEAN_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )

    # ---- Evaluation Mode ----
    elif args.mode == "eval":
        evaluate_model(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            noisy_dir=VAL_NOISE_DIR,
            clean_dir=VAL_CLEAN_DIR,
            device=args.device,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    main()
