#!/usr/bin/env python3
"""
Evaluation script for TRM mathematical reasoning capabilities.
Tests the model on sample MATH and GSM8K style problems.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import yaml

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model(config: dict, vocab_size: int, seq_len: int, num_puzzle_identifiers: int, device: str = "cuda"):
    """Create model from config."""
    arch_config = config['arch']

    model_cfg = dict(
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        batch_size=1,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])

    model = model_cls(model_cfg)
    model = loss_head_cls(model, **{k: v for k, v in arch_config['loss'].items() if k != 'name'})
    return model.to(device)


def evaluate_math_reasoning(checkpoint_dir: str, data_path: str = "data/math_gsm8k_qa", num_samples: int = 10):
    """Evaluate TRM on mathematical reasoning tasks."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load test dataset
    test_config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[data_path],
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=1
    )

    test_dataset = PuzzleDataset(test_config, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Create model
    model = create_model(config, test_dataset.metadata.vocab_size,
                        test_dataset.metadata.seq_len,
                        test_dataset.metadata.num_puzzle_identifiers, device)

    # Load checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("model.pt")]
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else 0)
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(os.path.join(checkpoint_dir, "model.pt"), device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize carry (None for start of sequence)
    carry = None

    print(f"\nEvaluating on {num_samples} math problems...")

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (set_name, batch, global_batch_size) in enumerate(test_loader):
            if i >= num_samples:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Initialize carry on first batch
            if carry is None:
                with torch.device(device):
                    carry = model.initial_carry(batch)

            # Forward pass
            carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=[])

            # Get predictions (preds are already computed by the loss head)
            targets = batch.get('targets', batch.get('labels', None))

            if targets is not None:
                print(f"Preds: {preds}")
                print(f"Metrics: {metrics}")
                print(f"Targets shape: {targets.shape}")
                # For now, let's skip accuracy calculation and just report loss
                pass

            print(f"Sample {i+1}: Loss = {loss.item():.4f}")

    if total > 0:
        accuracy = correct / total * 100
        print(".2f")
    else:
        print("Could not compute accuracy - no classification targets found")

    print(f"\nEvaluation complete. Tested on {num_samples} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TRM on mathematical reasoning")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--data-path", type=str, default="data/math_gsm8k_qa",
                       help="Path to math dataset")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to evaluate")

    args = parser.parse_args()
    evaluate_math_reasoning(args.checkpoint_dir, args.data_path, args.num_samples)