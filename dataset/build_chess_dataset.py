#!/usr/bin/env python3
"""
Build Chess dataset for TinyRecursiveModels training.
Generates chess puzzle positions and solutions.
"""

import json
import numpy as np
import random
from pathlib import Path
import argparse

from common import PuzzleDatasetMetadata

# Chess piece representations
PIECES = {
    'P': 'pawn', 'N': 'knight', 'B': 'bishop', 'R': 'rook', 'Q': 'queen', 'K': 'king',
    'p': 'pawn', 'n': 'knight', 'b': 'bishop', 'r': 'rook', 'q': 'queen', 'k': 'king'
}

# Chess puzzle templates - simplified positions requiring specific moves
CHESS_PUZZLES = [
    # Checkmate in 1 puzzles
    {
        "description": "checkmate_in_1",
        "positions": [
            # Queen checkmate
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                "solution": "Qh4#",
                "description": "Black queen delivers checkmate on h4"
            },
            # Rook checkmate
            {
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
                "solution": "Rh8#",
                "description": "White rook checkmates on h8"
            },
            # Knight checkmate
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                "solution": "Nf3#",
                "description": "Black knight checkmates on f3"
            }
        ]
    },
    # Capture puzzles
    {
        "description": "capture_piece",
        "positions": [
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",
                "solution": "Bxc6",
                "description": "White bishop captures black knight on c6"
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                "solution": "Nxe4",
                "description": "Black knight captures white pawn on e4"
            }
        ]
    },
    # Defensive moves
    {
        "description": "defend_attack",
        "positions": [
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",
                "solution": "Nf3",
                "description": "White knight moves to f3 to defend against attack"
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
                "solution": "Nf6",
                "description": "Black knight moves to f6 for development and defense"
            }
        ]
    }
]

def generate_chess_puzzle(puzzle_template):
    """Generate a single chess puzzle from template."""
    position = random.choice(puzzle_template["positions"])
    return position["fen"], position["solution"], position["description"]

def create_chess_sequence(fen, solution, vocab_size=1000):
    """Convert chess puzzle to sequence format for TRM training."""
    # Combine FEN position and solution
    full_text = f"Position: {fen} Solution: {solution}"

    # Simple word-level tokenization
    import re
    words = re.findall(r'\b\w+\b', full_text.lower())

    # Build vocabulary from chess-related terms
    chess_vocab = set()
    for puzzle_type in CHESS_PUZZLES:
        for pos in puzzle_type["positions"]:
            # Add FEN pieces and coordinates
            chess_vocab.update(re.findall(r'\b\w+\b', pos["fen"].lower()))
            chess_vocab.update(re.findall(r'\b\w+\b', pos["solution"].lower()))
            chess_vocab.update(re.findall(r'\b\w+\b', pos["description"].lower()))

    # Add common chess terms
    chess_vocab.update(['position', 'solution', 'white', 'black', 'pawn', 'knight',
                       'bishop', 'rook', 'queen', 'king', 'checkmate', 'capture'])

    chess_vocab = sorted(list(chess_vocab))
    vocab = {word: i+1 for i, word in enumerate(chess_vocab)}
    id_to_word = {v: k for k, v in vocab.items()}

    # Convert words to token IDs
    tokens = [vocab.get(word, 1) for word in words]  # Default to 1 for unknown words

    # Pad or truncate to fixed length
    seq_len = 64  # Longer sequences for chess
    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))
    else:
        tokens = tokens[:seq_len]

    return tokens, len(vocab) + 1

def build_chess_dataset(num_train_puzzles=10000, num_test_puzzles=2000):
    """Build chess dataset with specified number of examples."""

    print(f"Building chess dataset with {num_train_puzzles} training and {num_test_puzzles} test examples...")

    # Generate training data
    train_inputs = []
    train_labels = []
    train_puzzle_identifiers = []
    train_puzzle_indices = []
    train_group_indices = []

    train_puzzle_indices.append(0)
    train_group_indices.append(0)

    vocab_size = 1000
    puzzle_id = 0

    for i in range(num_train_puzzles):
        # Randomly select puzzle type
        puzzle_type = random.choice(CHESS_PUZZLES)

        fen, solution, description = generate_chess_puzzle(puzzle_type)

        # Convert to sequence format
        tokens, actual_vocab_size = create_chess_sequence(fen, solution, vocab_size)
        vocab_size = max(vocab_size, actual_vocab_size)

        # Create training example (for sequence prediction)
        input_tokens = tokens[:-1]  # All but last token
        label_tokens = tokens[1:]   # All but first token, shifted

        train_inputs.append(input_tokens)
        train_labels.append(label_tokens)
        train_puzzle_identifiers.append(0)  # Single puzzle type
        puzzle_id += 1
        train_puzzle_indices.append(puzzle_id)
        train_group_indices.append(puzzle_id)

    # Generate test data (same process)
    test_inputs = []
    test_labels = []
    test_puzzle_identifiers = []
    test_puzzle_indices = []
    test_group_indices = []

    test_puzzle_indices.append(0)
    test_group_indices.append(0)

    puzzle_id = 0

    for i in range(num_test_puzzles):
        puzzle_type = random.choice(CHESS_PUZZLES)
        fen, solution, description = generate_chess_puzzle(puzzle_type)

        tokens, _ = create_chess_sequence(fen, solution, vocab_size)

        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]

        test_inputs.append(input_tokens)
        test_labels.append(label_tokens)
        test_puzzle_identifiers.append(0)
        puzzle_id += 1
        test_puzzle_indices.append(puzzle_id)
        test_group_indices.append(puzzle_id)

    # Create metadata
    metadata = {
        "vocab_size": vocab_size,
        "num_puzzle_identifiers": 1,
        "seq_len": 63,  # input length (64 - 1)
        "num_train_puzzles": num_train_puzzles,
        "num_test_puzzles": num_test_puzzles,
        "puzzle_type": "chess",
        "description": "Chess puzzle dataset for strategic reasoning"
    }

    return {
        "train": {
            "inputs": train_inputs,
            "labels": train_labels,
            "puzzle_identifiers": train_puzzle_identifiers,
            "puzzle_indices": train_puzzle_indices,
            "group_indices": train_group_indices
        },
        "test": {
            "inputs": test_inputs,
            "labels": test_labels,
            "puzzle_identifiers": test_puzzle_identifiers,
            "puzzle_indices": test_puzzle_indices,
            "group_indices": test_group_indices
        },
        "metadata": metadata
    }

def save_dataset(data, output_dir="data/chess"):
    """Save dataset to disk in the expected format."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create train and test directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Save JSON data
    with open(output_path / "train.json", "w") as f:
        json.dump(data["train"], f, indent=2)

    with open(output_path / "test.json", "w") as f:
        json.dump(data["test"], f, indent=2)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(data["metadata"], f, indent=2)

    # Convert to numpy arrays and save
    print("Converting to numpy arrays...")

    # Train data
    np.save(train_dir / "all__inputs.npy",
            np.array(data["train"]["inputs"]))
    np.save(train_dir / "all__labels.npy",
            np.array(data["train"]["labels"]))
    np.save(train_dir / "all__puzzle_identifiers.npy",
            np.array(data["train"]["puzzle_identifiers"]))
    np.save(train_dir / "all__puzzle_indices.npy",
            np.array(data["train"]["puzzle_indices"]))
    np.save(train_dir / "all__group_indices.npy",
            np.array(data["train"]["group_indices"]))

    # Test data
    np.save(test_dir / "all__inputs.npy",
            np.array(data["test"]["inputs"]))
    np.save(test_dir / "all__labels.npy",
            np.array(data["test"]["labels"]))
    np.save(test_dir / "all__puzzle_identifiers.npy",
            np.array(data["test"]["puzzle_identifiers"]))
    np.save(test_dir / "all__puzzle_indices.npy",
            np.array(data["test"]["puzzle_indices"]))
    np.save(test_dir / "all__group_indices.npy",
            np.array(data["test"]["group_indices"]))

    # Create and save dataset.json for train
    train_metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=data["metadata"]["vocab_size"],
        seq_len=data["metadata"]["seq_len"],
        num_puzzle_identifiers=1,
        total_groups=data["metadata"]["num_train_puzzles"],
        mean_puzzle_examples=1.0,
        total_puzzles=data["metadata"]["num_train_puzzles"],
        sets=["all"]
    )
    with open(train_dir / "dataset.json", "w") as f:
        json.dump(train_metadata.model_dump(), f)

    # Create and save dataset.json for test
    test_metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=data["metadata"]["vocab_size"],
        seq_len=data["metadata"]["seq_len"],
        num_puzzle_identifiers=1,
        total_groups=data["metadata"]["num_test_puzzles"],
        mean_puzzle_examples=1.0,
        total_puzzles=data["metadata"]["num_test_puzzles"],
        sets=["all"]
    )
    with open(test_dir / "dataset.json", "w") as f:
        json.dump(test_metadata.model_dump(), f)

    print(f"Dataset saved to {output_path}")
    print(f"Vocabulary size: {data['metadata']['vocab_size']}")
    print(f"Training examples: {data['metadata']['num_train_puzzles']}")
    print(f"Test examples: {data['metadata']['num_test_puzzles']}")

def main():
    parser = argparse.ArgumentParser(description="Build Chess dataset")
    parser.add_argument("--num-train-puzzles", type=int, default=10000,
                       help="Number of training examples")
    parser.add_argument("--num-test-puzzles", type=int, default=2000,
                       help="Number of test examples")
    parser.add_argument("--output-dir", type=str, default="data/chess",
                       help="Output directory")

    args = parser.parse_args()

    # Build dataset
    data = build_chess_dataset(args.num_train_puzzles, args.num_test_puzzles)

    # Save to disk
    save_dataset(data, args.output_dir)

if __name__ == "__main__":
    main()