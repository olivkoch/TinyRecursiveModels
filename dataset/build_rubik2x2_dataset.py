"""
build_rubik2x2_dataset.py - Generate 2x2x2 Rubik's cube dataset
"""

import json
import numpy as np
import os
import random
from typing import List, Tuple, Dict
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class Rubik2x2DataProcessConfig(BaseModel):
    output_dir: str = "data/rubik2x2"
    num_train_puzzles: int = 2000
    num_test_puzzles: int = 400
    seed: int = 42
    max_scramble_moves: int = 10  # Maximum moves to scramble
    min_scramble_moves: int = 3   # Minimum moves to scramble


# 2x2x2 Rubik's cube representation
# We'll represent the cube as 8 corner pieces, each with 3 colors
# Colors: 0=White, 1=Yellow, 2=Red, 3=Orange, 4=Blue, 5=Green

# Standard solved cube state (piece positions and orientations)
SOLVED_CUBE = np.array([
    # Corner pieces: [U, F, R] colors for each corner
    [0, 5, 2],  # UFL: White, Green, Red
    [0, 2, 4],  # UFR: White, Red, Blue
    [0, 4, 3],  # UBR: White, Blue, Orange
    [0, 3, 5],  # UBL: White, Orange, Green
    [5, 1, 2],  # DFL: Green, Yellow, Red
    [2, 1, 4],  # DFR: Red, Yellow, Blue
    [4, 1, 3],  # DBR: Blue, Yellow, Orange
    [3, 1, 5],  # DBL: Orange, Yellow, Green
], dtype=np.int32)

# Move definitions - each move cycles 4 corner pieces
MOVES = {
    'R': [1, 5, 6, 2],  # Right face: UFR, DFR, DBR, UBR
    'L': [0, 3, 7, 4],  # Left face: UFL, UBL, DBL, DFL
    'U': [0, 1, 2, 3],  # Up face: UFL, UFR, UBR, UBL
    'D': [4, 7, 6, 5],  # Down face: DFL, DBL, DBR, DFR
    'F': [0, 4, 5, 1],  # Front face: UFL, DFL, DFR, UFR
    'B': [3, 2, 6, 7],  # Back face: UBL, UBR, DBR, DBL
}

# Move names for scrambling
MOVE_NAMES = ['R', 'L', 'U', 'D', 'F', 'B']
MODIFIERS = ['', "'", '2']  # clockwise, counter-clockwise, 180 degrees


def apply_move(cube: np.ndarray, move_name: str) -> np.ndarray:
    """Apply a move to the cube state."""
    new_cube = cube.copy()

    # Parse move (e.g., "R'" -> move='R', modifier="'")
    if move_name.endswith("'"):
        move = move_name[:-1]
        modifier = "'"
    elif move_name.endswith('2'):
        move = move_name[:-1]
        modifier = '2'
    else:
        move = move_name
        modifier = ''

    if move not in MOVES:
        raise ValueError(f"Unknown move: {move}")

    corners = MOVES[move]

    if modifier == '':  # clockwise
        # Cycle corners: 0->1->2->3->0
        temp = new_cube[corners[0]].copy()
        new_cube[corners[0]] = new_cube[corners[3]]
        new_cube[corners[3]] = new_cube[corners[2]]
        new_cube[corners[2]] = new_cube[corners[1]]
        new_cube[corners[1]] = temp
    elif modifier == "'":  # counter-clockwise
        # Cycle corners: 0->3->2->1->0
        temp = new_cube[corners[0]].copy()
        new_cube[corners[0]] = new_cube[corners[1]]
        new_cube[corners[1]] = new_cube[corners[2]]
        new_cube[corners[2]] = new_cube[corners[3]]
        new_cube[corners[3]] = temp
    elif modifier == '2':  # 180 degrees
        # Swap pairs: 0<->2, 1<->3
        new_cube[corners[0]], new_cube[corners[2]] = new_cube[corners[2]].copy(), new_cube[corners[0]].copy()
        new_cube[corners[1]], new_cube[corners[3]] = new_cube[corners[3]].copy(), new_cube[corners[1]].copy()

    return new_cube


def scramble_cube(cube: np.ndarray, num_moves: int) -> Tuple[np.ndarray, List[str]]:
    """Scramble the cube with random moves. Returns (scrambled_cube, move_sequence)."""
    scrambled = cube.copy()
    moves = []

    for _ in range(num_moves):
        move = random.choice(MOVE_NAMES)
        modifier = random.choice(MODIFIERS)
        move_name = move + modifier

        scrambled = apply_move(scrambled, move_name)
        moves.append(move_name)

    return scrambled, moves


def cube_to_sequence(cube: np.ndarray, max_seq_len: int = 32) -> np.ndarray:
    """Convert cube state to a sequence for the model."""
    # Flatten the cube representation
    # Each corner has 3 colors, so 8 corners * 3 = 24 values
    flat_cube = cube.flatten()

    # Add padding to reach max_seq_len
    padding_needed = max_seq_len - len(flat_cube)
    if padding_needed > 0:
        # Pad with a special padding token (we'll use 6 for padding)
        padded = np.concatenate([flat_cube, np.full(padding_needed, 6)])
    else:
        padded = flat_cube[:max_seq_len]

    return padded


def moves_to_sequence(moves: List[str], vocab_offset: int = 7, max_seq_len: int = 32) -> np.ndarray:
    """Convert move sequence to a sequence for the model."""
    # Map moves to vocabulary indices
    move_vocab = {}
    idx = vocab_offset
    for move in MOVE_NAMES:
        for mod in MODIFIERS:
            move_name = move + mod
            move_vocab[move_name] = idx
            idx += 1

    # Convert moves to indices
    move_indices = [move_vocab[move] for move in moves]

    # Pad or truncate to max_seq_len
    if len(move_indices) < max_seq_len:
        # Pad with end token (we'll use idx for end)
        padding = [idx] * (max_seq_len - len(move_indices))
        sequence = move_indices + padding
    else:
        sequence = move_indices[:max_seq_len]

    return np.array(sequence)


def generate_dataset(config: Rubik2x2DataProcessConfig):
    """Generate the Rubik's cube dataset."""
    random.seed(config.seed)
    np.random.seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # Generate training data
    print("Generating training puzzles...")
    train_puzzles = []
    train_solutions = []

    for i in tqdm(range(config.num_train_puzzles)):
        # Random scramble length
        num_moves = random.randint(config.min_scramble_moves, config.max_scramble_moves)

        # Generate scrambled cube
        scrambled_cube, scramble_moves = scramble_cube(SOLVED_CUBE, num_moves)

        # For now, we'll use the scramble moves as the "solution"
        # In a full implementation, we'd solve the cube to get the optimal solution
        solution_moves = scramble_moves[::-1]  # Reverse moves (simplified)

        # Convert to sequences
        puzzle_seq = cube_to_sequence(scrambled_cube)
        solution_seq = moves_to_sequence(solution_moves)

        train_puzzles.append(puzzle_seq)
        train_solutions.append(solution_seq)

    # Generate test data
    print("Generating test puzzles...")
    test_puzzles = []
    test_solutions = []

    for i in tqdm(range(config.num_test_puzzles)):
        # Random scramble length
        num_moves = random.randint(config.min_scramble_moves, config.max_scramble_moves)

        # Generate scrambled cube
        scrambled_cube, scramble_moves = scramble_cube(SOLVED_CUBE, num_moves)

        # Reverse moves for solution
        solution_moves = scramble_moves[::-1]

        # Convert to sequences
        puzzle_seq = cube_to_sequence(scrambled_cube)
        solution_seq = moves_to_sequence(solution_moves)

        test_puzzles.append(puzzle_seq)
        test_solutions.append(solution_seq)

    # Create dataset structure similar to sudoku
    vocab_size = 7 + 18 + 1  # 6 colors + padding + 18 moves (6 moves * 3 modifiers) + end token
    seq_len = 32

    # Training set
    train_data = {
        "inputs": [puzzle.tolist() for puzzle in train_puzzles],
        "labels": [solution.tolist() for solution in train_solutions],
        "puzzle_identifiers": [0] * len(train_puzzles),  # Single puzzle type
        "puzzle_indices": list(range(len(train_puzzles))),
        "group_indices": list(range(len(train_puzzles))),
    }

    # Test set
    test_data = {
        "inputs": [puzzle.tolist() for puzzle in test_puzzles],
        "labels": [solution.tolist() for solution in test_solutions],
        "puzzle_identifiers": [0] * len(test_puzzles),
        "puzzle_indices": list(range(len(test_puzzles))),
        "group_indices": list(range(len(test_puzzles))),
    }

    # Save datasets
    with open(os.path.join(config.output_dir, "train.json"), "w") as f:
        json.dump(train_data, f)

    with open(os.path.join(config.output_dir, "test.json"), "w") as f:
        json.dump(test_data, f)

    # Create metadata
    metadata = PuzzleDatasetMetadata(
        pad_id=6,  # Padding token
        ignore_label_id=vocab_size - 1,  # End token for labels
        blank_identifier_id=0,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=1,
        total_groups=config.num_train_puzzles + config.num_test_puzzles,
        mean_puzzle_examples=1.0,
        total_puzzles=config.num_train_puzzles + config.num_test_puzzles,
        sets=["train", "test"]
    )

    with open(os.path.join(config.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    print(f"Dataset generated in {config.output_dir}")
    print(f"Training puzzles: {config.num_train_puzzles}")
    print(f"Test puzzles: {config.num_test_puzzles}")
    print(f"Vocab size: {vocab_size}")
    print(f"Sequence length: {seq_len}")


@cli.command(singleton=True)
def main(config: Rubik2x2DataProcessConfig):
    generate_dataset(config)


if __name__ == "__main__":
    cli()