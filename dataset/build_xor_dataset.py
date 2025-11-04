from typing import List, Optional
import json
import numpy as np
import os
from argdantic import ArgParser
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata

cli = ArgParser()

class XORDataProcessConfig(BaseModel):
    output_dir: str = "data/xor-simple"
    grid_size: int = 3  # 3x3 grids
    max_grid_size: int = 6  # Pad to 6x6 instead of 30x30!
    num_train_samples: int = 1000
    num_test_samples: int = 200
    seed: int = 42
    num_examples_per_puzzle: int = 3

def generate_simple_xor_examples(grid_size: int, num_examples: int):
    """Generate XOR examples.
    Input: 3x3 grid with values 0-3 representing bit pairs
    Output: 3x3 grid with XOR results (0 or 1)
    """
    examples = []
    
    for _ in range(num_examples):
        # Generate input with all 4 possible combinations
        input_grid = np.random.randint(0, 4, (grid_size, grid_size), dtype=np.uint8)
        
        # Compute XOR for each cell
        output_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        output_grid[input_grid == 1] = 1  # (1,0) -> 1
        output_grid[input_grid == 2] = 1  # (0,1) -> 1
        # (0,0) and (1,1) remain 0
        
        examples.append((input_grid, output_grid))
    
    return examples

def pad_and_flatten(inp: np.ndarray, out: np.ndarray, max_size: int):
    """Pad grids to max_size and flatten them."""
    # Add 2 to all values to reserve 0 for PAD and 1 for EOS
    inp = inp + 2
    out = out + 2
    
    # Pad to max_size x max_size
    inp_h, inp_w = inp.shape
    out_h, out_w = out.shape
    
    inp_padded = np.pad(inp, 
                        ((0, max_size - inp_h), (0, max_size - inp_w)), 
                        constant_values=0)
    out_padded = np.pad(out, 
                        ((0, max_size - out_h), (0, max_size - out_w)), 
                        constant_values=0)
    
    # Add simple EOS marker at the actual data boundary
    # if inp_h < max_size and inp_w < max_size:
    #     inp_padded[inp_h, inp_w] = 1  # Single EOS marker
    # if out_h < max_size and out_w < max_size:
    #     out_padded[out_h, out_w] = 1  # Single EOS marker
    if inp_h < max_size:
        inp_padded[inp_h, 0] = 1  # Position 18 for 3x3 data
    if out_h < max_size:
        out_padded[out_h, 0] = 1  # Position 18 for 3x3 data
        
    return inp_padded.flatten(), out_padded.flatten()

def generate_dataset(config: XORDataProcessConfig, split: str):
    """Generate XOR dataset for a given split."""
    np.random.seed(config.seed + (0 if split == "train" else 1000))
    
    num_puzzles = config.num_train_samples if split == "train" else config.num_test_samples
    
    # Generate dataset
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0]
    }
    
    example_id = 0
    puzzle_id = 0
    
    for puzzle_idx in range(num_puzzles):
        examples = generate_simple_xor_examples(config.grid_size, config.num_examples_per_puzzle)
        
        for inp, out in examples:
            inp_flat, out_flat = pad_and_flatten(inp, out, config.max_grid_size)
            
            results["inputs"].append(inp_flat)
            results["labels"].append(out_flat)
            example_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(1)  # All XOR puzzles have ID 1
        puzzle_id += 1
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy arrays
    results_np = {
        "inputs": np.stack(results["inputs"], axis=0),
        "labels": np.stack(results["labels"], axis=0),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32)
    }
    
    # Create metadata with CORRECT sequence length
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_grid_size * config.max_grid_size,  # 6x6 = 36, not 900!
        vocab_size=7,  # PAD(0) + EOS(1) + actual values (2-6)
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=2,
        total_groups=num_puzzles,
        mean_puzzle_examples=float(config.num_examples_per_puzzle),
        total_puzzles=num_puzzles,
        sets=["all"]
    )
    
    # Save dataset
    save_dir = os.path.join(config.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    
    for key, value in results_np.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    print(f"Generated {split} split: {num_puzzles} puzzles")
    print(f"  Sequence length: {metadata.seq_len} (was 900, now 36!)")
    print(f"  Input shape per example: {config.grid_size}x{config.grid_size} padded to {config.max_grid_size}x{config.max_grid_size}")

@cli.command(singleton=True)
def main(config: XORDataProcessConfig):
    """Generate XOR dataset."""
    np.random.seed(config.seed)
    
    generate_dataset(config, "train")
    generate_dataset(config, "test")
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "xor"], f)
    
    print("\nDataset generated with sequence length 36 instead of 900!")
    print("This should significantly reduce model complexity and memory usage.")

if __name__ == "__main__":
    cli()