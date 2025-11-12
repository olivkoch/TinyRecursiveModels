#!/usr/bin/env python3
"""
Build Q&A dataset for TinyRecursiveModels training.
Generates question-answer pairs using natural language sentences.
"""

import json
import numpy as np
import random
from pathlib import Path
import argparse

from common import PuzzleDatasetMetadata

# Q&A templates for generating diverse data
QA_TEMPLATES = [
    # Factual questions
    {
        "questions": [
            "What is the capital of {country}?",
            "What is {country}'s capital city?",
            "Which city is the capital of {country}?",
        ],
        "answers": ["{capital}"],
        "data": [
            {"country": "France", "capital": "Paris"},
            {"country": "Germany", "capital": "Berlin"},
            {"country": "Italy", "capital": "Rome"},
            {"country": "Spain", "capital": "Madrid"},
            {"country": "United Kingdom", "capital": "London"},
            {"country": "Japan", "capital": "Tokyo"},
            {"country": "China", "capital": "Beijing"},
            {"country": "India", "capital": "New Delhi"},
            {"country": "Brazil", "capital": "Brasília"},
            {"country": "Canada", "capital": "Ottawa"},
        ]
    },
    # Mathematical questions
    {
        "questions": [
            "What is {num1} plus {num2}?",
            "What is the sum of {num1} and {num2}?",
            "If you add {num1} and {num2}, what do you get?",
        ],
        "answers": ["{result}"],
        "data": [
            {"num1": "5", "num2": "3", "result": "8"},
            {"num1": "10", "num2": "7", "result": "17"},
            {"num1": "12", "num2": "8", "result": "20"},
            {"num1": "15", "num2": "9", "result": "24"},
            {"num1": "20", "num2": "5", "result": "25"},
        ]
    },
    # Color questions
    {
        "questions": [
            "What color is a {fruit}?",
            "What is the color of a {fruit}?",
            "Which color does a {fruit} have?",
        ],
        "answers": ["{color}"],
        "data": [
            {"fruit": "banana", "color": "yellow"},
            {"fruit": "apple", "color": "red"},
            {"fruit": "orange", "color": "orange"},
            {"fruit": "grape", "color": "purple"},
            {"fruit": "lemon", "color": "yellow"},
        ]
    },
    # Animal questions
    {
        "questions": [
            "What sound does a {animal} make?",
            "What noise does a {animal} make?",
            "How does a {animal} sound?",
        ],
        "answers": ["{sound}"],
        "data": [
            {"animal": "dog", "sound": "woof"},
            {"animal": "cat", "sound": "meow"},
            {"animal": "cow", "sound": "moo"},
            {"animal": "sheep", "sound": "baa"},
            {"animal": "duck", "sound": "quack"},
        ]
    },
    # Time questions
    {
        "questions": [
            "What day comes after {day}?",
            "What is the day after {day}?",
            "Which day follows {day}?",
        ],
        "answers": ["{next_day}"],
        "data": [
            {"day": "Monday", "next_day": "Tuesday"},
            {"day": "Tuesday", "next_day": "Wednesday"},
            {"day": "Wednesday", "next_day": "Thursday"},
            {"day": "Thursday", "next_day": "Friday"},
            {"day": "Friday", "next_day": "Saturday"},
            {"day": "Saturday", "next_day": "Sunday"},
            {"day": "Sunday", "next_day": "Monday"},
        ]
    },
    # Weather questions
    {
        "questions": [
            "What is the weather like when it {condition}?",
            "What kind of weather is {condition}?",
            "When it {condition}, what is the weather?",
        ],
        "answers": ["{weather}"],
        "data": [
            {"condition": "rains", "weather": "rainy"},
            {"condition": "snows", "weather": "snowy"},
            {"condition": "is sunny", "weather": "sunny"},
            {"condition": "is cloudy", "weather": "cloudy"},
            {"condition": "is windy", "weather": "windy"},
        ]
    }
]

def generate_qa_pair(template, data_item):
    """Generate a single Q&A pair from template and data."""
    question_template = random.choice(template["questions"])
    answer_template = random.choice(template["answers"])

    # Format question and answer
    question = question_template.format(**data_item)
    answer = answer_template.format(**data_item)

    return question, answer

def create_qa_sequence(question, answer, vocab_size=1000):
    """Convert Q&A pair to sequence format for TRM training."""
    # Simple tokenization - split into words and map to token IDs
    # In practice, you'd want a proper tokenizer, but this works for demo

    # Combine question and answer with special tokens
    full_text = f"Question: {question} Answer: {answer}"

    # Simple word-level tokenization (lowercase, basic punctuation)
    import re
    words = re.findall(r'\b\w+\b', full_text.lower())

    # Create a simple vocabulary mapping (in practice, use a real tokenizer)
    vocab = {}
    token_id = 1  # Start from 1, reserve 0 for padding

    # Build vocabulary from all possible words in our templates
    all_words = set()
    for template in QA_TEMPLATES:
        for data_item in template["data"]:
            for q_template in template["questions"]:
                q = q_template.format(**data_item)
                all_words.update(re.findall(r'\b\w+\b', q.lower()))
            for a_template in template["answers"]:
                a = a_template.format(**data_item)
                all_words.update(re.findall(r'\b\w+\b', a.lower()))

    # Sort for consistent ordering
    all_words = sorted(list(all_words))
    for word in all_words:
        vocab[word] = token_id
        token_id += 1

    # Convert words to token IDs
    tokens = [vocab.get(word, 1) for word in words]  # Default to 1 for unknown words

    # Pad or truncate to fixed length (adjust as needed)
    seq_len = 32
    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))
    else:
        tokens = tokens[:seq_len]

    return tokens, len(vocab) + 1  # +1 for padding token

def build_qa_dataset(num_train_puzzles=10000, num_test_puzzles=2000):
    """Build Q&A dataset with specified number of examples."""

    print(f"Building Q&A dataset with {num_train_puzzles} training and {num_test_puzzles} test examples...")

    # Generate training data
    train_data = []
    train_inputs = []
    train_labels = []
    train_puzzle_identifiers = []
    train_puzzle_indices = []
    train_group_indices = []

    train_puzzle_indices.append(0)  # Start with 0
    train_group_indices.append(0)   # Start with 0

    puzzle_id = 0

    puzzle_id = 0
    vocab_size = 1000  # Will be updated based on actual vocabulary

    for i in range(num_train_puzzles):
        # Select random template and data item
        template = random.choice(QA_TEMPLATES)
        data_item = random.choice(template["data"])

        # Generate Q&A pair
        question, answer = generate_qa_pair(template, data_item)

        # Convert to sequence format
        tokens, actual_vocab_size = create_qa_sequence(question, answer, vocab_size)
        vocab_size = max(vocab_size, actual_vocab_size)

        # Create training example (for sequence prediction)
        # Input: question tokens, Label: answer tokens shifted by 1
        input_tokens = tokens[:-1]  # All but last token
        label_tokens = tokens[1:]   # All but first token, shifted

        train_inputs.append(input_tokens)
        train_labels.append(label_tokens)
        train_puzzle_identifiers.append(0)  # Single puzzle type
        puzzle_id += 1
        train_puzzle_indices.append(puzzle_id)
        train_group_indices.append(puzzle_id)

    # Generate test data (same process)
    test_data = []
    test_inputs = []
    test_labels = []
    test_puzzle_identifiers = []
    test_puzzle_indices = []
    test_group_indices = []

    test_puzzle_indices.append(0)  # Start with 0
    test_group_indices.append(0)   # Start with 0

    puzzle_id = 0  # Reset for test

    for i in range(num_test_puzzles):
        template = random.choice(QA_TEMPLATES)
        data_item = random.choice(template["data"])

        question, answer = generate_qa_pair(template, data_item)
        tokens, _ = create_qa_sequence(question, answer, vocab_size)

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
        "seq_len": 31,  # input length (32 - 1)
        "num_train_puzzles": num_train_puzzles,
        "num_test_puzzles": num_test_puzzles,
        "puzzle_type": "qa_pairs",
        "description": "Question-Answer pairs dataset for language modeling"
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

def save_dataset(data, output_dir="data/qa_pairs"):
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
    parser = argparse.ArgumentParser(description="Build Q&A dataset")
    parser.add_argument("--num-train-puzzles", type=int, default=10000,
                       help="Number of training examples")
    parser.add_argument("--num-test-puzzles", type=int, default=2000,
                       help="Number of test examples")
    parser.add_argument("--output-dir", type=str, default="data/qa_pairs",
                       help="Output directory")

    args = parser.parse_args()

    # Build dataset
    data = build_qa_dataset(args.num_train_puzzles, args.num_test_puzzles)

    # Save to disk
    save_dataset(data, args.output_dir)

if __name__ == "__main__":
    main()