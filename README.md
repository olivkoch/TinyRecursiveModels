# About this branch

This branch builds on top of the official TRM implementation and adds the following:
- simpler setup with `uv`
- better checkpoint saving
- simpler problems for debugging (e.g. Sudoku 4x4)
- **NEW**: Advanced Q&A datasets for reasoning evaluation
- **NEW**: Mathematical reasoning datasets (MATH & GSM8K style problems)

Nothing is changed in the model/architecture/training.

The scripts to prepare the data and train the model remain the same.

## 🚀 Quick Start Examples

### Rubik's Cube 2x2x2
```bash
# Prepare data
uv run dataset/build_rubik2x2_dataset.py

# Train model
./train_rubik2x2.sh

# Evaluate
uv run python evaluate.py --data-path data/rubik2x2/ --config checkpoints/trm/<yours>/all_config.yaml --checkpoint checkpoints/trm/<yours>/final_step_4500/model.pt
```

### Q&A Pairs (Natural Language Understanding)
```bash
# Prepare data
uv run dataset/build_qa_dataset.py

# Train model
./train_qa_pairs.sh

# Evaluate
uv run python evaluate.py --data-path data/qa_pairs/ --config checkpoints/trm/<yours>/all_config.yaml --checkpoint checkpoints/trm/<yours>/final_step_4500/model.pt
```

### Sudoku 4x4
```bash
# Prepare data
uv run python dataset/build_sudoku_4x4_dataset.py

# Train model
./train_sudoku4x4.sh

# Evaluate
uv run python evaluate.py --data-path data/sudoku4x4/ --config checkpoints/trm/<yours>/all_config.yaml --checkpoint checkpoints/trm/<yours>/final_step_45/model.pt
```

## 🧠 Advanced Reasoning Examples

### Advanced Q&A Reasoning Tasks
```bash
# Ultra-advanced reasoning Q&A pairs (76.05% accuracy achieved)
uv run dataset/build_qa_dataset.py --advanced

# Train on advanced reasoning
uv run python pretrain.py --config-name cfg_qa_advanced

# Evaluate reasoning capabilities
uv run python evaluate.py --data-path data/qa_pairs_advanced/
```

### Ultra-Complex Reasoning Tasks
```bash
# Ultra-complex multi-step reasoning problems
uv run dataset/build_qa_dataset.py --ultra-complex

# Smaller version for testing
uv run dataset/build_qa_dataset.py --ultra-complex-small
```

### Mathematical Reasoning (MATH & GSM8K)
```bash
# Prepare comprehensive math dataset (10K training, 2K test)
uv run python dataset/build_math_gsm8k_dataset.py

# Train on mathematical reasoning
./train_math&gsmk8.sh

# Evaluate math capabilities
uv run python evaluate_math.py --checkpoint-dir checkpoints/TRM-Math-Reasoning/<run>/
```

**Math Dataset Composition:**
- **Basic Arithmetic**: Addition, subtraction, multiplication, division word problems
- **Algebra**: Linear equations, systems of equations, quadratic equations
- **Geometry**: Circle area/volume, triangle area, rectangle perimeter, sphere volume
- **Calculus**: Derivatives, indefinite/definite integrals, limits, Taylor series
- **Advanced Topics**: Differential equations, complex analysis, residues
- **Statistics**: Mean, standard deviation, probability distributions
- **Number Theory**: GCD, prime checking, modular arithmetic, Euler's totient
- **Discrete Math**: Combinatorics, recurrence relations, graph theory

## 🧪 Evaluation & Analysis

### Standard Evaluation
```bash
# Evaluate any trained model
uv run python evaluate.py \
  --data-path data/<dataset>/ \
  --config checkpoints/trm/<run>/all_config.yaml \
  --checkpoint checkpoints/trm/<run>/final_step_<N>/model.pt
```

### Mathematical Reasoning Evaluation
```bash
# Evaluate math capabilities specifically
uv run python evaluate_math.py \
  --checkpoint-dir checkpoints/TRM-Math-Reasoning/<run>/ \
  --data-path data/math_gsm8k_qa \
  --num-samples 100
```

### Available Scripts
- `evaluate.py` - General evaluation for all puzzle types
- `evaluate_math.py` - Specialized evaluation for mathematical reasoning
- `train_math&gsmk8.sh` - Training script for math dataset
- `train_math_gsmk8.sh` - Alternative training script

## Reference

# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments

# NEW: Advanced Q&A Reasoning Tasks
uv run python dataset/build_qa_dataset.py --advanced          # Ultra-advanced reasoning (76.05% accuracy achieved)
uv run python dataset/build_qa_dataset.py --ultra-complex     # Ultra-complex multi-step reasoning
uv run python dataset/build_qa_dataset.py --ultra-complex-small # Smaller version for testing

# NEW: Mathematical Reasoning (MATH & GSM8K style)
uv run python dataset/build_math_gsm8k_dataset.py             # 10K training, 2K test examples across 14 math categories
```

## Experiments

### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

### Maze-Hard (assuming 4 L40S GPUs):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Runtime:* < 24 hours

### Advanced Q&A Reasoning (assuming 1 GPU):

```bash
# Ultra-advanced reasoning tasks (achieved 76.05% accuracy)
run_name="pretrain_qa_advanced"
python pretrain.py \
arch=trm \
data_paths="[data/qa_pairs_advanced]" \
evaluators="[]" \
epochs=10000 eval_interval=1000 \
lr=1e-4 puzzle_emb_lr=1e-2 weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
arch.L_layers=2 \
arch.H_cycles=2 arch.L_cycles=2 \
+run_name=${run_name}

# Ultra-complex reasoning tasks
run_name="pretrain_qa_ultra_complex"
python pretrain.py \
arch=trm \
data_paths="[data/qa_pairs_ultra_complex]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-2 weight_decay=0.1 puzzle_emb_weight_decay=0.1 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name}
```

*Runtime:* 2-12 hours

### Mathematical Reasoning (MATH & GSM8K) (assuming 1 GPU):

```bash
# Comprehensive mathematical reasoning training
run_name="pretrain_math_gsm8k"
python pretrain.py --config-name cfg_math_pretrain

# Quick test version (10 epochs)
python pretrain.py --config-name cfg_math_test
```

*Runtime:* 4-120 hours (depending on configuration)

**Math Dataset Composition:**
- **Basic Arithmetic**: Addition, subtraction, multiplication, division word problems
- **Algebra**: Linear equations, systems of equations, quadratic equations
- **Geometry**: Circle area/volume, triangle area, rectangle perimeter, sphere volume
- **Calculus**: Derivatives, indefinite/definite integrals, limits, Taylor series
- **Advanced Topics**: Differential equations, complex analysis, residues
- **Statistics**: Mean, standard deviation, probability distributions
- **Number Theory**: GCD, prime checking, modular arithmetic, Euler's totient
- **Discrete Math**: Combinatorics, recurrence relations, graph theory

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
