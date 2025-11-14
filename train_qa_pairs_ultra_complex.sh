#!/bin/sh

export PYTORCH_DISABLE_COMPILE=1
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0
uv run python pretrain.py \
    arch=trm \
    data_paths="[data/qa_pairs_ultra_complex]" \
    arch.halt_exploration_prob=0.0 \
    arch.halt_max_steps=8 \
    arch.H_cycles=2 \
    arch.L_cycles=2 \
    arch.H_layers=0 \
    arch.L_layers=1 \
    arch.hidden_size=128 \
    arch.num_heads=4 \
    arch.expansion=2 \
    arch.puzzle_emb_ndim=8 \
    arch.forward_dtype=float32 \
    arch.puzzle_emb_len=8 \
    global_batch_size=8 \
    epochs=10000 \
    lr=0.001 \
    puzzle_emb_lr=0.01 \
    weight_decay=0.0 \
    puzzle_emb_weight_decay=0.0 \
    lr_warmup_steps=1000 \
    eval_interval=10 \
    use_wandb=false \
    beta1=0.9 \
    beta2=0.999 \
    +project_name="qa_pairs_ultra_complex"