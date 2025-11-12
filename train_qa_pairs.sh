#!/bin/sh

DISABLE_COMPILE=1 uv run python pretrain.py \
    arch=trm \
    data_paths="[data/qa_pairs]" \
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
    global_batch_size=256 \
    epochs=10000 \
    lr=0.001 \
    puzzle_emb_lr=0.01 \
    weight_decay=0.0 \
    puzzle_emb_weight_decay=0.0 \
    lr_warmup_steps=1000 \
    eval_interval=10 \
    use_wandb=false \
    +project_name="qa_pairs_baseline"