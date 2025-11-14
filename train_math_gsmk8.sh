#!/bin/bash

# TRM Mathematical Reasoning Training Script
# Trains TRM on MATH and GSM8K style mathematical reasoning problems

echo "🚀 Starting TRM Mathematical Reasoning Training"
echo "=============================================="

# Set environment variables
export DISABLE_COMPILE=1  # Disable torch.compile to avoid compilation issues

# Change to project directory
cd /home/anto/TinyRecursiveModels

# Run training with math config
echo "📚 Training on MATH & GSM8K dataset..."
echo "💾 Checkpoints will be saved to: checkpoints/TRM-Math-Reasoning/"
echo "📊 Training progress can be monitored via wandb (if enabled)"
echo ""

# Execute training
uv run python3 pretrain.py --config-name cfg_math_pretrain

echo ""
echo "✅ Training completed!"
echo "📁 Check the checkpoints directory for saved models"
echo "🧪 Use evaluate_math.py to test mathematical reasoning capabilities"