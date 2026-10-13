#!/bin/bash

# models=("bert" "legalbert" "inlegalbert")
models=("llama" "qwen" "mistral" "saul")
datasets=("SCIJuDAC")
# Change any parameters' value as needed.
# Sample arrays for other parameters. Uncomment and insert as iteration to use:
# batch_size=(4 8 16 32)
# epochs=(3 4 5)
# seed=(42 4242 2121)
# collapse_labels_flag=("n" "y")
# collapse_classifier=("n" "y")
# load_ckpt_flag=("n" "y")

for model in "${models[@]}"; do
  python main.py \
  --name "exp" \
  --model "$model" \
  --dataset "SCIJuDAC" \
  --batch_size 16 \
  --epochs 3 \
  --seed 42 \
  --collapse_labels "y" \
  --collapse_classifier "n" \
  --load_ckpt "n"
done
