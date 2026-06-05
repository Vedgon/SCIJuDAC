#!/bin/bash

name=("last_toks" "all_toks")
models=("bert" "legalbert" "inlegalbert" "llama" "qwen" "mistral" "saul")
datasets=("ILDC_multi" "SCIJuDAC")
all_toks_flag=("n" "y")
# Change any parameters' value as needed.
# Sample arrays for other parameters. Uncomment and insert as iteration to use:
# batch_size=(4 8 16 32)
# epochs=(3 4 5)
# seed=(42 4242 2121)
# collapse_labels_flag=("n" "y")
# load_ckpt_flag=("n" "y")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for flag in "${all_toks[@]}"; do
      if [[ "$flag" == "y" ]]; then
        name="all_toks"
      else
        name="last_toks"
      fi

      python main.py \
      --name "$name" \
      --model "$model" \
      --dataset "$dataset" \
      --batch_size 16 \
      --epochs 3 \
      --seed 42 \
      --all_toks "$flag" \
      --collapse_labels "n" \
      --load_ckpt "n"
    done
  done
done
