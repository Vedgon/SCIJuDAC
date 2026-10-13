#!/bin/bash

bert_models=("bert" "legalbert" "inlegalbert")
llms=("llama" "qwen")
datasets=("ILDC_multi" "SCIJuDAC")

# for model in "${bert_models[@]}"; do
#   python run_analysis.py \
#   --name "sep_cal_lc" \
#   --model "$model" \
#   --dataset "SCIJuDAC" \
#   --load_lc_ckpt "y"
# done

# for model in "${bert_models[@]}"; do
#   python run_analysis.py \
#   --name "sep_cal_cc" \
#   --model "$model" \
#   --dataset "SCIJuDAC" \
#   --load_cc_ckpt "y"
# done

for model in "${llms[@]}"; do
  python run_analysis.py \
  --name "sep_cal_lc" \
  --model "$model" \
  --dataset "SCIJuDAC" \
  --load_lc_ckpt "y"
done