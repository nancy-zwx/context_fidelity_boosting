#!/bin/bash
hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=0


## run evaluate

PRED_PATH=./output/llama7b/cnndm_1_0.jsonl.output_topp0.9_genlen100_boost1.0.jsonl
GOLD_DATA_PATH=./eval/cnndm_example_input/cnndm_1_0.jsonl
python ./eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH

