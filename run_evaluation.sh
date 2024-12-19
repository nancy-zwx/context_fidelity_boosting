# 环境变量设置
# hf_cache="/data/home/weixuzhang/.cache/huggingface"
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0


## 运行evaluate

# PRED_PATH=./output/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100_llama7b.jsonl
# GOLD_DATA_PATH=./eval/cnndm_example_input/cnndm_1.5_-0.5.jsonl
# python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH


PRED_PATH=./output/nqswap_1_0.jsonl.output_topp0.0_genlen10_boost2.0.jsonl
GOLD_DATA_PATH=./eval/nqswap_example_input/nqswap_1_0.jsonl
python eval/evaluate_summary_nqswap.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH 

