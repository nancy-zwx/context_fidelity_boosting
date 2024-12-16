# # nq-swap
# GLOBALLEN="2048"
# MAXCTXLEN="2038"
# GENLEN="10"
# FN_PREFIX="eval/nqswap_example_input/nqswap"

# for WEIGHT in "1_0" "2_-1"
# do
#     TOPP="0.0"
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
#     bash run_group_decode_fileio.sh 2023 "0" $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP
# done


#!/bin/bash

# 环境变量设置
hf_cache="/data/home/weixuzhang/.cache/huggingface"
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# 配置参数
GLOBALLEN="2048"
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="eval/nqswap_example_input/nqswap"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

for WEIGHT in "1_0" "2_-1" # "1_0" "2_-1"
do
    TOPP="0.0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    
    python group_decode_fileio.py \
        --max_seq_length ${GLOBALLEN} \
        --model_name_or_path dummy \
        --seed 2023 \
        --use_slow_tokenizer \
        --file_mode ${TESTFILE} \
        --decode_truncate_len ${MAXCTXLEN} \
        --decode_depth ${GENLEN} \
        --train_mode decode \
        --projection_top_p ${TOPP}
done