# # cnndm
# GLOBALLEN="2048"
# MAXCTXLEN="1948"
# GENLEN="100"
# FN_PREFIX="eval/cnndm_example_input/cnndm"

# for WEIGHT in "1_0" "1.5_-0.5"
# do
#     TOPP="0.9"
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
#     bash run_group_decode_fileio.sh 2023 "0,1" $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP
# done


#!/bin/bash

# 环境变量设置
# hf_cache="/data/home/weixuzhang/.cache/huggingface"
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# cnndm 数据集参数
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
FN_PREFIX="eval/cnndm_example_input/cnndm"



# 运行decode
for WEIGHT in "1_0" "1.5_-0.5"
do
    TOPP="0.9"
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