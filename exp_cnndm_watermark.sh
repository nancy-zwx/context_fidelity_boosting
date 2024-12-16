#!/bin/bash

# 环境变量设置
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0


# 实验参数
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
FN_PREFIX="eval/cnndm_example_input/cnndm"
TOPP="0.9"

# 输入文件名
TESTFILE="fin|${FN_PREFIX}_1_0.jsonl"
###温度参数还没加

# Context boost参数配置
BOOST_DELTAS=("1.0") # "1.0" "2.0" "5.0" "10.0"

# 主实验循环
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    python group_decode_watermark_fileio.py \
        --max_seq_length ${GLOBALLEN} \
        --model_name_or_path dummy \
        --seed 2023 \
        --use_slow_tokenizer \
        --file_mode ${TESTFILE} \
        --decode_truncate_len ${MAXCTXLEN} \
        --decode_depth ${GENLEN} \
        --train_mode decode \
        --projection_top_p ${TOPP} \
        --context_boost_delta ${BOOST_DELTA} 

done
