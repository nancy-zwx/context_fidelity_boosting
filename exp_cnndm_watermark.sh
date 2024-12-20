#!/bin/bash

# 环境变量设置
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1


# # 实验参数
# GLOBALLEN="2048"
# MAXCTXLEN="1948"
# GENLEN="100"
# FN_PREFIX="eval/cnndm_example_input/cnndm"
# TOPP="0.9"

# # 输入文件名
# TESTFILE="fin|${FN_PREFIX}_1_0.jsonl"
# ###温度参数还没加

# # Context boost参数配置
# BOOST_DELTAS=("1.0") # "1.0" "2.0" "5.0" "10.0"

# # 主实验循环
# for BOOST_DELTA in "${BOOST_DELTAS[@]}"
# do
#     python group_decode_watermark_fileio.py \
#         --max_seq_length ${GLOBALLEN} \
#         --model_name_or_path dummy \
#         --seed 2023 \
#         --use_slow_tokenizer \
#         --file_mode ${TESTFILE} \
#         --decode_truncate_len ${MAXCTXLEN} \
#         --decode_depth ${GENLEN} \
#         --train_mode decode \
#         --projection_top_p ${TOPP} \
#         --context_boost_delta ${BOOST_DELTA} 

# done


################### self cluster gpu ###################

# parameters
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/eval/cnndm_example_input/cnndm"

# results dir
RESULTS_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/results/llama7b"
OUTPUT_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output/llama7b"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

# Context boost参数配置
BOOST_DELTAS=("1.0" "2.0" "5.0" "10.0") # "1.0" "2.0" "5.0" "10.0"

# decode and evaluate
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    echo "------------------Processing Boost Delta: ${BOOST_DELTA}--------------------"
    WEIGHT="1_0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}_boost${BOOST_DELTA}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
    
    # 运行decode
    echo "Running decode with boost delta ${BOOST_DELTA}..."
    python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/group_decode_watermark_fileio.py \
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
    
    # 检查decode是否成功
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for boost delta ${BOOST_DELTA}"
        
        # 运行evaluate
        echo "Running evaluate for boost delta ${BOOST_DELTA}..."
        python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/eval/evaluate_summary.py \
            --pred_path $OUTPUT_FILE \
            --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
            2>&1 | tee "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log"
        
        if [ $? -eq 0 ]; then
            echo "Evaluate completed successfully for boost delta ${BOOST_DELTA}"
        else
            echo "Error: Evaluate failed for boost delta ${BOOST_DELTA}"
            exit 1
        fi
    else
        echo "Error: Decode failed for boost delta ${BOOST_DELTA}"
        exit 1
    fi
done

## results
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    echo "Results for boost delta ${BOOST_DELTA}:"
    if [ -f "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log" ]; then
        cat "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log"
        echo "----------------------------------------"
    else
        echo "No results found for boost delta ${BOOST_DELTA}"
    fi
done

# sh /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/exp_cnndm_watermark.sh