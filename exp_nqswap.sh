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

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# # 配置参数
# GLOBALLEN="2048"
# MAXCTXLEN="2038"
# GENLEN="10"
# FN_PREFIX="eval/nqswap_example_input/nqswap"


# for WEIGHT in "1_0" "2_-1" # "1_0" "2_-1"
# do
#     TOPP="0.0"
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    
#     python group_decode_fileio.py \
#         --max_seq_length ${GLOBALLEN} \
#         --model_name_or_path dummy \
#         --seed 2023 \
#         --use_slow_tokenizer \
#         --file_mode ${TESTFILE} \
#         --decode_truncate_len ${MAXCTXLEN} \
#         --decode_depth ${GENLEN} \
#         --train_mode decode \
#         --projection_top_p ${TOPP}
# done


################### self cluster gpu ###################

# parameters
GLOBALLEN="2048"
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="nqswap"
TOPP="0.0"

# results dir
RESULTS_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/results"
OUTPUT_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}


WEIGHTS=("1_0" "2_-1")
for WEIGHT in "${WEIGHTS[@]}"  # "1_0" "2_-1"
do
    echo "------------------Processing Weight: ${WEIGHT}--------------------"
    
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
    
    # 运行decode
    echo "Running decode for weight ${WEIGHT}..."
    python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/group_decode_fileio.py \
        --max_seq_length ${GLOBALLEN} \
        --model_name_or_path dummy \
        --seed 2023 \
        --use_slow_tokenizer \
        --file_mode ${TESTFILE} \
        --decode_truncate_len ${MAXCTXLEN} \
        --decode_depth ${GENLEN} \
        --train_mode decode \
        --projection_top_p ${TOPP}
    
    # 检查decode是否成功
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for weight ${WEIGHT}"
        
        # 运行evaluate
        echo "Running evaluate for weight ${WEIGHT}..."
        python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/eval/evaluate_summary_nqswap.py \
            --pred_path $OUTPUT_FILE \
            --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
            2>&1 | tee "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
        
        if [ $? -eq 0 ]; then
            echo "Evaluate completed successfully for weight ${WEIGHT}"
        else
            echo "Error: Evaluate failed for weight ${WEIGHT}"
            exit 1
        fi
    else
        echo "Error: Decode failed for weight ${WEIGHT}"
        exit 1
    fi
done

## results
for WEIGHT in "${WEIGHTS[@]}"
do
    echo "Results for weight ${WEIGHT}:"
    if [ -f "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log" ]; then
        cat "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
        echo "----------------------------------------"
    else
        echo "No results found for weight ${WEIGHT}"
    fi
done

# sh /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/exp_nqswap.sh