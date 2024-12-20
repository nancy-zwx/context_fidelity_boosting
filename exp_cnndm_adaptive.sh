#!/bin/bash

# 环境变量设置
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

################### self cluster gpu ###################

# 基础参数
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/eval/cnndm_example_input/cnndm"

# 目录设置
RESULTS_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/results/llama7b_adaptive"
OUTPUT_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output/llama7b_adaptive"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

# 自适应参数配置
ADAPTIVE_MODES=("delta_only" "temp_only" "both")
MIN_DELTAS=("1.0")
MAX_DELTAS=("10.0")
MIN_TEMPS=("0.1")
BASE_TEMPS=("1.0")

# decode and evaluate
for MODE in "${ADAPTIVE_MODES[@]}"
do
    for MIN_DELTA in "${MIN_DELTAS[@]}"
    do
        for MAX_DELTA in "${MAX_DELTAS[@]}"
        do
            for MIN_TEMP in "${MIN_TEMPS[@]}"
            do
                for BASE_TEMP in "${BASE_TEMPS[@]}"
                do
                    echo "-------------Processing Adaptive Mode: ${MODE}-------------"
                    echo "Min Delta: ${MIN_DELTA}, Max Delta: ${MAX_DELTA}"
                    echo "Min Temp: ${MIN_TEMP}, Base Temp: ${BASE_TEMP}"
                    
                    WEIGHT="1_0"
                    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
                    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}_adaptive_${MODE}.jsonl"
                    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
                    
                    # 运行decode
                    echo "Running decode with adaptive mode ${MODE}..."
                    python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/group_decode_adaptive_fileio.py \
                        --max_seq_length ${GLOBALLEN} \
                        --model_name_or_path dummy \
                        --seed 2023 \
                        --use_slow_tokenizer \
                        --file_mode ${TESTFILE} \
                        --decode_truncate_len ${MAXCTXLEN} \
                        --decode_depth ${GENLEN} \
                        --train_mode decode \
                        --projection_top_p ${TOPP} \
                        --adaptive_mode ${MODE} \
                        --min_delta ${MIN_DELTA} \
                        --max_delta ${MAX_DELTA} \
                        --min_temp ${MIN_TEMP} \
                        --base_temp ${BASE_TEMP} \
                        --output_stats \
                        --stats_logging
                    
                    # 检查decode是否成功
                    if [ $? -eq 0 ]; then
                        echo "Decode completed successfully for mode ${MODE}"
                        
                        # 运行evaluate
                        echo "Running evaluate for mode ${MODE}..."
                        python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/eval/evaluate_summary.py \
                            --pred_path $OUTPUT_FILE \
                            --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
                            2>&1 | tee "${RESULTS_DIR}/evaluate_results_${MODE}.log"
                        
                        if [ $? -eq 0 ]; then
                            echo "Evaluate completed successfully for mode ${MODE}"
                        else
                            echo "Error: Evaluate failed for mode ${MODE}"
                            exit 1
                        fi
                    else
                        echo "Error: Decode failed for mode ${MODE}"
                        exit 1
                    fi
                done
            done
        done
    done
done

# 打印结果
echo "----------------Final Results----------------"
for MODE in "${ADAPTIVE_MODES[@]}"
do
    echo "Results for adaptive mode ${MODE}:"
    if [ -f "${RESULTS_DIR}/evaluate_results_${MODE}.log" ]; then
        cat "${RESULTS_DIR}/evaluate_results_${MODE}.log"
        echo "----------------------------------------"
    else
        echo "No results found for mode ${MODE}"
    fi
done

# 生成统计报告
echo "Generating statistics report..."
python /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/analysis/generate_stats_report.py \
    --input_dir ${OUTPUT_DIR} \
    --output_file "${RESULTS_DIR}/adaptive_statistics_report.json"

echo "Experiment completed!"

# sh /apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/exp_cnndm_adaptive.sh