#!/bin/bash

hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=0

# fix parameters
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="./eval/cnndm_example_input/cnndm"

# dir
RESULTS_DIR="./results/llama7b"
OUTPUT_DIR="./output/llama7b"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

# adaptive parameters
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
                    
                    # run decode
                    echo "Running decode with adaptive mode ${MODE}..."
                    python group_decode_adaptive_fileio.py \
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
                    
                    # check if succeed
                    if [ $? -eq 0 ]; then
                        echo "Decode completed successfully for mode ${MODE}"
                        
                        # run evaluate
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

# results
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