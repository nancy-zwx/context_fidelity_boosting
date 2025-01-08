#!/bin/bash

# hf_cache=".cache/huggingface"
# mkdir -p ${hf_cache}
# export TRANSFORMERS_CACHE="${hf_cache}"
# export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=2

# parameters
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="./eval/cnndm_example_input/cnndm"

# results dir
RESULTS_DIR="./results/opt350m"
OUTPUT_DIR="./output/opt350m"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

# decode and evaluate

WEIGHTS=("1_0" "1.5_-0.5")

MODEL_PATH=/mnt/nlp/gaoqiang/ckpt/opt-350m

for WEIGHT in "${WEIGHTS[@]}"  
do
    
    
    echo "------------------Processing Weight: ${WEIGHT}--------------------"
    exec > "${RESULTS_DIR}/weight_${WEIGHT}.log" 2>&1
    
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"


    # run decode
    echo "Running decode for weight ${WEIGHT}..."
    python group_decode_fileio.py \
        --max_seq_length ${GLOBALLEN} \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR}\
        --seed 2023 \
        --use_slow_tokenizer \
        --file_mode ${TESTFILE} \
        --decode_truncate_len ${MAXCTXLEN} \
        --decode_depth ${GENLEN} \
        --train_mode decode \
        --projection_top_p ${TOPP}

    # 恢复输出到终端
    exec > /dev/tty 2>&1
    
    # check if succeed
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for weight ${WEIGHT}"
        
        # run evaluate
        echo "Running evaluate for weight ${WEIGHT}..."
        python ./eval/evaluate_summary.py \
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
