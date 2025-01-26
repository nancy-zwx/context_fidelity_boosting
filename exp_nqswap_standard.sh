#!/bin/bash

hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=0

# parameters
GLOBALLEN="2048"
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="./eval/nqswap_example_input/nqswap"
TOPP="0.0"


MODEL_PATH=/home/gaoqiang/ckpt/Mistral-7B-Instruct-v0.3
MODEL_NAME=$(basename ${MODEL_PATH})

# results dir
RESULTS_DIR="./results/nq/${MODEL_NAME}"
OUTPUT_DIR="./output/nq/${MODEL_NAME}"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}


WEIGHTS=("1_0" "2_-1")
for WEIGHT in "${WEIGHTS[@]}"  
do
    echo "------------------Processing Weight: ${WEIGHT}--------------------"
    exec > "${RESULTS_DIR}/nqstandard_weight_${WEIGHT}_.log" 2>&1
    
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
    
    # run decode
    echo "Running decode for weight ${WEIGHT}..."
    
    python group_decode_fileio.py \
        --max_seq_length ${GLOBALLEN} \
        --model_name_or_path ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
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
        python ./eval/evaluate_summary_nqswap.py \
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
