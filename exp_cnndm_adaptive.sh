#!/bin/bash

hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=1

# parameters
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="./eval/cnndm_example_input/cnndm"

# results dir
RESULTS_DIR="./results/llama7b"
OUTPUT_DIR="./output/llama7b"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}


# parameters
MIN_DELTAS=("2.0")  
MAX_DELTAS=("10.0")  
LAMBDA1S=("0.6")    # lambda2 = 1-lambda1
USE_GLOBALS=("false") # 是否只用全局增强

# decode and evaluate
for MIN_DELTA in "${MIN_DELTAS[@]}"
do
    for MAX_DELTA in "${MAX_DELTAS[@]}"
    do
        for LAMBDA1 in "${LAMBDA1S[@]}"
        do
            LAMBDA2=$(echo "1 - $LAMBDA1" | bc)
            
            for USE_GLOBAL in "${USE_GLOBALS[@]}"
            do
                echo "Processing: min_delta=${MIN_DELTA}, max_delta=${MAX_DELTA}, lambda1=${LAMBDA1}, lambda2=${LAMBDA2}, global=${USE_GLOBAL}"
                
                WEIGHT="1_0"
                TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
                BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}_delta${MIN_DELTA}-${MAX_DELTA}_l1${LAMBDA1}_l2${LAMBDA2}_global${USE_GLOBAL}.jsonl"
                OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
                
                # run decode
                echo "Running decode..."
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
                    --min_delta ${MIN_DELTA} \
                    --max_delta ${MAX_DELTA} \
                    --lambda1 ${LAMBDA1} \
                    --lambda2 ${LAMBDA2} \
                    --use_global ${USE_GLOBAL}
                
                # check if succeed
                if [ $? -eq 0 ]; then
                    echo "Decode completed successfully"
                    
                    # run evaluation
                    echo "Running evaluate..."
                    python ./eval/evaluate_summary.py \
                        --pred_path $OUTPUT_FILE \
                        --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
                        2>&1 | tee "${RESULTS_DIR}/evaluate_results_delta${MIN_DELTA}-${MAX_DELTA}_l1${LAMBDA1}_l2${LAMBDA2}_global${USE_GLOBAL}.log"
                    
                    if [ $? -eq 0 ]; then
                        echo "Evaluate completed successfully"
                    else
                        echo "Error: Evaluate failed"
                        exit 1
                    fi
                else
                    echo "Error: Decode failed"
                    exit 1
                fi
            done
        done
    done
done

# print results
echo "All Results:"
for MIN_DELTA in "${MIN_DELTAS[@]}"
do
    for MAX_DELTA in "${MAX_DELTAS[@]}"
    do
        for LAMBDA1 in "${LAMBDA1S[@]}"
        do
            LAMBDA2=$(echo "1 - $LAMBDA1" | bc)
            for USE_GLOBAL in "${USE_GLOBALS[@]}"
            do
                RESULT_FILE="${RESULTS_DIR}/evaluate_results_delta${MIN_DELTA}-${MAX_DELTA}_l1${LAMBDA1}_l2${LAMBDA2}_global${USE_GLOBAL}.log"
                if [ -f "$RESULT_FILE" ]; then
                    echo "Results for min_delta=${MIN_DELTA}, max_delta=${MAX_DELTA}, lambda1=${LAMBDA1}, lambda2=${LAMBDA2}, global=${USE_GLOBAL}:"
                    cat "$RESULT_FILE"
                    echo "----------------------------------------"
                else
                    echo "No results for min_delta=${MIN_DELTA}, max_delta=${MAX_DELTA}, lambda1=${LAMBDA1}, lambda2=${LAMBDA2}, global=${USE_GLOBAL}"
                fi
            done
        done
    done
done