#!/bin/bash

hf_cache=".cache/huggingface"
mkdir -p ${hf_cache}
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export CUDA_VISIBLE_DEVICES=0

# parameters
GLOBALLEN="2048"
MAXCTXLEN="1948"
GENLEN="100"
TOPP="0.9"
FN_PREFIX="./eval/cnndm_example_input/cnndm"

# results dir
RESULTS_DIR="./results/cnn/llama2-7b"
OUTPUT_DIR="./output/cnn/llama2-7b"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

MODEL_PATH=/home/gaoqiang/ckpt/Llama-2-7b-chat-hf

# Context boost delta
# BOOST_DELTAS=("1.0" "2.0" "5.0" "10.0") # "1.0" "2.0" "5.0" "10.0"
BOOST_DELTAS=($(seq 1.0 1.0 1.0))

# decode and evaluate
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    
    echo "------------------Processing Boost Delta: ${BOOST_DELTA}--------------------"
    # exec > "${RESULTS_DIR}/delta_${BOOST_DELTA}.log" 2>&1
    WEIGHT="1_0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}_boost${BOOST_DELTA}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
    
    # # run decode
    # echo "Running decode with boost delta ${BOOST_DELTA}..."
    # python group_decode_watermark_fileio.py \
    #     --max_seq_length ${GLOBALLEN} \
    #     --model_name_or_path ${MODEL_PATH} \
    #     --output_dir ${OUTPUT_DIR}\
    #     --seed 2023 \
    #     --use_slow_tokenizer \
    #     --file_mode ${TESTFILE} \
    #     --decode_truncate_len ${MAXCTXLEN} \
    #     --decode_depth ${GENLEN} \
    #     --train_mode decode \
    #     --projection_top_p ${TOPP} \
    #     --context_boost_delta ${BOOST_DELTA}

    # # 恢复输出到终端
    # exec > /dev/tty 2>&1
    # check if succeed
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for boost delta ${BOOST_DELTA}"
        
        # run evaluate
        echo "Running evaluate for boost delta ${BOOST_DELTA}..."
        python ./eval/evaluate_summary.py \
            --pred_path $OUTPUT_FILE \
            --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
            2>&1 | tee "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}_test.log"
        
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
