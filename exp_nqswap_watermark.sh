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

# results dir
RESULTS_DIR="./results/llama7b"
OUTPUT_DIR="./output/llama7b"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}


# Context boost参数配置
BOOST_DELTAS=("1.0" "2.0" "5.0" "10.0") 

# decode and evaluate
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    echo "------------------Processing Boost Delta: ${BOOST_DELTA}--------------------"
    WEIGHT="1_0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    BASE_OUTPUT_FILE="$(basename ${FN_PREFIX}_${WEIGHT}.jsonl).output_topp${TOPP}_genlen${GENLEN}_boost${BOOST_DELTA}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_OUTPUT_FILE}"
    
    # run decode
    echo "Running decode with boost delta ${BOOST_DELTA}..."
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
    
    # check if succeed
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for boost delta ${BOOST_DELTA}"
        
        # run evaluate
        echo "Running evaluate for boost delta ${BOOST_DELTA}..."
        python ./eval/evaluate_summary_nqswap.py \
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
