echo "------------------Installation Environment--------------------"

# python设置
export PYTHON=/usr/bin/python3.7
export PYTHON_HOME=/usr/lib/python3.7
export PATH=$PYTHON_HOME/bin:$PATH

# proxy设置
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

# NCCL设置
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=1

# HF设置
hf_cache="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
mkdir -p ${hf_cache}
# export TRANSFORMERS_OFFLINE=1  # 如果模型已经下载过，强制使用本地缓存
export TRANSFORMERS_CACHE="${hf_cache}"
export HF_HOME="${hf_cache}"
export HF_ENDPOINT="https://huggingface.co"
export HF_TOKEN="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"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0

# 安装依赖

# pip install -r requirements.txt
pip install torch
pip install transformers
pip install --upgrade transformers
pip install accelerate
pip install evaluate
pip install rouge-score
pip install bert-score
pip install datasets
pip install tqdm
pip install numpy
pip install pandas
pip install scikit-learn
pip install ipdb
pip install sentencepiece
pip install nltk
pip install tensorboard
pip install time
pip install termcolor


echo "Verifying installations..."


echo "------------------Experiment cnndm--------------------"

# # parameters
# GLOBALLEN="2048"
# MAXCTXLEN="1948"
# GENLEN="100"
# TOPP="0.9"
# FN_PREFIX="eval/cnndm_example_input/cnndm"

# # results dir
# RESULTS_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/results"
# OUTPUT_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/output"
# mkdir -p ${RESULTS_DIR}
# mkdir -p ${OUTPUT_DIR}



######################### standard and cad decoding##########################
# # choice1: decode
# for WEIGHT in "1_0" "1.5_-0.5"
# do
#     TOPP="0.9"
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

# # choice2: evaluate
# PRED_PATH=./eval/cnndm_example_input/cnndm_1_0.jsonl.output_topp0.9_genlen100.jsonl 
# GOLD_DATA_PATH=./eval/cnndm_example_input/cnndm_1_0.jsonl
# python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH


# # decode and evaluate
# for WEIGHT in "1_0" "1.5_-0.5"  #for WEIGHT in "1_0" "1.5_-0.5"
# do
#     echo "------------------Processing Weight: ${WEIGHT}--------------------"
    
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
#     OUTPUT_FILE="${FN_PREFIX}_${WEIGHT}.jsonl.output_topp${TOPP}_genlen${GENLEN}.jsonl"
#     OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}"


#     # 运行decode
#     echo "Running decode for weight ${WEIGHT}..."
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
    
#     # 检查decode是否成功
#     if [ $? -eq 0 ]; then
#         echo "Decode completed successfully for weight ${WEIGHT}"
        
#         # 运行evaluate
#         echo "Running evaluate for weight ${WEIGHT}..."
#         python eval/evaluate_summary.py \
#             --pred_path $OUTPUT_FILE \
#             --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
#             2>&1 | tee "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
        
#         if [ $? -eq 0 ]; then
#             echo "Evaluate completed successfully for weight ${WEIGHT}"
#         else
#             echo "Error: Evaluate failed for weight ${WEIGHT}"
#             exit 1
#         fi
#     else
#         echo "Error: Decode failed for weight ${WEIGHT}"
#         exit 1
#     fi
# done

# ## results
# for WEIGHT in "1_0" "1.5_-0.5"
# do
#     echo "Results for weight ${WEIGHT}:"
#     if [ -f "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log" ]; then
#         cat "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
#         echo "----------------------------------------"
#     else
#         echo "No results found for weight ${WEIGHT}"
#     fi
# done


# ######################### watermark decoding##########################

# # Context boost参数配置
# BOOST_DELTAS=("5.0" "10.0") # "1.0" "2.0" "5.0" "10.0"

# # decode and evaluate
# for BOOST_DELTA in "${BOOST_DELTAS[@]}"
# do
#     echo "------------------Processing Boost Delta: ${BOOST_DELTA}--------------------"
#     WEIGHT="1_0"
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
#     OUTPUT_FILE="${FN_PREFIX}_${WEIGHT}.jsonl.output_topp${TOPP}_genlen${GENLEN}_boost${BOOST_DELTA}.jsonl"
#     OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}"
    
#     # 运行decode
#     echo "Running decode with boost delta ${BOOST_DELTA}..."
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
    
#     # 检查decode是否成功
#     if [ $? -eq 0 ]; then
#         echo "Decode completed successfully for boost delta ${BOOST_DELTA}"
        
#         # 运行evaluate
#         echo "Running evaluate for boost delta ${BOOST_DELTA}..."
#         python eval/evaluate_summary.py \
#             --pred_path $OUTPUT_FILE \
#             --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
#             2>&1 | tee "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log"
        
#         if [ $? -eq 0 ]; then
#             echo "Evaluate completed successfully for boost delta ${BOOST_DELTA}"
#         else
#             echo "Error: Evaluate failed for boost delta ${BOOST_DELTA}"
#             exit 1
#         fi
#     else
#         echo "Error: Decode failed for boost delta ${BOOST_DELTA}"
#         exit 1
#     fi
# done

# ## results
# for BOOST_DELTA in "${BOOST_DELTAS[@]}"
# do
#     echo "Results for boost delta ${BOOST_DELTA}:"
#     if [ -f "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log" ]; then
#         cat "${RESULTS_DIR}/evaluate_results_boost${BOOST_DELTA}.log"
#         echo "----------------------------------------"
#     else
#         echo "No results found for boost delta ${BOOST_DELTA}"
#     fi
# done


echo "------------------Experiment nqswap--------------------"

# parameters
GLOBALLEN="2048"
MAXCTXLEN="2038"
GENLEN="10"
FN_PREFIX="eval/nqswap_example_input/nqswap"
TOPP="0.0"

# results dir
RESULTS_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/results"
OUTPUT_DIR="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/output"
mkdir -p ${RESULTS_DIR}
mkdir -p ${OUTPUT_DIR}

######################## standard and cad decoding##########################

# for WEIGHT in "1_0" "2_-1"  # "1_0" "2_-1"
# do
#     echo "------------------Processing Weight: ${WEIGHT}--------------------"
    
#     TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
#     OUTPUT_FILE="${FN_PREFIX}_${WEIGHT}.jsonl.output_topp${TOPP}_genlen${GENLEN}.jsonl"
#     OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}"
    
#     # 运行decode
#     echo "Running decode for weight ${WEIGHT}..."
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
    
#     # 检查decode是否成功
#     if [ $? -eq 0 ]; then
#         echo "Decode completed successfully for weight ${WEIGHT}"
        
#         # 运行evaluate
#         echo "Running evaluate for weight ${WEIGHT}..."
#         python eval/evaluate_summary_nqswap.py \
#             --pred_path $OUTPUT_FILE \
#             --data_path "${FN_PREFIX}_${WEIGHT}.jsonl" \
#             2>&1 | tee "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
        
#         if [ $? -eq 0 ]; then
#             echo "Evaluate completed successfully for weight ${WEIGHT}"
#         else
#             echo "Error: Evaluate failed for weight ${WEIGHT}"
#             exit 1
#         fi
#     else
#         echo "Error: Decode failed for weight ${WEIGHT}"
#         exit 1
#     fi
# done

# ## results
# for WEIGHT in "1_0" "2_-1"
# do
#     echo "Results for weight ${WEIGHT}:"
#     if [ -f "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log" ]; then
#         cat "${RESULTS_DIR}/evaluate_results_${WEIGHT}.log"
#         echo "----------------------------------------"
#     else
#         echo "No results found for weight ${WEIGHT}"
#     fi
# done

######################## watermark decoding##########################

# Context boost参数配置
BOOST_DELTAS=("1.0" "2.0") # "1.0" "2.0" "5.0" "10.0"

# decode and evaluate
for BOOST_DELTA in "${BOOST_DELTAS[@]}"
do
    echo "------------------Processing Boost Delta: ${BOOST_DELTA}--------------------"
    WEIGHT="1_0"
    TESTFILE="fin|${FN_PREFIX}_${WEIGHT}.jsonl"
    OUTPUT_FILE="${FN_PREFIX}_${WEIGHT}.jsonl.output_topp${TOPP}_genlen${GENLEN}_boost${BOOST_DELTA}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE}"
    
    # 运行decode
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
    
    # 检查decode是否成功
    if [ $? -eq 0 ]; then
        echo "Decode completed successfully for boost delta ${BOOST_DELTA}"
        
        # 运行evaluate
        echo "Running evaluate for boost delta ${BOOST_DELTA}..."
        python eval/evaluate_summary_nqswap.py \
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



## result cnndm, opt-125m:
## weight 1_0(standard): Average fact score: 0.5743 ROUGE results: {'rouge1': 0.2216483068181287, 'rouge2': 0.035331129550254706, 'rougeL': 0.13635891213281745}
## weight 1.5_-0.5(cad): Average fact score: 0.3673 ROUGE results: {'rouge1': 0.15058797377058003, 'rouge2': 0.03063898742636132, 'rougeL': 0.09281894999281314}
## boost delta 1: Average fact score: 0.6072 ROUGE results: {'rouge1': 0.24611884504527154, 'rouge2': 0.05474418347040802, 'rougeL': 0.15699378698132646}
## boost delta 2: Average fact score: 0.6349 ROUGE results: {'rouge1': 0.26237905276087137, 'rouge2': 0.07462329641606252, 'rougeL': 0.17588758583711162}
## boost delta 5: Average fact score: 0.6099 ROUGE results: {'rouge1': 0.23276223921074368, 'rouge2': 0.08193827153778216, 'rougeL': 0.16525847159154564}
## boost delta 10: Average fact score: 0.7236 ROUGE results: {'rouge1': 0.22185340462514158, 'rouge2': 0.08237843274501104, 'rougeL': 0.1605608075893064}

## result cnndm, opt-350m:
## weight 1_0(standard): Average fact score: 0.5786 ROUGE results: {'rouge1': 0.22694484731756787, 'rouge2': 0.03967234821559698, 'rougeL': 0.13897066228136815}
## weight 1.5_-0.5(cad): Average fact score: 0.4052 ROUGE results: {'rouge1': 0.15416726621638735, 'rouge2': 0.034005134542419574, 'rougeL': 0.09778655059335113}
## boost delta 1: Average fact score: 0.6125 ROUGE results: {'rouge1': 0.2497402464125327, 'rouge2': 0.06407571110488566, 'rougeL': 0.16205419015225098}
## boost delta 2: Average fact score: 0.6330 ROUGE results: {'rouge1': 0.25297441845986207, 'rouge2': 0.07363212898450203, 'rougeL': 0.16990657133327663}
## boost delta 5: Average fact score: 0.6885 ROUGE results: {'rouge1': 0.2397600605772296, 'rouge2': 0.08759701923338263, 'rougeL': 0.17284471631803983}
## boost delta 10: Average fact score: 0.7967 ROUGE results: {'rouge1': 0.23373475317564102, 'rouge2': 0.08328019066338498, 'rougeL': 0.16718793812908503}

## result nqswap, opt-125m: 
## weight 1_0(standard): Average fact score: 0.2994 ROUGE results: {'rouge1': 0.04585790310659028, 'rouge2': 0.024843214771946957, 'rougeL': 0.04585790310659028}
## weight 2_-1(cad): Average fact score: 0.2082, ROUGE results: {'rouge1': 0.04555347995895633, 'rouge2': 0.024843214771946957, 'rougeL': 0.04555347995895633}
## boost delta 1: Average fact score: 0.3797 ROUGE results: {'rouge1': 0.044586486853170244, 'rouge2': 0.02424696756106691, 'rougeL': 0.044586486853170244}
## boost delta 2: 
## boost delta 5: Average fact score: 0.6172, ROUGE results: {'rouge1': 0.031176000825806273, 'rouge2': 0.015783169180890136, 'rougeL': 0.031176000825806273}
## boost delta 10: Average fact score: 0.4133, ROUGE results: {'rouge1': 0.04030921318271373, 'rouge2': 0.02056905189402463, 'rougeL': 0.04030921318271373}

## result nqswap, opt-350m:
## weight 1_0(standard): 
## weight 2_-1(cad): 
## boost delta 1: 
## boost delta 2: 
## boost delta 5: 
## boost delta 10: 

## result cnndm, llama-7b:
## weight 1_0(standard): Average fact score: 0.7494 ROUGE results: {'rouge1': 0.2792295751583305, 'rouge2': 0.08255960177635246, 'rougeL': 0.1862742341199251}
## weight 1.5_-0.5(cad): Average fact score: 0.5609 ROUGE results: {'rouge1': 0.20336594785387901, 'rouge2': 0.08294773873759222, 'rougeL': 0.1384313522025869}
## boost delta 1: 
## boost delta 2: 
## boost delta 5: 
## boost delta 10: 