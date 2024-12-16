import argparse
from collections import defaultdict
from datasets import load_dataset
import evaluate
from huggingface_hub import login
from ipdb import set_trace as bp
import json
import logging
import os
from pathlib import Path
import statistics
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


login(token="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")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_timeout(url, filename, timeout=30):
    """带超时的下载函数"""
    try:
        response = requests.get(url, timeout=timeout)
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def load_models_with_cache(cache_dir="/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"):
    """使用本地缓存加载模型"""
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 设置模型缓存路径
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base",
            cache_dir=cache_dir,
            local_files_only=True if os.path.exists(os.path.join(cache_dir, "roberta-base")) else False
        )
        
        logger.info("Loading FactKB model...")
        model_path = os.path.join(cache_dir, "FactKB")
        if not os.path.exists(model_path):
            logger.info("Downloading FactKB model...")
            os.makedirs(model_path, exist_ok=True)
            
        factkb = AutoModelForSequenceClassification.from_pretrained(
            "bunsenfeng/FactKB",
            cache_dir=cache_dir,
            local_files_only=True if os.path.exists(model_path) else False,
            num_labels=2
        )
        
        if torch.cuda.is_available():
            factkb = factkb.cuda()
            
        return tokenizer, factkb
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None
    
# evaluate fackKB: Put your huggingface access tokens
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2, use_auth_token=access_token)

# def evaluate_qa(index2ex, eval_file):
#     print(eval_file)
#     all_gold = []
#     all_pred = []
#     all_doc = []
#     all_fact_score = []

#     if os.path.exists(eval_file) == False:
#         return 0
#     with open(eval_file, "r") as f:
#         output_data = [json.loads(line) for line in f]
#     cov_em_all = []
#     category2em = defaultdict(list)
#     id2ex_output = {}
#     for i, output in enumerate(output_data):
#         index = output["input_index"]
#         pred = output["string"][0]
#         gold = index2ex[index]["gold_answers"] 
#         if len(pred) < 3:
#             print(pred)
#             continue
#         all_gold.append(gold)
#         all_pred.append(pred)
#         if len(pred) < 3:
#             print(f"pred: {pred}")

#         article = index2ex[index]["article"]
#         summary = pred
#         input = [[summary, article]]
#         tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
#         result = torch.softmax(factkb(**tokens).logits, dim = 1)
#         # bp()
#         fact_score = result[0][1].item()

#         all_fact_score.append(fact_score)
#         all_doc.append(article)
#         output_dict = index2ex[index].copy()
#         output_dict["pred"] = pred
#         id2ex_output[i] = output_dict

#     print("fact_score: ", statistics.mean(all_fact_score))
#     # print(statistics.mean(cov_em_all))

#     # 使用 rouge-score 包替代 evaluate.load('rouge')
#     from rouge_score import rouge_scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
#     # 计算每个预测的 ROUGE 分数
#     rouge_scores = defaultdict(list)
#     for pred, gold in zip(all_pred, all_gold):
#         score = scorer.score(pred, gold)
#         for key, value in score.items():
#             rouge_scores[key].append(value.fmeasure)
    
#     # 计算平均值
#     results = {
#         'rouge1': statistics.mean(rouge_scores['rouge1']),
#         'rouge2': statistics.mean(rouge_scores['rouge2']),
#         'rougeL': statistics.mean(rouge_scores['rougeL'])
#     }
#     print("rouge results: ", results)

#     # BERTScore 评估
#     try:
#         # 明确指定 token
#         bertscore = evaluate.load("bertscore", use_auth_token="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")
#         results = bertscore.compute(
#             predictions=all_pred, 
#             references=all_doc, 
#             lang="en",
#             model_type="microsoft/deberta-xlarge-mnli"  # 指定一个具体的模型
#         )
#         print("bertscore: ")
#         for k, v in results.items():
#             if k in ["precision", "recall", "f1"]:
#                 print(f"{k}: {statistics.mean(v)}")
#     except Exception as e:
#         print("Error computing BERTScore:", e)
#         # 如果还是失败，可以尝试使用本地的 bert-score 包
#         try:
#             from bert_score import score
#             P, R, F1 = score(all_pred, all_doc, lang="en", verbose=True)
#             print("Local BERTScore:")
#             print(f"precision: {P.mean().item()}")
#             print(f"recall: {R.mean().item()}")
#             print(f"f1: {F1.mean().item()}")
#         except Exception as e2:
#             print("Error computing local BERTScore:", e2)


#     # rouge = evaluate.load('rouge')
#     # results = rouge.compute(predictions=all_pred, references=all_gold)
#     # print("rouge results: ", results)

#     # bertscore = evaluate.load("bertscore")
#     # results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
#     # # print("bertscore: ", results)
#     # print("bertscore: ")    
#     # for k, v in results.items():
#     #     if k in ["precision", "recall", "f1"]:
#     #         print(f"{k}: {statistics.mean(v)}")


#     return id2ex_output



def evaluate_qa(index2ex, eval_file, tokenizer, factkb):
    """评估函数"""
    if tokenizer is None or factkb is None:
        logger.error("Models not loaded properly")
        return
        
    logger.info(f"Processing file: {eval_file}")
    
    # 读取和验证数据
    try:
        with open(eval_file, "r") as f:
            output_data = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    # 初始化列表
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []
    
    # 处理每个样本
    for i, output in enumerate(tqdm(output_data)):
        try:
            index = output["input_index"]
            pred = output["string"][0]
            gold = index2ex[index]["gold_answers"]
            
            if len(pred) < 3:
                continue
                
            all_gold.append(gold)
            all_pred.append(pred)
            
            # 计算fact score
            article = index2ex[index]["context_string"]  # 使用context_string而不是article
            summary = pred
            inputs = [[summary, article]]
            
            # 转换为tensor并移至GPU
            with torch.no_grad():
                tokens = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True)
                if torch.cuda.is_available():
                    tokens = {k: v.cuda() for k, v in tokens.items()}
                result = torch.softmax(factkb(**tokens).logits, dim=1)
                fact_score = result[0][1].cpu().item()
            
            all_fact_score.append(fact_score)
            all_doc.append(article)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            continue
    
    # 计算指标
    if all_fact_score:
        print(f"Average fact score: {statistics.mean(all_fact_score):.4f}")
    
    # ROUGE计算
    logger.info("Computing ROUGE scores...")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = defaultdict(list)
    for pred, gold in zip(all_pred, all_gold):
        score = scorer.score(pred, gold)
        for key, value in score.items():
            rouge_scores[key].append(value.fmeasure)
    
    results = {k: statistics.mean(v) for k, v in rouge_scores.items()}
    print("ROUGE results:", results)


# read data
def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["assigned_process"] == 0:
                raw_data.append(ex)
            # break
        # raw_data = json.loads(f.read())
    return raw_data


# if __name__ == "__main__":
#     # args parse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", type=str, default="./eval/cnndm_example_input/cnndm_1_0.jsonl")
#     parser.add_argument("--pred_path", type=str, default="./eval/cnndm_example_input/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100.jsonl")
#     args = parser.parse_args()

#     data_path = args.data_path
#     pred_path = args.pred_path
#     index2ex = entity_data(data_path)
#     evaluate_qa(index2ex, pred_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./eval/cnndm_example_input/cnndm_1_0.jsonl")
    parser.add_argument("--pred_path", type=str, default="./eval/cnndm_example_input/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100.jsonl")
    args = parser.parse_args()

    # 设置缓存目录
    cache_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    
    # 加载模型（使用缓存）
    tokenizer, factkb = load_models_with_cache(cache_dir)
    
    if tokenizer is not None and factkb is not None:
        # 读取数据
        index2ex = entity_data(args.data_path)
        # 运行评估
        evaluate_qa(index2ex, args.pred_path, tokenizer, factkb)
    else:
        logger.error("Failed to load models. Exiting...")
    

