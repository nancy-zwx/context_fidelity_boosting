import argparse
from collections import defaultdict
from datasets import load_dataset
import evaluate
from huggingface_hub import login
import json
import logging
import os
from pathlib import Path
import statistics
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer

from evaluation import calculate_acc,ems
import bert_score.score as bert_score
from alignscore.alignscore import AlignScore

# login(token="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")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models_with_cache(cache_dir=".cache/huggingface"):
    # os.makedirs(cache_dir, exist_ok=True)
    
    try:
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/gaoqiang/ckpt/roberta-base",
            cache_dir=cache_dir,
            local_files_only=True if os.path.exists(os.path.join(cache_dir, "roberta-base")) else False
        )
        
        logger.info("Loading FactKB model...")
        model_path = os.path.join(cache_dir, "FactKB")
        if not os.path.exists(model_path):
            logger.info("Downloading FactKB model...")
            os.makedirs(model_path, exist_ok=True)
            
        factkb = AutoModelForSequenceClassification.from_pretrained(
            "/home/gaoqiang/ckpt/FactKB",
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

def evaluate_qa(index2ex, eval_file, tokenizer, factkb):
    if tokenizer is None or factkb is None:
        logger.error("Models not loaded properly")
        return
        
    logger.info(f"Processing file: {eval_file}")
    
    try:
        with open(eval_file, "r") as f:
            output_data = [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return
    
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []
    
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
            article = index2ex[index]["context_string"]
            summary = pred
            inputs = [[summary, article]]
            
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
    
    if all_fact_score:
        print(f"Average fact score: {statistics.mean(all_fact_score):.4f}")
    
    logger.info("Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = defaultdict(list)
    for pred, gold in zip(all_pred, all_gold):
        score = scorer.score(pred, gold)
        for key, value in score.items():
            rouge_scores[key].append(value.fmeasure)
    
    results = {k: statistics.mean(v) for k, v in rouge_scores.items()}
    print("ROUGE results:", results)


        # # bertscore
    # print("Computing BERTScore...")
    # P, R, F1 = bert_score(all_pred, all_gold, lang="en", verbose=True)
    # # P, R, F1 分别是 Precision, Recall 和 F1 分数
    # print(P.align)
    # print("Bertscore Precision:", P.mean().item())
    # print("Bertscore Recall:", R.mean().item())
    # print("Bertscore F1 Score:", F1.mean().item())

    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
    # print("bertscore: ", results)
    print("bertscore: ")
    for k, v in results.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}: {statistics.mean(v)}")

    # 两种算法区别很大


    # AlignScore
    # logger.info("Computing AlignScore...")
    

    # scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='/home/gaoqiang/ckpt/roberta-base', evaluation_mode='nli_sp')
    # align_score = scorer.score(contexts=all_doc, claims=all_pred)
    # print(align_score)

    # Accuracy
    logger.info("Computing Accuracy...")
    exact_match_count=0
    acc_scores=0
    import pdb
    for pred, gold in zip(all_pred, all_gold):
        exact_match_count += ems(pred, gold)
        acc_scores += calculate_acc(pred, gold)
    
    em = round(exact_match_count/len(all_pred), 4)
    acc_score = round(acc_scores/len(all_pred), 4)
    print("Exact Match:", em)
    print("Accuracy:", acc_score)


def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["assigned_process"] == 0:
                raw_data.append(ex)
    return raw_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./eval/cnndm_example_input/cnndm_1_0.jsonl")
    parser.add_argument("--pred_path", type=str, default="./eval/cnndm_example_input/cnndm_1.5_-0.5.jsonl.output_topp0.9_genlen100.jsonl")
    args = parser.parse_args()

    cache_dir = ".cache/huggingface"
    tokenizer, factkb = load_models_with_cache(cache_dir)
    
    if tokenizer is not None and factkb is not None:
        index2ex = entity_data(args.data_path)
        evaluate_qa(index2ex, args.pred_path, tokenizer, factkb)
    else:
        logger.error("Failed to load models. Exiting...")