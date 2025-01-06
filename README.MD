## Context Fidelity Boosting (CFB)

A novel approach for enhancing context fidelity in large language models through dynamic context optimization and boosting techniques.

### Overview

Context Fidelity Boosting (CFB) is a framework designed to improve the contextual understanding and response accuracy of large language models. It implements dynamic context optimization strategies and boosting techniques to enhance model performance in context-sensitive tasks.

### Input format
```json
{
    "input_index": 0, // instances 
    "assigned_model": "huggyllama/llama-7b", // used model
    "assigned_process": 0, // device
    "context_string": "The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 1078 episodes . How many episodes are in chicago fire season 4 ?", // with context input
    "assigned_weight": 2, // weight(1+alpha)
    "filter_p": 1.0, // optional filtering for low-probablity tokens, disabled by default
}
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "huggyllama/llama-7b", // used model
    "assigned_process": 1, // device
    "context_string": "How many episodes are in chicago fire season 4 ?", //without context input
    "assigned_weight": -1, // weight(-alpha)
}
...
```

### Running experiments on CNN-DM and NQ-Swap
for experiments on CNN_DM:
run `exp_cnndm_standard.sh` for standard decoding, which subsequently calls `group_decode_fileio.py`.
run `exp_cnndm_watermark.sh` for context fidelity boosting where delta is fixed, which subsequently calls `group_decode_watermark_fileio.py`.
run `exp_cnndm_adaptive.sh` for context fidelity boosting where delta is adaptive, which subsequently calls `group_decode_adaptive_fileio.py`.(not complete yet)

experiments on NQ-Swap is the same.
The output will be saved in `output`. 

The conda environment we used can be found in `requirements.txt`. The main packages used are `pytorch`, `transformers`, and `accelerate`. 

### Notes
Baselines and other benchmarks haven't been added. The evaluation metrics also need to be extended.
baselines:Context-aware decoding (CAD), ADACAD, DECORE, etc.
benchmarks: XSum, truthQA，clasheval, ragtruth, refgpt, etc.
metrics: bert-score, exact match accuracy in QA, gpt evaluation, etc.

### Evaluation
After generating the prediction data, you can run the evaluation by`run_evaluation.sh` and compare with the gold data.  

