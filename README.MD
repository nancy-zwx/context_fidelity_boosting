## Context Fidelity Boosting (CFB)

A novel approach for enhancing context fidelity in large language models through dynamic context optimization and boosting techniques.

### Overview

Context Fidelity Boosting (CFB) is a framework designed to improve the contextual understanding and response accuracy of large language models. It implements dynamic context optimization strategies and boosting techniques to enhance model performance in context-sensitive tasks.

### Boosting Modes

CFB supports three modes of boosting:

1. Static Boosting
- Uses fixed delta value for all context tokens
- Suitable for tasks with uniform context importance
- Run with `group_decode_watermark_fileio.py`

2. Global Adaptive Boosting
- Adjusts boosting strength globally based on context-query distribution difference
- Uses Jensen-Shannon divergence to measure distribution difference
- Delta value ranges from min_delta to max_delta
- Suitable for general context enhancement
- Run with `group_decode_adaptive_fileio.py`

3. Token-wise Adaptive Boosting
- Considers per-token importance through attention scores and semantic similarity
- Combines attention weights (λ1) and semantic similarity weights (λ2)
- Better for precise context utilization
- Run with `group_decode_adaptive_fileio.py`

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
run `exp_cnndm_watermark.sh` for context fidelity boosting where delta is fixed, which subsequently calls `group_decode_watermark_fileio.py`. 
Key Parameters:
context_boost_delta: Fixed boosting value for static mode

run `exp_cnndm_adaptive.sh` for context fidelity boosting where delta is adaptive, which subsequently calls `group_decode_adaptive_fileio.py`.
Key Parameters:
min_delta: Minimum boosting value for adaptive modes (default: 2.0)
max_delta: Maximum boosting value for adaptive modes (default: 10.0)
use_global: Whether to use global mode ("true") or token-wise mode ("false")
lambda1: Weight for attention scores in token-wise mode (default: 0.6)
lambda2: Weight for semantic similarity in token-wise mode (default: 0.4)

experiments on NQ-Swap is the same.
The output will be saved in `output`. 

The conda environment we used can be found in `requirements.txt`. The main packages used are `pytorch`, `transformers`, and `accelerate`. 


### Evaluation
After generating the prediction data, you can run the evaluation by`run_evaluation.sh` and compare with the gold data.  

