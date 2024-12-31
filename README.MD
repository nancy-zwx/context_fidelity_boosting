## Context Fidelity Boosting (CFB)

A novel approach for enhancing context fidelity in large language models through dynamic context optimization and boosting techniques.

### Overview

Context Fidelity Boosting (CFB) is a framework designed to improve the contextual understanding and response accuracy of large language models. It implements dynamic context optimization strategies and boosting techniques to enhance model performance in context-sensitive tasks.

### Input format
```json
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "huggyllama/llama-7b", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 0, // which GPU should take this instance
    "context_string": "The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 1078 episodes . How many episodes are in chicago fire season 4 ?", // the context-aware input
    "assigned_weight": 2, // weight for current instance/process (1+alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
    "filter_p": 1.0, // optional filtering for low-probablity tokens, disabled by default
}
{
    "input_index": 0, // instances that decode together should have the same input_index
    "assigned_model": "huggyllama/llama-7b", // same model for all instances in context-aware decoding, but can use different models here, e.g., DExperts, contrastive decoding, proxy tuning, etc.
    "assigned_process": 1, // which GPU should take this instance
    "context_string": "How many episodes are in chicago fire season 4 ?", // the context-unaware input
    "assigned_weight": -1, // weight for current instance/process (-alpha, weights should add up to 1 by default, but can also incorporate sampling temperature if needed)
}
...
```

### Running context-aware decoding on CNN-DM and NQ-Swap
for experiments on CNN_DM:
run `exp_cnndm.sh` for standard decoding or CAD baselines, which subsequently calls `group_decode_fileio.py`.
run `exp_cnndm_watermark.sh` for context fidelity boosting where delta is fixed, which subsequently calls `group_decode_watermark_fileio.py`.
run `exp_cnndm_adaptive.sh` for context fidelity boosting where delta is adaptive, which subsequently calls `group_decode_adaptive_fileio.py`.

experiments on NQ-Swap is the same.
The output will be saved in `output`. 

The conda environment we used can be found in `environment.yml`. The main packages used are `pytorch`, `transformers`, and `accelerate`. 

### Evaluation
After generating the prediction data, you can run the evaluation by running the following script and compare with the gold data.  
```bash
PRED_PATH=./output/llama7b/cnndm_1_0.jsonl.output_topp0.9_genlen100_boost10.0.jsonl
GOLD_DATA_PATH=./eval/cnndm_example_input/cnndm_1_0.jsonl
python eval/evaluate_summary.py --pred_path $PRED_PATH --data_path $GOLD_DATA_PATH
```
