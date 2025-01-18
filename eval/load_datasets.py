
import json
def load_dataset():
    import os,sys
    from datasets import load_dataset
    path = "google-research-datasets/natural_questions"
    subset = "dev"
    split = None
    dataset_name = path.split('/')[-1]
    dataset = load_dataset(path,subset,split=split,cache_dir=None)
    dataset.to_json(f"/Users/qianggao/project/intern/rag/dataset/{dataset_name}/{subset}/{split}.json")

    # for subset_name, subset_data in dataset.items():
    #     save_dir = f"../dataset/ragbench/{subset_name}"
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
        
    #     for split_name, split_data in subset_data.items():
    #         split_data.to_json(os.path.join(save_dir,f"{split_name}.json"))
    
def process():
    # 处理成论文需要的格式
    # {"input_index": 0, "assigned_model": "huggyllama/llama-7b", "assigned_process": 0, "context_string":, "gold_answers"
    file_path = "/Users/qianggao/project/intern/rag/dataset/hotpot_qa/hotpot_dev_fullwiki_v1.jsonl"
    data_list  = json.load(open(file_path))
    result = []
    for data in data_list:

        input_index = data["_id"]
        context_string = ""
        for context in data["context"]:
            context_string += context[0] + ":"+"".join(context[1])+"\n"


        result.append({
            "input_index": input_index,
            "assigned_model": "huggyllama/llama-7b",
            "assigned_process": 0,
            "context_string": context_string.strip()+" "+data['question'],
            "gold_answers": data["answer"]
        })
    # json.dump(result,open(file_path.replace(".json","_processed.json"),"w"),ensure_ascii=False)
    # jsonline
    with open(file_path.replace(".json","_processed.jsonl"),"w") as f:
        for item in result:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")



if __name__ == "__main__":
    load_dataset()
    # process()