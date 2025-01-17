# -*- encoding: utf-8 -*-
'''
@File    :   statistic.py
@Time    :   2025/01/16 
@Author  :   tensorgao 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
@Desc    :   None
'''
import os

# statistics about delta for 1-50 
def statistic(file_path):
    import re
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    average_fact_score = re.search(r'Average fact score: ([\d.]+)', content)
    rouge1 = re.search(r'rouge1\': ([\d.]+)', content)
    rouge2 = re.search(r'rouge2\': ([\d.]+)', content)
    rougeL = re.search(r'rougeL\': ([\d.]+)', content)

    if average_fact_score is None or rouge1 is None or rouge2 is None or rougeL is None:
        raise Exception("Error in reading file")
    return float(average_fact_score.group(1)), float(rouge1.group(1)), float(rouge2.group(1)), float(rougeL.group(1))

def draw_line_chat(dataset:str,average_fact_score_list:list, rouge1_list:list, rouge2_list:list, rougeL_list:list):
    # draw line chat 1-50 for horizontal axis, average_fact_score, rouge1, rouge2, rougeL for vertical axis
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(1, len(average_fact_score_list)+1, 1)
    plt.xticks(x)  # Ensure every x-axis position is labeled
    plt.grid(axis='x', linestyle='--', linewidth=0.5, which='both')  # Add 'which' parameter to ensure grid lines at every x position

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(x, average_fact_score_list, label='average_fact_score', color='b')
    axs[0, 0].set_title('Average Fact Score')
    axs[0, 0].set_xlabel('Delta')
    axs[0, 0].set_ylabel('Score')

    axs[0, 1].plot(x, rouge1_list, label='rouge1', color='g')
    axs[0, 1].set_title('Rouge1')
    axs[0, 1].set_xlabel('Delta')
    axs[0, 1].set_ylabel('Score')

    axs[1, 0].plot(x, rouge2_list, label='rouge2', color='r')
    axs[1, 0].set_title('Rouge2')
    axs[1, 0].set_xlabel('Delta')
    axs[1, 0].set_ylabel('Score')

    axs[1, 1].plot(x, rougeL_list, label='rougeL', color='c')
    axs[1, 1].set_title('RougeL')
    axs[1, 1].set_xlabel('Delta')
    axs[1, 1].set_ylabel('Score')

    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    fig.suptitle(f'Dataset: {dataset}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()
    # save to file
    fig.savefig(f"{dataset}.png")

def process(file_path):
    # open jsonl file
    import json
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]
    new_result = []
    for index,data in enumerate(data_list):
        if index%2==0:
            new_result.append(data)
    
    # write to new file
    with open(file_path, 'w', encoding='utf-8') as file:
        for data in new_result:
            file.write(json.dumps(data)+'\n')


if __name__ == '__main__':
    
    # dataset = 'cnndm_1_0'
    # file_dir = "/home/gaoqiang/context_fidelity_boosting/results/cnn/llama2-7b"

    dataset = 'nqswap_1_0'
    file_dir = "/home/gaoqiang/context_fidelity_boosting/results/nq/llama2-7b"
    average_fact_score_list = []
    rouge1_list = []
    rouge2_list = []
    rougeL_list = []

    files = [file for file in os.listdir(file_dir) if "evaluate" in file]

    # Sort files numerically and alphabetically
    sorted_files = sorted(files, key=lambda x: float(x.split('_')[-1].replace('.log', '').split('boost')[1]))

    for file in sorted_files:
            delta = file.split('_')[-1].split('.')[0]

            file_path = os.path.join(file_dir, file)
            average_fact_score, rouge1, rouge2, rougeL = statistic(file_path)
            average_fact_score_list.append(average_fact_score)
            rouge1_list.append(rouge1)
            rouge2_list.append(rouge2)
            rougeL_list.append(rougeL)
            print(f'delta:{delta},{file} average_fact_score: {average_fact_score}, rouge1: {rouge1}, rouge2: {rouge2}, rougeL: {rougeL}')
    draw_line_chat(dataset, average_fact_score_list, rouge1_list, rouge2_list, rougeL_list)