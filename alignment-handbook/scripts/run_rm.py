from typing import List, Optional, Literal
from dataclasses import dataclass, field
import sys, os
from datasets import Dataset, load_from_disk, load_dataset
from RM_compare import multiprocess_rank
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_confidence_name', default="save_data", type=str, required=False)
parser.add_argument('--dataset_mixer', default="datasets/spa_0", type=str, required=False)
parser.add_argument('--mode', default="ArmoRM", type=str, required=False)

def prepare_pairwise_data(dataset):
    """Convert (index, prompt, response) format to (index, prompt, chosen, rejected) pairs"""
    processed_data = []
    
    # Group by prompt to get all responses for each prompt
    grouped_data = {}
    for item in dataset:
        if item['index'] not in grouped_data:
            grouped_data[item['index']] = []
        grouped_data[item['index']] = {
            'prompt': item['prompt'],
            'response': item['response'],
            'truncated': item['truncated'] if 'truncated' in item.keys() else [False] * len(item['response'])
        }

    
    # Create pairs for each group
    for index, responses in grouped_data.items():
        n_responses = len(responses['response'])
        # For even number of responses
        for i in range(0, n_responses - 1, 2):
            processed_data.append({
                'index': f"{i}_{i+1}_{index}",
                'prompt': responses['prompt'],
                'chosen':   responses['response'][i],
                'rejected': responses['response'][i+1],
                'is_chosen_truncated':  responses['truncated'][i], 
                'is_rejected_truncated':  responses['truncated'][i+1],             
            })
            
        # For odd number of responses, use the last response twice if needed
        if n_responses % 2 == 1:
            processed_data.append({
                'index': f"-1_{0}_{index}",
                'prompt': responses['prompt'],
                'chosen':   responses['response'][-1],
                'rejected': responses['response'][0],
                'is_chosen_truncated':  responses['truncated'][-1], 
                'is_rejected_truncated':  responses['truncated'][0],             
            })

    return processed_data


if __name__ == '__main__':
    
    script_args = parser.parse_args()
        
    ###############
    # Load datasets
    ###############
    raw_datasets = load_from_disk(script_args.dataset_mixer)
    column_names = list(raw_datasets["train"].features)
    column_names.remove("index")
    
    initial_training_raw = raw_datasets['train']
    initial_testing_raw = raw_datasets['test']
    print("raw_datasets", raw_datasets)
    
    processed_data = prepare_pairwise_data(initial_training_raw)
    
    scores, ranks = multiprocess_rank(
        inputs= [ i['prompt'] for i in processed_data ],
        candidates=[ [i['chosen'], i['rejected']] for i in processed_data ],
        batch_size=2,
        mode=script_args.mode,
        return_scores=True
    )
    assert len(processed_data) == len(scores)
    data = { processed_data[i]['index']: [scores[i][0], scores[i][1]] for i in range(len(scores))}
    
    save_confidence_name = script_args.save_confidence_name.split("/")[1] + ".json" 
    with open(os.path.join('save_confidence', save_confidence_name), 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Reward Model Score saved to {os.path.join('save_confidence', save_confidence_name)}")
    
    
    