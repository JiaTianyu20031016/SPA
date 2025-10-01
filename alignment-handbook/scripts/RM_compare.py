import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
from datasets import Dataset, DatasetDict
import torch
from typing import Dict, List
import llm_blender
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import multiprocessing as mp

import math



class ArmoRMPipeline:
    def __init__(self, model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1", device_map='cuda', torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        # self.accelerator = Accelerator()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.tokenizer.padding_side = "left"  # For LLaMA-style decoder models
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages_list: List[List[Dict[str, str]]]) -> List[Dict[str, float]]:
        """
        messages_list: List of OpenAI-format chat messages.
        Each element is a list of {"role": ..., "content": ...} dicts.
        
        Returns: list of scores between 0 and 1.
        """

        # Step 1: Convert messages to string using chat template
        prompts = [self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_list]

        # Step 2: Tokenize with left-padding and truncation
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)

        # Step 3: Forward pass with attention_mask
        with torch.no_grad():
            output = self.model(**encoded)
            scores = output.score.float().tolist()

        return scores
        
        '''scores = []
        for messages in messages_list:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(input_ids)
                score = output.score.float().item()
            scores.append(score)'''
            
        return scores
 

def rank(inputs, candidates, batch_size=4, mode='llm-blender', device='cuda', return_scores=False):
    if mode == 'llm-blender':
        #for split in full_ds.split
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM", device=device) # load ranker checkpoint
        scores = blender.rank(inputs, candidates, return_scores=True, batch_size=batch_size).tolist()
        ranks = []
        for sub_scores in scores:
            sub_ranks = len(sub_scores) - np.argsort(sub_scores)
            ranks.append(sub_ranks.tolist())
        
    elif mode == 'ArmoRM':
        pipeline = ArmoRMPipeline(device_map=device)
        message_list = []
        for input, candidate in zip(inputs, candidates):
            message_list += [
                [
                    {
                        'role': 'user',
                        'content': input
                    },
                    {
                        'role': 'assistant',
                        'content': cand_i
                    },
                ]
                for cand_i in candidate
            ]
            
        flatten_scores = []
        for start in tqdm(range(0, len(message_list), batch_size)):
            end = min(start+batch_size, len(message_list))
            flatten_scores += pipeline(message_list[start: end])
        
        ranks = []
        scores = []
        start = 0
        for input, candidate in zip(inputs, candidates):
            end = start + len(candidate)
            sub_scores = flatten_scores[start: end]
            sub_ranks = len(sub_scores) - np.argsort(sub_scores)
            ranks.append(sub_ranks.tolist())
            scores.append(sub_scores)
            start = end
            
    else:
        raise(NotImplementedError(f'Unsupported mode: {mode}'))
    
    if return_scores:
        return scores, ranks
    else:
        return ranks


def get_physical_gpu_ids():
    import re
    # 获取环境变量 CUDA_VISIBLE_DEVICES 的值
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    
    if not cuda_visible_devices:
        # 环境变量未设置，返回所有可用的物理GPU编号
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        except ImportError:
            try:
                from tensorflow.python.client import device_lib
                devices = device_lib.list_local_devices()
                gpu_ids = [int(re.search(r'GPU:(\d+)', d.name).group(1)) 
                          for d in devices if d.device_type == 'GPU']
                return sorted(gpu_ids)
            except ImportError:
                raise RuntimeError("Neither PyTorch nor TensorFlow is available to detect GPUs")
    
    # 处理环境变量中的逗号分隔值
    parts = cuda_visible_devices.split(',')
    gpu_ids = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # 处理连续范围的表示 (如 2-5)
        if '-' in part:
            start, end = part.split('-')
            try:
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                gpu_ids.extend(range(start_idx, end_idx + 1))
            except ValueError:
                raise ValueError(f"Invalid GPU range format: {part}")
        else:
            try:
                gpu_ids.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid GPU ID format: {part}")
    
    return gpu_ids


def worker_process(device, kwargs, return_queue: mp.Queue):
    # map each process to its corresponding GPU
    pack = rank(device=device, **kwargs)
    return_queue.put(pack)
    

def multiprocess_rank(inputs, candidates, batch_size=4, mode='llm-blender', proc_num=None, return_scores=False):
    visible_gpu_ids = get_physical_gpu_ids()
    print( f"Visible GPUs: {visible_gpu_ids}")
    split = proc_num if proc_num else len(visible_gpu_ids)
    assert split <= len(visible_gpu_ids)
    
    batch = math.ceil(len(inputs)/split)
    inputs_list = [inputs[i:i+batch] for i in range(0, len(inputs), batch)]
    candidates_list = [candidates[i:i+batch] for i in range(0, len(inputs), batch)]

    processes = []
    queues = []
    for proc_id, (sub_inputs, sub_candidates) in enumerate(zip(inputs_list, candidates_list)):
        q = mp.Queue()
        p = mp.Process(
            target=worker_process,
            args=(
                f'cuda:{proc_id}',
                {
                    'inputs': sub_inputs, 
                    'candidates': sub_candidates, 
                    'batch_size': batch_size, 
                    'mode': mode,
                    'return_scores': return_scores
                }, 
                q
            ),
        )
        p.start()
        processes.append(p)
        queues.append(q)

    pack = []
    for q in queues:
        pack.append(q.get())  # blocks until child puts

    for p in processes:
        p.join()
    
    if return_scores:
        ranks = []
        scores = []
        for score, rank in pack:
            ranks += rank
            scores += score
        return scores, ranks
    else:
        ranks = []
        for rank in pack:
            ranks += rank
        return ranks
    
    
def GPT4_compare(path, mode="ArmoRM"):
    import json
    with open(os.path.join(path, 'model_outputs.json'), 'r') as f:
        model_outputs = json.load(f)
    with open(os.path.join(path, 'reference_outputs.json'), 'r') as f:
        ref_outputs = json.load(f)
    inputs = [ item['instruction'] for item in model_outputs ]
    candidates = [ 
                  [model['output'], ref['output']] for model, ref in zip(model_outputs, ref_outputs)
                ]
    ranks = multiprocess_rank(inputs, candidates, batch_size=2, mode=mode)
    comparison_results = [(r[0] == 1) for r in ranks]
    
    model_name = model_outputs[0]['generator']
    ref_name = ref_outputs[0]['generator']
    precision = sum(comparison_results) / len(comparison_results)
    print(f'{mode} winrate for model {model_name} against model {ref_name}: {precision}')
    return comparison_results, precision
    
    
if __name__ == '__main__':
    #full_ds = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized')
    full_ds = datasets.load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned')
    visible_gpu_ids = get_physical_gpu_ids()
    full_ds_dict = { column: full_ds[column] for column in full_ds}
    
    for name in full_ds:
        
        ds = full_ds[name]
        
        inputs = [ item[0]['content'] for item in ds['chosen'] ]
        candidates = [ [item['chosen'][1]['content'][:1200], item['rejected'][1]['content'][:1200]] for item in ds]
        
        ranks = multiprocess_rank(inputs, candidates, batch_size=2, mode='ArmoRM')
        
        #ranks = compare(inputs, candidates, mode='ArmoRM', batch_size=2)
        comparison_results = [(r[0] == 1) for r in ranks]

        ds = ds.to_dict()
        for i in range(len(comparison_results)):
            if not comparison_results[i]:
                for column in ds.keys():
                    if 'chosen' in column:
                        temp = ds[column][i]
                        ds[column][i] = ds[column.replace('chosen', 'rejected')][i]
                        ds[column.replace('chosen', 'rejected')][i] = temp
        ds = Dataset.from_dict(ds)
        
        precision = sum(comparison_results) / len(comparison_results)
        print(f'precision of split {name}: {precision}')
        full_ds_dict[name] = ds

    save_path = '/root/jiaty/projects/SPA/ultrafeedback-binarized-preferences-cleaned-corrected'
    full_ds = DatasetDict(full_ds_dict)
    #full_ds.save_to_disk(save_path)