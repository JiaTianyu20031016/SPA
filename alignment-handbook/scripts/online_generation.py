'''
Note:
This script is adapted from its single-process version (online_generation_no_multiprocessing.py). 
Specifically, we wrap the original core code into `generate` function, and use multiprocessing to call it on different GPUs in order to mannually realise parallel acceleration.
To use this script, make sure `script_args.my_world_size is equal to the total number of visible GPUs.
We keep script_args.tp equal to 1, so that tensor parallel is disabled.
If the model is too large to fit in a single GPU, you should use the original script (online_generation_no_multiprocessing.py) instead.
'''


import os

#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
import json
import multiprocessing as mp


def maybe_insert_system_message(messages):
    # confirm the jinja template refers to a system message before inserting
    messages.insert(0, {"role": "system", "content": ""})
    return messages


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=2,
        metadata={"help": "the number of generations per prompt"},
    )
    tp: Optional[int] = field(
        default=1,
        metadata={"help": "tensor_parallel_size"},
    )
    max_input_length: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [], 
        metadata={"help": "the ids of the end of sentence tokens"}
    )


def split_dataset(dataset, num_splits):
    import math
    total = len(dataset)
    size = math.ceil(total / num_splits)
    ds_list = []
    for i in range(num_splits):
        sub_ds = dataset.select(range(i * size, min((i + 1) * size, total)))
        ds_list.append(sub_ds)
    return ds_list


def generate(ds):
    import torch
    from vllm import LLM, SamplingParams
    
    model_path = script_args.model_name_or_path
    print("model_path", model_path)
    seed = script_args.seed
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side  = 'left'
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=script_args.K,
        stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
        seed=seed
        #stop=["<|user|>"],
    )
    
    original_prompt = ds['prompt']
    index = ds['index']
    sys_prompt = None
    print(ds)
    ds = ds.map(
    lambda x: {
        "prompt": tokenizer.apply_chat_template(maybe_insert_system_message(x["chosen"][:-1]), tokenize=False, add_generation_prompt=True)
    }
    )
    prompt_chat_tem = ds.map(
        lambda x: 
            {"prompt_chat_tem": x["chosen"][:-1]})
        
    data_size = len(ds["prompt"])
    prompts = ds["prompt"]

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=seed,
        tensor_parallel_size=1,
        # ray_workers_use_nsight=True, distributed_executor_backend="ray"
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    completions = []
    used_prompts = []
    new_dataset = {}
    new_dataset['prompt'] = []
    new_dataset['response'] = []
    new_dataset['original_prompt'] = []
    new_dataset['truncated'] = []
    new_dataset['index'] = []

    for i, output in enumerate(outputs):
        tmp_data = {"prompt": prompts[i], "response": [out.text for out in output.outputs] ,"original_prompt" :original_prompt[i]}
        new_dataset['prompt'].append(original_prompt[i])
        new_dataset['response'].append([out.text for out in output.outputs])
        new_dataset['truncated'].append([len(out.token_ids) == script_args.max_new_tokens for out in output.outputs])
        new_dataset['original_prompt'].append(original_prompt[i])
        new_dataset['index'].append(index[i])
        for out in output.outputs:
            assert len(out.token_ids) <= script_args.max_new_tokens
        # print([len(out.token_ids) == script_args.max_new_tokens for out in output.outputs])
        # print([out.text for out in output.outputs])

    # print(new_dataset)

    from datasets import Dataset,DatasetDict
    new_dataset = Dataset.from_dict(new_dataset)

    return new_dataset
    

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


def worker_process(proc_id, ds, return_queue: mp.Queue):
    # map each process to its corresponding GPU
    assert proc_id < script_args.my_world_size
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_ids[proc_id])
    new_dataset = generate(ds)
    return_queue.put(new_dataset)
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    visible_gpu_ids = get_physical_gpu_ids()
    #print(visible_gpu_ids)
    print(f"Visible devices: {visible_gpu_ids}")

    if script_args.dataset_name_or_path.split("/")[0] == 'vangard703' :
        ds = load_dataset(script_args.dataset_name_or_path, split="train")
        ds_test = load_dataset(script_args.dataset_name_or_path, split="test")
    else :
        ds = load_from_disk(script_args.dataset_name_or_path)['train']
        ds_test = load_from_disk(script_args.dataset_name_or_path)['test']

    from datasets import Dataset, DatasetDict
    def add_index_to_dataset(dataset: Dataset) -> Dataset:
        def add_index(example, idx):
            example['index'] = idx
            return example
        
        return dataset.map(add_index, with_indices=True)

    if not "index" in ds.features :
        ds = add_index_to_dataset(ds)

    ds_list = split_dataset(ds, num_splits=script_args.my_world_size)
    processes = []
    queues = []
    
    for proc_id, sub_ds in enumerate(ds_list):
        q = mp.Queue()
        p = mp.Process(
            target=worker_process,
            args=(proc_id, sub_ds, q),
        )
        p.start()
        processes.append(p)
        queues.append(q)

    result_datasets = []
    for q in queues:
        result = q.get()  # blocks until child puts
        result_datasets.append(result)

    for p in processes:
        p.join()
    
    merged = concatenate_datasets(result_datasets)
    
    assert [item['prompt'] for item in merged] == [item['prompt'] for item in ds]
    
    def del_base_len(sample) :
        return len(sample['response']) > (script_args.K -1)

    merged = merged.filter(del_base_len)
    save_dataset = DatasetDict({
        "train": merged,
        "test": ds_test
    })

    save_path = script_args.output_dir
    save_dataset.save_to_disk(save_path)