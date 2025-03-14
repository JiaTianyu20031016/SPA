#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import Dataset, DatasetDict
import json
from tqdm import tqdm

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)


if script_args.dataset_name_or_path.split("/")[0] == 'vangard703' :
    ds = load_dataset(script_args.dataset_name_or_path, split="train")
    ds_test = load_dataset(script_args.dataset_name_or_path, split="test")
else :
    ds = load_from_disk(script_args.dataset_name_or_path)['train']
    ds_test = load_from_disk(script_args.dataset_name_or_path)['test']


def add_index_to_dataset(dataset: Dataset) -> Dataset:
    def add_index(example, idx):
        example['index'] = idx
        return example
    
    return dataset.map(add_index, with_indices=True)

if "index" not in ds.features :
    ds = add_index_to_dataset(ds)

original_prompt = ds['prompt']
index = ds['index']
data_size = len(ds["prompt"])
prompts = ds["prompt"]


completions = []
used_prompts = []
new_dataset = {}
new_dataset['prompt'] = []
new_dataset['response'] = []
new_dataset['original_prompt'] = []
new_dataset['index'] = []

for item in tqdm(ds, desc='processing original data'):
    new_dataset['prompt'].append(item['prompt'])
    new_dataset['response'].append(
        [
            item['chosen'][-1]['content'],
            item['rejected'][-1]['content'],
        ]
    )
    new_dataset['original_prompt'].append(item['prompt'])
    new_dataset['index'].append(item['index'])

new_dataset = Dataset.from_dict(new_dataset)

save_dataset = DatasetDict({
    "train": new_dataset,
    "test": ds_test
})
print(save_dataset)

save_path = script_args.output_dir
save_dataset.save_to_disk(save_path)