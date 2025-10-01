from datasets import load_dataset, Dataset, DatasetDict
import datasets
import random

#original_data = datasets.load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
original_data = datasets.load_from_disk("/root/jiaty/projects/SPA/datasets/ultrafeedback-binarized-preferences-cleaned-corrected-Armo")
def add_index(example, idx):
    example['index'] = idx
    return example
original_data = original_data.map(add_index, with_indices=True)
suffle = original_data['train'].shuffle(seed=42)

train_data_2K = suffle[:2000]
train_data_8K = suffle[2000:10000]
train_data_20K = suffle[10000:30000]
train_data_30K = suffle[30000:60000]
test_data = suffle[60000:]


train_data_2K = Dataset.from_dict(train_data_2K)
train_data_8K = Dataset.from_dict(train_data_8K)
train_data_20K = Dataset.from_dict(train_data_20K)
train_data_30K = Dataset.from_dict(train_data_30K)
test_data = Dataset.from_dict(test_data)
new_dataset_2K = DatasetDict({
    "train": train_data_2K,
    "test": test_data,
})

new_dataset_8K = DatasetDict({
    "train": train_data_8K,
    "test": test_data,
})

new_dataset_20K = DatasetDict({
    "train": train_data_20K,
    "test": test_data,
})
new_dataset_30K = DatasetDict({
    "train": train_data_30K,
    "test": test_data,
})

save_path = "datasets/spa_0"
new_dataset_2K.save_to_disk(save_path)

save_path = "datasets/spa_1"
new_dataset_8K.save_to_disk(save_path)

save_path = "datasets/spa_2"
new_dataset_20K.save_to_disk(save_path)

save_path = "datasets/spa_3"
new_dataset_30K.save_to_disk(save_path)
