# Spread Preference Annotation (SPA)

This repository contains the implementation of **Spread Preference Annotation (SPA)**, as described in the paper:

SPA is a framework that aligns large language models (LLMs) to human preferences using only a small amount of annotated data. It leverages direct preference judgment and introduces self-refinement methods to mitigate noise in self-generated data, significantly reducing the reliance on extensive human annotations.

# Experiment Setup and Execution Guide

This guide provides detailed instructions on setting up and running experiments with the specified environment and configurations.

## Condition
- CUDA: 12.1.1
- PyTorch: 2.1.2

## Setup Script

1. **Create and activate the conda environment**:
    ```bash
    conda create -n vllm python=3.10
    conda activate vllm
    pip install vllm datasets
    conda deactivate

    conda create -n spa python=3.10
    conda activate spa
    ```

2. **Install CUDA toolkit (if necessary)**:
    ```bash
    conda install nvidia/label/cuda-12.1.1::cuda-toolkit
    ```

3. **Install PyTorch and related packages**:
    ```bash
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. **Install additional required packages**:
    ```bash
    pip install -e trl/.
    pip install -e alignment-handbook/.
    pip install transformers==4.36.2
    pip install numpy==1.26.4
    pip install torchmetrics
    pip install huggingface-hub==0.24.7
    ```

5. **Navigate to the alignment-handbook directory**:
    ```bash
    cd alignment-handbook
    ```

## Data Preparation

1. **Split the data**:
    ```bash
    python make_data_split.py
    ```

## Model Training
    ```bash
    bash scripts/run_all.sh
    ```

## Notes

- The current setup is based on the **zephyr-7b-beta** model. Sampling code may need to be adjusted for different models or chat templates.

- Models trained to align with the current setup include:
  - `princeton-nlp/Llama-3-Base-8B-SFT`
  - `alignment-handbook/zephyr-7b-sft-full`

- For initial DPO training:
  - Using **3 epochs** is effective for datasets with fewer than 2,000 samples.
  - For larger datasets, even if initial performance is lower, **1 epoch** training has been observed to yield better trends.

- Hyperparameters can be customized using the configuration files:
  - `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_full_initial.yaml`
  - `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_full.yaml`
