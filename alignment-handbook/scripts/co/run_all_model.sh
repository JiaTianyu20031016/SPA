eval "$(conda shell.bash hook)"

base_model_0=zephyr-2K
base_model_1=llama3-2K
recipe_0=recipes/zephyr-7b-beta/dpo/config_full.yaml
recipe_1=recipes/llama3-8b-sft/dpo/config_full.yaml
infer_model_0=save_model/0-zephyr-2K
infer_model_1=save_model/1-llama3-2K

## inital DPO training
conda activate spa
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full_initial.yaml --output_dir=${infer_model_0}&
wait
sleep 30

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b-sft/dpo/config_full_initial.yaml --output_dir=${infer_model_1}&
wait
sleep 30

## loop start
for iteration in 1 2 3
do
    prompt_dir=datasets/spa_${iteration}

    # role 0
    base_model=${base_model_0}
    sample_output_dir=datasets/0-sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/0-training-${base_model}-spa_${iteration}
    final_model_path=save_model/0-${base_model}-spa_${iteration}

    conda deactivate 
    conda activate vllm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/online_generation.py --model_name_or_path ${infer_model_0} --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir}&
    
    wait
    conda deactivate 
    conda activate spa
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py ${recipe_1} --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model_1} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} &
    # for observation
    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py ${recipe_0} --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model_0} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir}-obs &
    
    wait
    sleep 30
    python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} --select_num=30000 &

    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py ${recipe_0} --dataset_mixer=${judge_output_dir}  --model_name_or_path=${infer_model_0} --ref_model_for_refine=${infer_model_0} --output_dir=${final_model_path} &
    wait
    sleep 30

    # role 1
    base_model=${base_model_1}
    sample_output_dir=datasets/1-sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/1-training-${base_model}-spa_${iteration}
    final_model_path=save_model/1-${base_model}-spa_${iteration}

    conda deactivate 
    conda activate vllm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/online_generation.py --model_name_or_path ${infer_model_1} --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir} &

    wait
    conda deactivate 
    conda activate spa
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py ${recipe_0} --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model_0} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} &
    # for observation
    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py ${recipe_1} --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model_1} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir}-obs &

    wait
    sleep 30
    python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} --select_num=30000 &
    
    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py ${recipe_1} --dataset_mixer=${judge_output_dir}  --model_name_or_path=${infer_model_1} --ref_model_for_refine=${infer_model_1} --output_dir=${final_model_path} &
    wait
    sleep 30

    # Update infer_model for the next iteration
    infer_model_0=save_model/0-${base_model_0}-spa_${iteration}
    infer_model_1=save_model/1-${base_model_1}-spa_${iteration}
done