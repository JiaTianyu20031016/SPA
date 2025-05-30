eval "$(conda shell.bash hook)"

## inital DPO training
conda activate spa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b-instruct/dpo/config_full_initial.yaml &
wait
sleep 30

## loop start
base_model=llama3-2K
infer_model=save_model/llama3-2K
past_training_set=datasets/spa_0

for iteration in 1 2 3
do
    prompt_dir=datasets/spa_${iteration}
    sample_output_dir=datasets/sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/training-${base_model}-spa_${iteration}
    final_model_path=save_model/${base_model}-spa_${iteration}
    past_training_set=${past_training_set},${judge_output_dir}

    python scripts/prepare_original_data.py --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir} &

    wait
    conda deactivate 
    conda activate spa
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py recipes/llama3-8b-instruct/dpo/config_full.yaml --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} &

    wait
    sleep 30
    python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} --select_num=30000 &
    
    wait
    sleep 30
    # Train from sft model, on all past training data
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/llama3-8b-instruct/dpo/config_full.yaml --dataset_mixer=${past_training_set} --output_dir=${final_model_path} &
    wait
    sleep 30
    # Update infer_model for the next iteration
    if [ ${iteration} -ne 10 ]; then
        infer_model=${final_model_path}
    fi
done