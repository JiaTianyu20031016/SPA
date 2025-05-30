eval "$(conda shell.bash hook)"

## inital DPO training
conda activate spa
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full_initial.yaml &
wait
sleep 30

## loop start
base_model=zephyr-2K
infer_model=save_model/zephyr-2K

for iteration in 1 2 3
do
    prompt_dir=datasets/spa_${iteration}
    sample_output_dir=datasets/sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/training-${base_model}-spa_${iteration}
    final_model_path=save_model/${base_model}-spa_${iteration}

    conda deactivate 
    conda activate vllm
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/online_generation.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir} &

    wait
    conda deactivate 
    conda activate spa
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py recipes/zephyr-7b-beta/dpo/config_full.yaml --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} &

    wait
    sleep 30
    python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} --select_num=30000 &

    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml --dataset_mixer=${judge_output_dir}  --model_name_or_path=${infer_model} --ref_model_for_refine=${infer_model} --output_dir=${final_model_path} &
    wait
    sleep 30
    # Update infer_model for the next iteration
    if [ ${iteration} -ne 10 ]; then
        infer_model=${final_model_path}
    fi
done