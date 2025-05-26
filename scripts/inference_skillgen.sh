#!/bin/bash
gpu='3'
device="cuda:0"
model="./Qwen/Qwen2.5-7B-Instruct"
max_steps=20 
max_length=64
temperature=0.0 
seed=42 


for prompt_mode in 'skillgen'  
do
    for dataset in "alfworld" "babyai" "sc"
    do   
        for fold_num_id in $(seq 0 3)
        do
        echo "Running inference with fold_num_id = ${fold_num_id}"
        CUDA_VISIBLE_DEVICES=${gpu} TOKENIZERS_PARALLELISM=false python inference_skillgen.py \
            --save_path "./log/inference_${prompt_mode}" \
            --dataset_name ${dataset} \
            --model_name ${model} \
            --max_steps ${max_steps} \
            --max_length ${max_length} \
            --temperature ${temperature} \
            --seed ${seed} \
            --fold_num ${fold_num_id} \
            --prompt_mode ${prompt_mode} \
            --top_ac 1 \
            --top_s 1 \
            --device ${device} \
            --hist_size 20 \
            --sampling_count 6 \
            --debug
        done
    done
done
