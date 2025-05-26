#!/bin/bash
gpu='3'
device="cuda:0"

model_name="./Qwen/Qwen2.5-7B-Instruct"
prompt_mode="1-shot"

max_steps=10 
max_length=64
temperature=1.0

for dataset in "alfworld" "babyai" "sc"
do
    for count in $(seq 0 5)
    do
        for fold_num_id in $(seq 0 3)
        do
        echo "Running inference with fold_num_id = ${fold_num_id}"
        CUDA_VISIBLE_DEVICES=${gpu} TOKENIZERS_PARALLELISM=false python sampling.py \
            --prompt_mode ${prompt_mode} \
            --save_path "./sampling" \
            --dataset_name ${dataset} \
            --model_name ${model_name} \
            --max_steps ${max_steps} \
            --max_length ${max_length} \
            --temperature ${temperature} \
            --fold_num ${fold_num_id} \
            --device ${device} \
            --do_sample \
            --sampling_count ${count} \
            --debug
        done
    done
done
