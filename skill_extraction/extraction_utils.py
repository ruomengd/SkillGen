import json
from json import dumps
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict

# Load variables from .env file
load_dotenv()
# Access the API key
gpt_api_key = os.getenv("OPENAI_API_KEY")
qwen_api_key = os.getenv("QWEN_API_KEY")



def get_task_category(item, dataset_name):
    """Extract task category based on dataset type"""
    if dataset_name == 'sc':
        return item["additional_info"]["env_name"]
    elif dataset_name == 'alfworld':
        return item["additional_info"]["description"].split('-')[0].strip()
    elif dataset_name == 'babyai':
        return item["additional_info"]["subtask"]
    
def get_task_id(item, dataset_name):
    """Extract task ID based on dataset type"""
    if dataset_name == 'sc':
        return f"{item['additional_info']['env_name']}_{item['additional_info']['var']}"
    elif dataset_name == 'alfworld':
        return item["additional_info"]["description"]
    elif dataset_name == 'babyai':
        return item["id"]

def process_trajectories(valid_trajs):
    """Process and clean trajectories"""
    trajectories = []
    for data in valid_trajs:
        actions = ['INIT_STATE'] + data['actions'] + ['FINAL_STATE']
        obs = ['INIT_OBS', data['init_obs']] + data['observations']
        progress = [0] + data['progress'] + [data['progress'][-1]]
        
        # Remove self-cycles while preserving unique state transitions
        unique_actions, unique_obs, unique_progress = actions[:2], obs[:2], progress[:2]
        for idx in range(2, len(actions)):
            if actions[idx] != actions[idx-1]:
                unique_actions.append(actions[idx])
                unique_obs.append(obs[idx])
                unique_progress.append(progress[idx])
                
        trajs = list(zip(unique_obs, unique_actions, unique_progress))
        trajectories.append({
            'task_name': data['task_name'],
            'sample_name': data['sample_name'],
            'goal': data['goal'],
            'trajs': trajs
        })
    return trajectories


def transform_dict(d):
    sorted_keys = sorted(d.keys())  # Sort keys
    prev_value = 0  # Initialize previous value
    new_d = {}
    for key in sorted_keys:
        new_d[key] = d[key] - prev_value
        prev_value = d[key]  # Update previous value
    
    return new_d


def extract_res(dataset_name, count, model_name, id_list):
    """
    Loads and filters trajectory data from JSONL files based on dataset type and model.
    
    Args:
        dataset_name (str): Name of dataset ('alfworld', 'babyai', or 'sc')
        count (int): Number of sampling iterations
        model_name (str): Name of model used for generation
        id_list (list): List of task IDs to filter by
        
    Returns:
        tuple: Lists of grounded trajectories and trajectories with positive progress
    """
    datalist = []
    model_name = model_name.split('/')[-1]

    # Load data from files based on model type
    for c in range(count):
        for fold in range(4):
            filename =  f'./logs/sampling/{dataset_name}/{model_name}_step10_1-shot_sampling{c}_maxlen64_temp1.0_topp0.95_hist20_fold{fold}.jsonl'
            with open(filename, 'r', encoding='utf-8') as file:
                datalist.extend([json.loads(line.strip()) for line in file])

    print(f'Total trajectories loaded: {len(datalist)}')

    # Filter trajectories based on dataset-specific criteria
    g_list, progress_list = [], []
    for data in datalist:
        task_match = False
        
        if 'task_uid' in data:
            task_match = data['task_uid'] in id_list
        else:
            if dataset_name == 'alfworld':
                task_match = data['task_name'] in id_list
            elif dataset_name == 'babyai':
                task_match = data['task_id'] in id_list
            elif dataset_name == 'sc':
                task_match = f"{data['task_name']}_{data['var']}" in id_list

        if task_match and len(data['grounding']) > 0:
            g_list.append(data)
            if data['progress_rate'] > 0:
                progress_list.append(data)

    return g_list, progress_list


def remove_numbers(s):
    return re.sub(r'\d', '', s)

def keep_one_of_duplicates_dicts(lst):
    seen = set()
    result = []
    for d in lst:
        serialized = dumps(d, sort_keys=True)  # Convert to string for hashability
        if serialized not in seen:
            seen.add(serialized)
            result.append(d)  # Keep the first occurrence
    return result
    
def collect_valid_traj(p_list_drop_duplicates, dataset_name=None):
    """
    Collects and processes valid trajectories from a list of samples.
    
    Args:
        p_list_drop_duplicates (list): List of deduplicated trajectory samples
        dataset_name (str, optional): Name of dataset ('sc', 'alfworld', or 'babyai')
        
    Returns:
        list: List of processed trajectory dictionaries with standardized format
    """
    res_list = []
    unique_id = 0
    
    for sample in p_list_drop_duplicates:
        # Extract trajectory components
        traj = sample['trajectory']
        grounding_idx = sample['grounding']
        progress_dict = dict(sample['progress'])
        progress_idx = list(progress_dict.keys())
        
        # Create dense progress list and fill gaps
        dense_progress_list = [progress_dict.get(i, 0) for i in range(10)]
        for idx in range(1, len(dense_progress_list)):
            if dense_progress_list[idx] == 0 and dense_progress_list[idx-1] > 0:
                dense_progress_list[idx] = dense_progress_list[idx-1]

        # Extract observations and actions
        init_obs = traj[0][1]
        obs_lst = [item[1] for item in traj[1:] if item[0] == 'OBSERVATION']
        ac_lst = [item[1].lower() for item in traj[1:] if item[0] == 'ACTION']

        # Filter grounded steps
        grounding_action_lst = []
        grounding_obs_lst = []
        grounding_progress_lst = []
        
        for idx, action in enumerate(ac_lst):
            if idx in grounding_idx or idx in progress_idx:
                grounding_action_lst.append(action)
                grounding_obs_lst.append(obs_lst[idx])
                grounding_progress_lst.append(dense_progress_list[idx])

        # Process task and sample names based on dataset
        if dataset_name == 'sc':
            task_name = sample['task_name']
            sample_name = f"{task_name}_{sample['var']}" if 'var' in sample else sample['task_uid']
        
        elif dataset_name == 'alfworld':
            task_name = sample['task_name'].split('-')[0].strip()
            sample_name = sample['task_name']
            
        elif dataset_name == 'babyai':
            task_name = sample['task_name']
            id_field = 'task_id' if 'task_id' in sample else 'task_uid'
            sample_name = f"{task_name}_{sample[id_field]}"

        # Create standardized trajectory dictionary
        new_sample = {
            "unique_id": unique_id,
            "task_name": task_name,
            "sample_name": sample_name,
            "num_step": len(grounding_action_lst),
            "progress_rate": sample['progress_rate'],
            "goal": sample['goal'],
            "init_obs": init_obs,
            "actions": grounding_action_lst,
            "progress": grounding_progress_lst,
            "observations": grounding_obs_lst
        }
        
        res_list.append(new_sample)
        unique_id += 1

    return res_list


def save_to_jsonl(data, filename):
    """Appends a list of dictionaries to a JSONL file."""
    with open(filename, 'a', encoding='utf-8') as file:  # Open in append mode
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")  # Ensure newline separation


def generate_output_from_api(model, prompt, script_args=None):
    """
    Generates output using the OpenAI API with the specified model and prompt.

    Args:
        model (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
        prompt (str): The text prompt to send to the model.
        script_args (dict, optional): Additional generation arguments (e.g., temperature, max_tokens).

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the model type is not recognized.
    """
    if script_args is None:
        script_args = {}

    if 'qwen' in model.lower():
        client = OpenAI(api_key=qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    elif 'gpt' in model.lower():
        client = OpenAI(api_key=gpt_api_key)
    else:
        raise ValueError(f"Invalid model type: {model}. Model must contain either 'qwen' or 'gpt' in its name.")
        
    temperature = script_args.temperature
    max_tokens = script_args.max_length

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def generate_output_from_local(model, tokenizer, prompt, script_args=None):
    """
    Generates output using a local language model with the specified model, tokenizer and prompt.

    Args:
        model: The local language model to use for generation
        tokenizer: The tokenizer associated with the model
        prompt (str): The text prompt to send to the model
        script_args (dict, optional): Additional generation arguments (e.g., temperature, max_tokens)

    Returns:
        str: The generated response from the model
    """
    # Format the messages as chat-style prompt
    chat_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    # Tokenize the prompt
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=script_args.max_length,
            temperature=script_args.temperature,
            top_p=script_args.top_p,
            do_sample=script_args.do_sample,
            eos_token_id=tokenizer.eos_token_id)

    # Slice off the input tokens to get only the generated part
    generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]

    # Decode and print
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()



def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def load_embedding_jsonl(path: str) -> Dict[str, Dict]:
    id_to_item = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item['embedding'] = np.array(item['embedding'])
            id_to_item[item['id']] = item
    return id_to_item

def extract_task_ids(obj):
    task_ids = set()

    def recurse(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'task_uid':
                    task_ids.add(v)
                recurse(v)
        elif isinstance(o, list):
            for item in o:
                recurse(item)
    recurse(obj)
    return task_ids

def extract_and_sort_by_task_id(data: dict, target_task_id: str):
    matched_items = []

    def recurse(obj):
        if isinstance(obj, dict):
            if obj.get('task_uid') == target_task_id and 'progress_rate' in obj:
                matched_items.append(obj)
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)

    # Sort by progress_rate
    matched_items.sort(key=lambda x: x['progress_rate'], reverse=True)
    return matched_items
