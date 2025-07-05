import pandas as pd
import torch
import json
from tqdm.auto import tqdm
import random
import numpy as np
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.alfworld_dataset import AlfWorldDataset
from dataset.babyai_dataset import BabyAIDataset
from dataset.scienceworld_dataset import ScienceWorldDataset
import json
import warnings
warnings.simplefilter("ignore", UserWarning)

# Environment setup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import *
from prompt.prompts import *
from prompt.func_prompting import *
load_dotenv()  # load .env file

    
def parse_args():
    parser = argparse.ArgumentParser(description="Script for generating responses using language models with skill-based guidance.")

    # Inference and sampling settings
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum number of steps for task completion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling during text generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (higher = more random).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability threshold.")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of generated text.")
    parser.add_argument("--hist_size", type=int, default=20, help="Number of previous interactions to include in history.")
    parser.add_argument("--dataset_name", type=str, default="./data/scienceworld/test.jsonl", help="Path to dataset file.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b-it", help="Name or path of language model.")
    parser.add_argument("--save_path", type=str, default="./save_path", help="Path to save generation results.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run model on (e.g. cuda:0, cpu).")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited samples.")

    # Skill retrieval and guidance settings
    parser.add_argument("--prompt_mode", type=str, default="skillgen", help="Prompting mode for inference.")
    parser.add_argument("--sampling_count", type=int, default=6, help="Number of samples for skill extraction.")
    parser.add_argument("--fold_num", type=int, default=0, help="Data fold number for cross validation.")
    parser.add_argument("--top_s", type=int, default=3, help="Number of similar skills to retrieve.")
    parser.add_argument("--top_ac", type=int, default=1, help="Number of top actions to consider per skill.")

    return parser.parse_args()


def load_dataset(script_args):
    # Load dataset
    print('Load dataset.')

    if script_args.dataset_name == 'alfworld':
        label_path = './data/%s/all.jsonl'%(script_args.dataset_name)
        data_path = './data/%s/test_data_%s.jsonl'%(script_args.dataset_name, script_args.fold_num)
        id_list = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                id_list.append(json.loads(line.strip())["additional_info"]["description"])
        dataset_all = AlfWorldDataset(label_path)
        dataset = dataset_all.load_from_fold(id_list)
    
    elif script_args.dataset_name == 'babyai':
        label_path = './data/%s/all.jsonl'%(script_args.dataset_name)
        data_path = './data/%s/test_data_%s.jsonl'%(script_args.dataset_name, script_args.fold_num)
        id_list = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                id_list.append(int(json.loads(line.strip())['id']))
        dataset_all = BabyAIDataset(label_path)
        dataset = dataset_all.load_from_fold(id_list)

    elif script_args.dataset_name == 'sc':
        data_path = './data/%s/test_data_%s.jsonl'%(script_args.dataset_name, script_args.fold_num)
        dataset = ScienceWorldDataset(data_path)

    else:
        raise ValueError(f"Unsupported dataset name: {script_args.dataset_name}")

    if script_args.debug:
        dataset = dataset.select(0, 10)

    total_size = len(dataset)
    print(f"Total Dataset Size: {total_size}")
    print("Model & Datasets Loaded.")

    return dataset


def make_prompt(step, args, category, goal, history=None, prompt_dict=None, check_actions="check valid actions", check_inventory="inventory", query_id=None, retrieve_model=None):
    if args.dataset_name == 'alfworld':
        prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'}
        category = category.split('-')[0].strip()
        # examples
        for i, (k, v) in enumerate(prefixes.items()):
            if category.startswith(k):
                examples = "".join(prompt_dict['examples'][v])
    elif args.dataset_name == 'babyai' or args.dataset_name == 'sc':
        examples = prompt_dict["examples"]
    
    
    if history is not None:
        hist_size = args.hist_size * 2
        local_history = history[-hist_size:]
        hist_info = "\n".join([item[0] + ": " + item[1] for item in local_history])

    if args.prompt_mode == 'skillgen':
        # Load domain mapping and retrieve similar domains
        domain_map_path = os.path.join('./logs/skill_extraction/segments', f'{args.dataset_name}_retrieved_domains.jsonl')
        domain_map = load_domain_map(domain_map_path)
        retrieved_category = query_domains(category, domain_map)
        # print('retrieved_category', retrieved_category)
        
        # Combine original domain with retrieved ones, avoiding duplicates
        domains = [category]
        domains.extend([cat for cat in retrieved_category if cat != category])

        # Golden Segment
        # Construct path to domain segments file
        file_path = os.path.join(
            './logs/skill_extraction/segments',
            args.dataset_name,
            args.model_name.split('/')[-1],
            f'count{args.sampling_count}',
            f'query_{query_id.replace("/", "_")}.txt'
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                demo_info = file.read()
        except FileNotFoundError:
            print(f"Warning: Could not find domain segments file at {file_path}")
            demo_info = ""

        # Step-wise Skills
        retrieved_skills = skillgen_prompting(query_id=category, retrieved_category=retrieved_category, query_hist=history, model=retrieve_model, args=args, top_s=args.top_s, top_ac=args.top_ac, temp=1.0)
        prompts = skillgen_demoac_skill.format(instructions=prompt_dict["simple_instruction"], goal=goal, skills=retrieved_skills, trajectories=demo_info, history=hist_info)
        # print(prompts)
        # input()
    
    else:
        raise ValueError("Invalid Input")
   
    query = prompts
    # print(query)
    # input()
   
    messages = [
            {"role": "system", "content": prompt_dict["system_msg"]},
            {"role": "user", "content": query}]
    return messages


def inference_step_from_api(model, entry, prompt_dict, args, retrieve_model=None):
    query_id = str(entry['task_uid'])

    for step in range(args.max_steps):
        prompt = make_prompt(step=step, args=args, category=entry["task_name"], goal=entry["goal"], history=entry["trajectory"], prompt_dict=prompt_dict, query_id=query_id, retrieve_model=retrieve_model)
        # print(prompt)
        # input()
        outputs = generate_output_from_api(model, prompt, script_args=args)
        action = extract_action(outputs)
        
        env = entry["env"]
        
        if args.dataset_name == 'alfworld' or args.dataset_name == 'sc':
            valid_actions = entry["env"].get_action_space()
            if action in valid_actions:
                entry["grounding"].append(step)
            
        # Excecute action
        observation, reward, is_done, info = env.step(action)

        if args.dataset_name == 'babyai':
            if info.get("action_is_valid", False): 
                entry["grounding"].append(step)

        print('STEP: %s - ACTION: %s - REWARD: %s'%(step, action, reward))
        entry["num_step"] += 1
        entry["trajectory"].extend([['ACTION', action], ['OBSERVATION', observation]])
        
        if reward > entry["progress_rate"]:
            entry["progress"].append([step, reward])
        entry["progress_rate"] = reward

        if is_done:
            break

    results = {
        "task_uid": entry['task_uid'],
        'task_name': entry['task_name'],
        'goal': entry['goal'],
        'progress': entry['progress'],
        'trajectory': entry["trajectory"],
        "grounding": entry['grounding'],
        "num_step": entry['num_step'],
        "grounding_rate": len(entry['grounding']) / entry['num_step'],
        'progress_rate': entry['progress_rate']
    }

    return results


def inference_step_from_local(model, tokenizer, entry, prompt_dict, args, retrieve_model=None):
    query_id = str(entry['task_uid'])

    for step in range(args.max_steps):
        prompt = make_prompt(step=step, args=args, category=entry["task_name"], goal=entry["goal"], history=entry["trajectory"], prompt_dict=prompt_dict, query_id=query_id, retrieve_model=retrieve_model)
        print(prompt)
        input()
        outputs = generate_output_from_local(model, tokenizer, prompt, script_args=args)
        action = extract_action(outputs)
        
        
        env = entry["env"]
        
        if args.dataset_name == 'alfworld' or args.dataset_name == 'sc':
            valid_actions = entry["env"].get_action_space()
            if action in valid_actions:
                entry["grounding"].append(step)
            
        # Excecute action
        observation, reward, is_done, info = env.step(action)

        if args.dataset_name == 'babyai':
            if info.get("action_is_valid", False): 
                entry["grounding"].append(step)

        print('STEP: %s - ACTION: %s - REWARD: %s'%(step, action, reward))
        entry["num_step"] += 1
        entry["trajectory"].extend([['ACTION', action], ['OBSERVATION', observation]])
        
        if reward > entry["progress_rate"]:
            entry["progress"].append([step, reward])
        entry["progress_rate"] = reward

        if is_done:
            break

    results = {
        "task_uid": entry['task_uid'],
        'task_name': entry['task_name'],
        'goal': entry['goal'],
        'progress': entry['progress'],
        'trajectory': entry["trajectory"],
        "grounding": entry['grounding'],
        "num_step": entry['num_step'],
        "grounding_rate": len(entry['grounding']) / entry['num_step'],
        'progress_rate': entry['progress_rate']
    }

    return results


# Main Execution Function
def generate_samples(script_args):
    """
    Manages dataset loading, model initialization, and inference execution.
    """
    # Build base filename components
    model_name = script_args.model_name.split('/')[-1]
    base_name = f"{model_name}_temp{script_args.temperature}_step{script_args.max_steps}"
    fold_suffix = f"_fold{script_args.fold_num}.jsonl"

    # Add prompt mode specific components and all relevant parameters
    save_name = (f"{base_name}_{script_args.prompt_mode}_"
                f"sampling{script_args.sampling_count}_"
                f"maxsteps{script_args.max_steps}_"
                f"maxlen{script_args.max_length}_"
                f"temp{script_args.temperature}_"
                f"topp{script_args.top_p}_"
                f"tops{script_args.top_s}_"
                f"topac{script_args.top_ac}_"
                f"hist{script_args.hist_size}{fold_suffix}")

    output_file = os.path.join(script_args.save_path, script_args.dataset_name, save_name)
    
    # Remove old file for clean results (optional)
    if os.path.exists(output_file):
        os.remove(output_file)

    # Load dataset
    dataset = load_dataset(script_args)
   
    # Load prompt
    if script_args.dataset_name == 'alfworld':
        prompt_path = './prompt/task/alfworld_base.json'
    elif script_args.dataset_name == 'babyai':
        prompt_path = './prompt/task/babyai_base.json'
    elif script_args.dataset_name == 'sc':
        prompt_path = './prompt/task/scienceworld_base.json'
    else:
        raise ValueError(f"Unsupported dataset name: {script_args.dataset_name}")
    with open(prompt_path, 'r') as f:
        prompt_dict = json.load(f)

    if script_args.prompt_mode in ['skillgen']:
        retrieve_model = SentenceTransformer('all-MiniLM-L6-v2').to(script_args.device)  
        retrieve_model.eval() # Set model to evaluation mode
    else:
        retrieve_model = None


    def process_sample(sample):
        """Helper function to format sample results consistently"""
        return [{
            "task_uid": sample['task_uid'],
            'task_name': sample['task_name'], 
            'goal': sample['goal'],
            'progress': sample['progress'],
            'trajectory': sample["trajectory"],
            "grounding": sample['grounding'],
            "num_step": sample['num_step'],
            "grounding_rate": sample['grounding_rate'],
            'progress_rate': sample['progress_rate']
        }]

    def save_results(results, output_file):
        """Helper function to save and log results"""
        save_to_jsonl(results, output_file)
        print(f"Saved {len(results)} results to {output_file}.")

    # Handle API-based models
    if 'qwen-turbo' in script_args.model_name.lower() or 'gpt-4o-mini' in script_args.model_name.lower():
        for data in tqdm(dataset, desc="Generating responses with API model"):
            sample = inference_step_from_api(
                script_args.model_name, 
                data,
                prompt_dict,
                script_args,
                retrieve_model=retrieve_model
            )
            results = process_sample(sample)
            save_results(results, output_file)

    # Handle local models
    else:
        # model = AutoModelForCausalLM.from_pretrained(
        #     script_args.model_name,
        #     torch_dtype="bfloat16"
        # ).to(script_args.device)
        # model.eval() # Set model to evaluation mode
        
        # tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        # tokenizer.pad_token = tokenizer.eos_token

        model, tokenizer = None, None

        for data in tqdm(dataset, desc="Generating responses with local model"):
            sample = inference_step_from_local(
                model,
                tokenizer, 
                data,
                prompt_dict,
                script_args,
                retrieve_model=retrieve_model
            )
            results = process_sample(sample)
            save_results(results, output_file)


# Run main function
if __name__ == "__main__":
     # Parse arguments
    script_args = parse_args()

    # Set random seeds for reproducibility across all libraries
    seed = script_args.seed
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not os.path.exists(f'{script_args.save_path}/{script_args.dataset_name}'):
        os.makedirs(f'{script_args.save_path}/{script_args.dataset_name}', exist_ok=True)

    generate_samples(script_args)

