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
from collections import Counter, defaultdict
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from extraction_utils import *
import random
random.seed(42)


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract skills from sampled trajectories')
    parser.add_argument('--fold_count', type=int, default=4,
                        help='Number of folds for cross validation')
    parser.add_argument('--model_names', nargs='+', 
                        default=['gpt-4o-mini', 'qwen-turbo', 'Qwen2.5-7B-Instruct'],
                        help='List of model names to process')
    parser.add_argument('--datasets', nargs='+',
                        default=['alfworld', 'babyai', 'sc'],
                        help='List of datasets to process') 
    parser.add_argument('--sampling_count', type=int, default=6,
                        help='Number of sampling used for skill extraction')
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset_name in args.datasets:
        embed_path = f'./logs/skill_extraction/segments/{dataset_name}_meta_embeddings.jsonl'
        embedding_dict = load_embedding_jsonl(embed_path)

        for model_name in args.model_names:
            save_path = './logs/skill_extraction/segments/%s/%s/count%s'%(dataset_name, model_name, args.sampling_count)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)


            candidate_dict = dict()
            for fold_num in range(args.fold_count):
                train_path = './data/%s/train_data_%s.jsonl'%(dataset_name, fold_num)
                id_list_train = []
                with open(train_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        item = json.loads(line.strip()) 
                        goal = item["goal"]
                        if dataset_name == 'sc':
                            id_list_train.append('%s_%s'%(item["additional_info"]["env_name"], item["additional_info"]["var"]))
                        elif dataset_name == 'alfworld':
                            id_list_train.append((item["additional_info"]["description"]))
                        elif dataset_name == 'babyai':
                            id_list_train.append(item["id"])
                test_path = './data/%s/test_data_%s.jsonl'%(dataset_name, fold_num)
                id_list_test = []
                with open(test_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        item = json.loads(line.strip()) 
                        goal = item["goal"]
                        if dataset_name == 'sc':
                            id_list_test.append('%s_%s'%(item["additional_info"]["env_name"], item["additional_info"]["var"]))
                        elif dataset_name == 'alfworld':
                            id_list_test.append((item["additional_info"]["description"]))
                        elif dataset_name == 'babyai':
                            id_list_test.append((item["id"]))
                for item in id_list_test:
                    candidate_dict[item] = list(id_list_train)

            for query_id, candidates in tqdm(candidate_dict.items(), desc="Retrieving"):
                print('query_id', query_id)
                
                id_list = candidate_dict[query_id]
                print(id_list)
                # if query_id in id_list:
                #     input()
                g_list, progress_list = extract_res(dataset_name=dataset_name, count=args.sampling_count, model_name=model_name, id_list=id_list)
                # g_list_drop_duplicates = keep_one_of_duplicates_dicts(g_list)
                progress_list_drop_duplicates = keep_one_of_duplicates_dicts(progress_list)
                valid_trajs = collect_valid_traj(progress_list_drop_duplicates, dataset_name=dataset_name)
                candidate_ids = extract_task_ids(valid_trajs)
        
                query_domain = embedding_dict[query_id]["domain"]
                sims = []
                for cid in valid_trajs:
                    cand_domain = cid["task_name"]
                    if cand_domain == query_domain:
                        sims.append(cid)
                
                # Sort by progress_rate
                sims.sort(key=lambda x: x['progress_rate'], reverse=True)
                # for item in sims:
                #     print(item)
                #     input()
        
                chosen_trajs_set = sims[:1]
                
                # assert len(chosen_trajs_set) > 0
                current_traj_str_set = []
                for idx, traj in enumerate(chosen_trajs_set):
                    print(traj)
                    goal, init_obs = traj['goal'], traj['init_obs']
                    # print(f"POS: {p}")
                    assert len(traj['actions']) == len(traj['observations'])
                    history_pos = list(zip(traj['actions'], traj['observations'], traj['progress'])) 
                    history_pos_str = [f'Goal: {goal}\n{init_obs}']
                    history_pos_str.extend([f"ACTION: {action}" for action, obs, progress in history_pos])
                    current_traj_str = "\n".join(history_pos_str)
                    current_traj_str_set.append(current_traj_str)

                output_retrieved_1_shot = "\n\n".join([f"Example {i+1}:\n{text}" for i, text in enumerate(current_traj_str_set)])
                if dataset_name == 'alfworld':
                    file_path = '%s/query_%s.txt'%(save_path, query_id.replace('/', '_'))
                else:
                    file_path = '%s/query_%s.txt'%(save_path, query_id)
                print(len(chosen_trajs_set), file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(output_retrieved_1_shot)
            


if __name__ == "__main__":
    main()
