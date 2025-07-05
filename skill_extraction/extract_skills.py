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
from extraction_utils import *
from domain_graph import *
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

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

    for model_name in args.model_names:
        for dataset_name in args.datasets:
            for fold_num in range(args.fold_count):
                # Setup directories
                save_dir = f"./logs/skill_extraction/skills/{dataset_name}/{model_name}/sampling_count_{args.sampling_count}/fold_{fold_num}"
                os.makedirs(save_dir, exist_ok=True)
                print(f"Processing {save_dir}")

                # Load and process fold data
                task_set = set()
                goal_dict = defaultdict(set)
                id_list = []
                
                fold_path = f'./data/{dataset_name}/train_data_{fold_num}.jsonl'
                with open(fold_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        item = json.loads(line.strip())
                        task_category = get_task_category(item, dataset_name)
                        task_id = get_task_id(item, dataset_name)
                        
                        id_list.append(task_id)
                        task_set.add(task_category)
                        goal_dict[task_category].add(item["goal"])

                # Save goal-category mapping
                goal_identify_lst = [[str(goal), category] for category in goal_dict for goal in goal_dict[category]]
                pd.DataFrame(goal_identify_lst, columns=['Goal', 'Category']).to_csv(f'{save_dir}/goal_category.csv', index=False)

                # Extract and process results
                g_list, p_list = extract_res(dataset_name=dataset_name, count=args.sampling_count, model_name=model_name, id_list=id_list)
                g_list_drop_duplicates = keep_one_of_duplicates_dicts(g_list)
                p_list_drop_duplicates = keep_one_of_duplicates_dicts(p_list)
                valid_trajs = collect_valid_traj(p_list_drop_duplicates, dataset_name=dataset_name)

                # Process each task category
                for idx, category_name in enumerate(tqdm(list(task_set), desc="Generating Rules")):
                    current_valid_trajs = [item for item in valid_trajs if item['task_name'] == category_name]
                    print(f'Processing category {idx+1}/{len(task_set)}: {category_name}')
                    print(f'Total valid trajectories: {len(valid_trajs)}')
                    print(f'Current category trajectories: {len(current_valid_trajs)}')

                    if not current_valid_trajs:
                        save_to_jsonl([], filename=f'{save_dir}/{category_name}.jsonl')
                        continue

                    # Build and analyze task graph
                    task_graph = TaskGraph(save_dir=save_dir, task_name=category_name)
                    trajectories = process_trajectories(current_valid_trajs)
                    task_graph.add_sample_trajectory([traj['trajs'] for traj in trajectories])
                    task_graph.visualize()

                    print(f'Task Graph initialized with {len(task_graph.graph.nodes)} nodes and {len(task_graph.graph.edges)} edges')

                    # Compute action contributions
                    estimator = ActionContributionEstimator(task_graph)
                    estimator.compute_q_values(num_iterations=500)
                    final_ranking = estimator.combined_ranking_step_wise()

                    # Save results
                    data_list = [
                        {
                            "current_node": parent_action,
                            "children": [{"action": child, "score": score} for child, score in children]
                        }
                        for parent_action, children in final_ranking.items()
                    ]
                    
                    with open(f'{save_dir}/{category_name}.jsonl', 'w', encoding='utf-8') as f:
                        for item in data_list:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
