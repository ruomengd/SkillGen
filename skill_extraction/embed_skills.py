from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

def add_current_node_embeddings(graph_data, model):
    """
    Add embeddings to each graph node based on current_node string.
    
    Args:
        graph_data (list of dict): The input graph.
        model_name (str): SentenceTransformer model name.

    Returns:
        list of dict: Graph data with added 'embedding' key per entry.
    """
    
    # Extract all unique current_node strings
    all_nodes = [entry["current_node"] for entry in graph_data]
    
    # Compute embeddings
    embeddings = model.encode(all_nodes, convert_to_numpy=True)

    # Add embeddings to original graph data
    for i, entry in enumerate(graph_data):
        entry["embedding"] = embeddings[i].tolist()

    return graph_data



import argparse
import random
import numpy as np
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)



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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for model_name in args.model_names:
        for dataset_name in args.datasets:
            for fold_num in range(args.fold_count):
                save_dir = "./logs/skill_extraction/skills/%s/%s/sampling_count_%s/fold_%s"%(dataset_name, model_name, args.sampling_count, fold_num)
                # filtering by each fold, count the num of category
                task_set, goal_dict = set(), dict()
                fold_path = './data/%s/train_data_%s.jsonl'%(dataset_name, fold_num)
                id_list = []
                with open(fold_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        item = json.loads(line.strip()) 
                        goal = item["goal"]
                        if dataset_name == 'sc':
                            task_category = item["additional_info"]["env_name"]
                        elif dataset_name == 'alfworld':
                            task_category = item["additional_info"]["description"].split('-')[0].strip()
                        elif dataset_name == 'babyai':
                            task_category = item["additional_info"]["subtask"]

                        task_set.add(task_category)
                        

                # for idx, category_name in enumerate(list(task_set)):
                for idx in tqdm(range(0, len(list(task_set))), desc="Generating Rules: "):
                    category_name = list(task_set)[idx]    

                    with open('%s/%s.jsonl'%(save_dir, category_name)) as f:
                        graph_data = [json.loads(line) for line in f]
                
                    # Add embeddings
                    updated_graph = add_current_node_embeddings(graph_data, model)
                    # Save if needed
                    with open('%s/%s_embed.jsonl'%(save_dir, category_name), "w") as f:
                        for item in updated_graph:
                            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
