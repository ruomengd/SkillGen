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

fold_count = 4
model = SentenceTransformer('all-MiniLM-L6-v2')
for dataset_name in ['alfworld', 'babyai', 'sc']:
# for dataset_name in ['alfworld']:
    for count in [6]:
        for model_name in ['gpt-4o-mini', 'qwen-turbo', 'Qwen2.5-7B-Instruct']: # 'gpt-4o-mini', 'qwen-turbo', 'Qwen2.5-7B-Instruct', 
        # for model_name in ['Qwen2.5-7B-Instruct']:
            print(model_name)
            for fold_num in range(fold_count):
                save_dir = "./domain_rules_temp1.0/extracted_rules_progress_stepwise_weighting/%s/%s/sampling_count_%s/fold_%s"%(dataset_name, model_name, count, fold_num)
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
