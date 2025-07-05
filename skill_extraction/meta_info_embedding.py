import os
import json
from openai import OpenAI
from tqdm import tqdm
# Environment setup
from dotenv import load_dotenv
load_dotenv()  # load .env file
# Access the API key
gpt_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=gpt_api_key)  # or pass api_key="your-key"

def get_embedding(text, model="text-embedding-3-small"):  # or use text-embedding-ada-002
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


for dataset_name in ['alfworld', 'babyai', 'sc']:
    fold_path = f'./data/{dataset_name}/all.jsonl'
    output_path = f'./logs/skill_extraction/segments/{dataset_name}_meta_embeddings.jsonl'


    with open(fold_path, 'r', encoding='utf-8') as file, open(output_path, 'a', encoding='utf-8') as out_file:
        for line in tqdm(file, desc=f"Embedding {dataset_name}"):
            item = json.loads(line.strip())
            goal = item["goal"]

            if dataset_name == 'sc':
                domain = item["additional_info"]["env_name"]
            elif dataset_name == 'alfworld':
                domain = item["additional_info"]["description"].split('-')[0]
            elif dataset_name == 'babyai':
                goal += '.'
                domain = item["additional_info"]["subtask"]
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            if dataset_name == 'sc':
                id = f'{item["additional_info"]["env_name"]}_{item["additional_info"]["var"]}'
            elif dataset_name == 'alfworld':
                id = item["additional_info"]["description"]
            elif dataset_name == 'babyai':
                id = item["id"]

            meta_data = f'Goal: {goal} Domain: {domain}.'
            embedding = get_embedding(meta_data)

            output_obj = {
                "id": id,
                "goal": goal,
                "domain": domain,
                "meta_data": meta_data,
                "embedding": embedding
            }
            out_file.write(json.dumps(output_obj) + '\n')
           
