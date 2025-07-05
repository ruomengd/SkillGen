import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_embeddings(jsonl_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            obj['embedding'] = np.array(obj['embedding'])
            records.append(obj)
    return records

def compute_top_k_domains(records, k=3):
    embeddings = np.stack([r['embedding'] for r in records])
    similarity_matrix = cosine_similarity(embeddings)

    results = []
    for i, record in enumerate(tqdm(records)):
        sim_scores = list(enumerate(similarity_matrix[i]))
        sim_scores = sorted([(j, s) for j, s in sim_scores if j != i], key=lambda x: -x[1])

        seen_domains = set()
        top_domains = []
        for j, _ in sim_scores:
            domain = records[j]['domain']
            if domain not in seen_domains:
                seen_domains.add(domain)
                top_domains.append(domain)
            if len(top_domains) == k:
                break

        results.append({
            'id': record['id'],
            'domain': record['domain'],
            'retrieved_domains': top_domains
        })

    return results


def save_results_to_jsonl(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



for dataset_name in ['alfworld', 'babyai', 'sc']:
    # Example usage
    jsonl_path = f'./logs/skill_extraction/segments/{dataset_name}_meta_embeddings.jsonl'
    output_path = f'./logs/skill_extraction/segments/{dataset_name}_retrieved_domains.jsonl'
    records = load_embeddings(jsonl_path)
    top_k_results = compute_top_k_domains(records, k=3)
    save_results_to_jsonl(top_k_results, output_path)

    # Print or save results
    for r in top_k_results:
        print(f"ID: {r['id']}, Domain: {r['domain']}, Relevant Domains: {r['retrieved_domains'][0]}")
        print(len(r['retrieved_domains']))
