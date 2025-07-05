

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


def retrieve_skills_by_action_embedding(graph_data, query_action, model, top_k_actions=1, top_k_skills=1):
    """
    Retrieve top-K similar action nodes based on embedding, and return top-K forward and backward skills for each.
    """
    # Step 1: Embed the query action
    query_emb = model.encode([query_action], convert_to_numpy=True)

    # Step 2: Compute or retrieve node embeddings
    if 'embedding' not in graph_data[0]:
        all_nodes = [entry["current_node"] for entry in graph_data]
        node_embeddings = model.encode(all_nodes, convert_to_numpy=True)
        for i, entry in enumerate(graph_data):
            entry["embedding"] = node_embeddings[i].tolist()
    else:
        node_embeddings = np.array([entry["embedding"] for entry in graph_data])

    # Step 3: Similarity & top-K matching
    similarities = cosine_similarity(query_emb, node_embeddings)[0]
    top_indices = similarities.argsort()[-top_k_actions:][::-1]

    results = {
        "matched_nodes": [graph_data[i]["current_node"] for i in top_indices],
        "skills": []
    }
    print('top_indices', top_indices)
    print(results)

    for idx in top_indices:
        matched_node = graph_data[idx]["current_node"]
        forward = []
        backward = []

        for node in graph_data:
            # Backward skills
            for child in node["children"]:
                if child["action"] == matched_node:
                    backward.append((node["current_node"], matched_node, child["score"]))
            # Forward skills
            if node["current_node"] == matched_node:
                forward.extend([(matched_node, child["action"], child["score"]) for child in node["children"]])

        forward = sorted(forward, key=lambda x: -x[2])[:top_k_skills]
        backward = sorted(backward, key=lambda x: -x[2])[:top_k_skills]

        results["skills"].append({
            "node": matched_node,
            "forward": forward,
            "backward": backward
        })

    print(results)

    return results



def skillgen_prompting(query_id, retrieved_category, query_hist, model, args, top_s=1, top_ac=1, temp=1.0):
    """
    Generate a prompt with relevant skills based on action history.
    
    Args:
        query_id: ID of the current query
        retrieved_category: List of related category IDs
        query_hist: History of previous actions
        model: Embedding model for similarity matching
        args: Runtime arguments
        top_s: Number of top skills to include
        top_ac: Number of top actions to match
        temp: Temperature parameter for path construction
        
    Returns:
        str: Generated prompt with relevant skills
    """
    # Combine query ID with retrieved categories
    candidates = [query_id] + list(retrieved_category)

    # Find first valid graph data file
    graph_data = []
    for id in candidates:
        jsonl_path = os.path.join(
            f'./logs/skill_extraction/skills',
            args.dataset_name,
            args.model_name.split('/')[-1],
            f'sampling_count_{args.sampling_count}',
            f'fold_{args.fold_num}',
            f'{id}_embed.jsonl'
        )
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                graph_data = [json.loads(line.strip()) for line in f]
            if len(graph_data) > 0:
                break

    if len(graph_data) == 0:
        return ""

    # Get query action from history
    query_ac = 'INIT_STATE' if len(query_hist) == 1 else query_hist[-2][1]
    
    # Retrieve relevant skills
    result = retrieve_skills_by_action_embedding(
        graph_data, 
        query_ac,
        model,
        top_k_actions=top_ac,
        top_k_skills=top_s
    )

    # Generate formatted prompt
    prompt_parts = []
    for i, skill in enumerate(result["skills"], 1):
        node = skill["node"]
        skill_parts = [f"### Skill {i}: Centered on action '{node}'"]

        if skill["backward"]:
            skill_parts.append("  - Common precursors to this action:")
            for from_node, to_node, score in skill["backward"]:
                skill_parts.append(f"    • Agents often perform '{from_node}' before '{to_node}'.")

        if skill["forward"]:
            skill_parts.append("  - Typical next steps after this action:")
            for from_node, to_node, score in skill["forward"]:
                skill_parts.append(f"    • After '{from_node}', agents usually continue with '{to_node}'.")

        prompt_parts.append("\n".join(skill_parts))

    prompt = "\n\n".join(prompt_parts)
    return prompt.replace('INIT_STATE', 'the beginning of the task')
