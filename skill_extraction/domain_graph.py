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
import random


class TaskGraph:
    def __init__(self, save_dir, task_name):
        self.graph = nx.DiGraph()
        self.edge_progress = defaultdict(list)  # Store per-edge observed progress changes
        self.obs = {}
        self.task_name = task_name
        self.save_dir = save_dir

    def add_sample_trajectory(self, trajectory):
        """
        Add a sample trajectory to build an action-only graph, while recording the actual progress changes for each edge.
        :param trajectory: list of (obs, action, progress)
        """
        for trajs in trajectory:
            for i in range(len(trajs) - 1):
                obs1, action1, progress1 = trajs[i]
                obs2, action2, progress2 = trajs[i + 1]

                filtered_action1 = remove_numbers(action1.split('(')[0].strip()).replace('  ', ' ').strip()
                filtered_action2 = remove_numbers(action2.split('(')[0].strip()).replace('  ', ' ').strip()
                if not filtered_action1 or not filtered_action2:
                    print(f"Warning: Empty action detected! action1: {action1}, progress1: {progress1}")
                    print(f"action2: {action2}, progress2: {progress2}")
                if filtered_action1 and filtered_action2:  # Only if both actions are valid
                    self.graph.add_node(filtered_action1)
                    self.graph.add_node(filtered_action2)
                    self.graph.add_edge(filtered_action1, filtered_action2)

                    progress_diff = progress2 - progress1
                    self.edge_progress[(filtered_action1, filtered_action2)].append(progress_diff)

        # Remove self-loops if any exist
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        # Control graph size
        MAX_NODES = 30
        if len(self.graph.nodes) > MAX_NODES:
            self._prune_graph_by_progress(MAX_NODES)
        
    
    def _prune_graph_by_progress(self, max_nodes):
        """
        When the number of nodes exceeds max_nodes, prune non-critical nodes based on progress scores and reconnect the structure.
        """
        node_scores = {}
        for node in self.graph.nodes:
            if node in {'INIT_STATE', 'FINAL_STATE'}:
                continue

            progress_values = []
            for u, v in self.graph.in_edges(node):
                deltas = self.edge_progress.get((u, v), [])
                if deltas:
                    progress_values.append(np.mean(deltas))
            if progress_values:
                node_scores[node] = np.mean(progress_values)

        # Sort and remove low-scoring nodes
        nodes_sorted = sorted(node_scores.items(), key=lambda x: x[1])
        num_to_remove = len(self.graph.nodes) - max_nodes
        nodes_to_remove = []

        for node, _ in nodes_sorted:
            if self.graph.in_degree(node) > 0 and self.graph.out_degree(node) > 0:
                nodes_to_remove.append(node)
            if len(nodes_to_remove) >= num_to_remove:
                break

        for node in nodes_to_remove:
            parents = list(self.graph.predecessors(node))
            children = list(self.graph.successors(node))
            self.graph.remove_node(node)

            for parent in parents:
                for child in children:
                    if parent in self.graph.nodes and child in self.graph.nodes:
                        if not self.graph.has_edge(parent, child):
                            try:
                                if not nx.has_path(self.graph, child, parent):
                                    self.graph.add_edge(parent, child)
                            except nx.NetworkXError:
                                continue  # Safety check
        # Remove self-loops
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))


    def visualize(self):
        """
        Visualize the task graph and optimize layout (top-to-bottom)
        """
        plt.figure(figsize=(20, 12))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color="lightblue", edge_color="gray")
        plt.savefig("%s/%s.png"%(self.save_dir, self.task_name), dpi=300, bbox_inches='tight')
        plt.close()


class ActionContributionEstimator:
    def __init__(self, task_graph, gamma=0.95, lambda_=0.9, alpha=0.05, reward_noise_std=0.001):
        self.graph = task_graph.graph
        self.edge_progress = task_graph.edge_progress  # access edge progress memory
        self.gamma = gamma
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reward_noise_std = reward_noise_std
        self.Q = defaultdict(lambda: np.random.uniform(0.01, 0.05))  # Q-value for each action
        self.E = defaultdict(float)  # Eligibility Trace

    def find_source_target(self):
        sources = [node for node in self.graph.nodes if node == 'INIT_STATE']
        targets = [node for node in self.graph.nodes if node == 'FINAL_STATE']
        return sources[0], targets[0]

    def sample_paths(self, source, target, num_samples=1000, cutoff=20):
        all_paths = list(nx.all_simple_paths(self.graph, source=source, target=target, cutoff=cutoff))
        print(f'Found {len(all_paths)} paths from INIT_STATE to FINAL_STATE.')
        if len(all_paths) <= num_samples:
            return all_paths
        return random.sample(all_paths, num_samples)

    def compute_q_values(self, num_iterations=1000, tolerance=1e-3, patience=5):
        source, target = self.find_source_target()

        if not source or not target:
            raise ValueError("Graph must have at least one source (INIT_STATE) and one target (FINAL_STATE)")

        no_improve_count = 0
        prev_q = self.Q.copy()

        for idx in range(num_iterations):
            print(f'Iteration {idx}: Updating Q-values')

            sampled_paths = self.sample_paths(source, target, num_samples=2000, cutoff=20)

            for path in sampled_paths:
                self.update_q_values(path)

            delta = sum(abs(self.Q[k] - prev_q.get(k, 0.0)) for k in self.Q) / (len(self.Q) + 1e-8)
            print(f'Average Q-value change: {delta:.6f}')

            if delta < tolerance:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at iteration {idx} (no significant Q-value change for {patience} steps).")
                    break
            else:
                no_improve_count = 0

            prev_q = self.Q.copy()

    def update_q_values(self, path):
        for t in range(len(path) - 1):
            action1 = path[t]
            action2 = path[t + 1]

            if (action1, action2) in self.edge_progress:
                progress_candidates = self.edge_progress[(action1, action2)]
                progress_diff = random.choice(progress_candidates)
            else:
                progress_diff = 0.0  # No progress observed -> Treat as neutral step

            reward = progress_diff + np.random.normal(0, self.reward_noise_std)

            delta_t = reward + self.gamma * self.Q[action2] - self.Q[action1]

            self.E[action1] += 1

            for key in self.E:
                self.Q[key] += self.alpha * delta_t * self.E[key]
                self.E[key] *= self.gamma * self.lambda_

    def estimate_action_contributions(self):
        action_contributions = defaultdict(float)

        for action, q_value in self.Q.items():
            q_value = max(0, q_value) 
            action_contributions[action] += q_value

        total = sum(action_contributions.values())
        if total > 0:
            action_contributions = {a: v / total  for a, v in action_contributions.items()}

        result = []
        for action, contribution in sorted(action_contributions.items(), key=lambda x: -x[1]):
            if action not in {'INIT_STATE', 'FINAL_STATE'}:
                result.append((action, contribution))

        return result

    def combined_ranking_step_wise(self):
        td_contributions = dict(self.estimate_action_contributions())
        combined_scores = defaultdict(float)

        for action in td_contributions:
            if action not in {'INIT_STATE', 'FINAL_STATE'}:
                combined_scores[action] = td_contributions.get(action, 0)

        stepwise_results = defaultdict(list)
        for edge in self.graph.edges:
            pre_action = edge[0]
            post_action = edge[1]
            if post_action in combined_scores:
                score = combined_scores[post_action]
                stepwise_results[pre_action].append((post_action, score))

        for pre_action in stepwise_results:
            stepwise_results[pre_action] = sorted(stepwise_results[pre_action], key=lambda x: -x[1])

        return dict(stepwise_results)
