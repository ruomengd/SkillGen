from torch.utils.data import Dataset, Subset
import json
from environment.babyai_env import BabyAI



class BabyAIDataset(Dataset):
    def __init__(self, label_path, env_step_limit=20, env_num_per_task=4, seed=1234, 
                game_level=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 29, 31, 32],
                selected_tasks=None):
        """Initialize dataset with task definitions from label_path."""
        self.label_path = label_path
        self.env_step_limit = env_step_limit
        self.env_num_per_task = env_num_per_task
        self.seed = seed
        self.game_level = game_level
        self.label_path = label_path
        self.env_configs = self.get_all_environment_configs()

        if selected_tasks is None:  # Load full dataset if no subset is provided
            self.tasks = [i for i in range(len(self.env_configs))]
        else:
            self.tasks = selected_tasks  # Assign subset of tasks

    def load_annotation(self, path):
        all_annotations = []
        difficulty = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    continue
                line = json.loads(line.strip())
                if "subgoals" in line and "subgoals_1" not in line:
                    all_annotations.append(line["subgoals"])
                else:
                    annotation = []
                    for key in line:
                        if "subgoals" in key:
                            annotation.append(line[key])
                    all_annotations.append(annotation)
                
                if "difficulty" in line:
                    difficulty.append(line["difficulty"])
                else:
                    raise ValueError("No difficulty in annotation file")
        return all_annotations, difficulty

    def get_all_environment_configs(self):
        iter_num = 0
        env_configs = []
        self.seeds = range(self.seed, self.seed + self.env_num_per_task)
        obs_to_reward_list, difficulties = self.load_annotation(self.label_path)
        assert len(self.game_level)*self.env_num_per_task == len(obs_to_reward_list)
        for level in self.game_level:
            for seed in self.seeds:
                env_configs.append({
                    "game_level": level,
                    "seed": seed,
                    "obs_to_reward": obs_to_reward_list[iter_num],
                    "difficulty": difficulties[iter_num],
                })
                iter_num += 1
                
        return env_configs
    

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        """Creates a new ScienceWorld instance for each task."""
        task_id = self.tasks[index]
        env = BabyAI.from_config(self.env_configs[task_id])
        game_name = env.game_name
        init_obs = env._get_obs()
        goal = env._get_goal()

        return {
            "task_uid": task_id,
            "task_name": game_name, 
            "goal": goal,
            "trajectory": [('OBSERVATION', init_obs)],
            "progress": [],
            "grounding": [],
            "num_step": 0,
            "grounding_rate": 0,
            "progress_rate": 0,
            "env": env,  # Specific env instance for each sample
        }
    
    def select(self, start_idx, end_idx):
        """Returns a subset of the dataset between start_idx and end_idx."""
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(self))
        selected_tasks = self.tasks[start_idx:end_idx]  # Subset of task definitions
        return BabyAIDataset(self.label_path, self.env_step_limit, selected_tasks=selected_tasks)  # New dataset instance

    def load_from_fold(self, id_list):
        return BabyAIDataset(self.label_path, self.env_step_limit, selected_tasks=id_list)  # New dataset instance

     