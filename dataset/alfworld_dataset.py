from torch.utils.data import Dataset, Subset
import yaml
from environment.alfworld.alfworld_env import AlfWorld
from alfworld.agents.environment import get_environment

class AlfWorldDataset(Dataset):
    def __init__(self, label_path, env_step_limit=20, selected_tasks=None, from_fold=None, batch_size=1, seed=42, 
                split='eval_out_of_distribution',
                base_config='./environment/alfworld/base_config.yaml'):
        """Initialize dataset with task definitions from label_path."""
        self.label_path = label_path
        self.env_step_limit = env_step_limit
        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.base_config = base_config

        with open(self.base_config) as reader:
            self.config = yaml.safe_load(reader)
        
        # env_wrap = getattr(alfworld.agents.environment, self.config["env"]["type"])(self.config, train_eval=split)
        env_class = get_environment(self.config["env"]["type"])
        env_wrap = env_class(self.config, train_eval=split)
        env_wrap.game_files.sort()
        self.total_game_files = env_wrap.game_files

        if selected_tasks is None:  # Load full dataset if no subset is provided
            self.tasks = [i for i in range(len(env_wrap.game_files))]
            self.game_files = env_wrap.game_files
        else:
            self.tasks = selected_tasks  # Assign subset of tasks
            self.game_files = [env_wrap.game_files[i] for i in self.tasks]
        
        if from_fold is not None:
            new_game_files, new_tasks = [], []
            for idx, game in enumerate(self.game_files):
                task_name = '/'.join(game.split('/')[-3:-1])
                if task_name in from_fold:
                    new_game_files.append(task_name)
                    new_tasks.append(idx)
            self.tasks = new_tasks
            self.game_files = new_game_files
      
            
    def __len__(self):
        return len(self.game_files)

    def __getitem__(self, index):
        """Creates a new ScienceWorld instance for each task."""
        task_id = self.tasks[index]
        # env_wrap = getattr(alfworld.agents.environment, self.config["env"]["type"])(self.config, train_eval=split)
        env_class = get_environment(self.config["env"]["type"])
        env_wrap = env_class(self.config, train_eval=self.split)
        env_wrap.game_files.sort()
        env = AlfWorld(task_id=task_id, env=env_wrap, split=self.split, base_config=self.base_config, 
                       batch_size=1, seed=self.seed, label_path=self.label_path)

        ob, info = env.reset()
        init_obs = '\n'.join(ob[0].split('\n\n')[1:])
        task_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        goal = env.goal.split('Your task is to:')[1].replace('.', '').strip()
        
        return {
            "task_uid": task_name,
            "task_name": task_name,
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
        return AlfWorldDataset(self.label_path, self.env_step_limit, selected_tasks=selected_tasks)  # New dataset instance

    def load_from_fold(self, id_list):
        return AlfWorldDataset(self.label_path, self.env_step_limit, from_fold=id_list)  # New dataset instance
    
   