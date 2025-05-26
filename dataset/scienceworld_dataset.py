from torch.utils.data import Dataset, Subset
from environment.scienceworld_env import Scienceworld

class ScienceWorldDataset(Dataset):
    def __init__(self, label_path, env_step_limit=20, selected_tasks=None):
        """Initialize dataset with task definitions from label_path."""
        self.label_path = label_path
        self.env_step_limit = env_step_limit
        if selected_tasks is None:  # Load full dataset if no subset is provided
            self.tasks = list(Scienceworld(envStepLimit=env_step_limit, label_path=label_path).labels.items())
        else:
            self.tasks = selected_tasks  # Assign subset of tasks

    def build_simplification_str(self):
        """Defines simplification settings for the environment."""
        simplifications = [
            "selfWateringFlowerPots",
            "openContainers",
            "openDoors",
            "noElectricalAction"
        ]
        return ",".join(simplifications)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        """Creates a new ScienceWorld instance for each task."""
        key, value = self.tasks[index]
        task_name, var, modified_goal = value["task_name"], value["var"], value["modified_goal"]

        # Each sample gets its own environment instance
        env = Scienceworld(envStepLimit=self.env_step_limit, label_path=self.label_path)
        env.load(task_name, var, simplificationStr=self.build_simplification_str())
        initialObs, _ = env.reset()
        init_obs = initialObs + f"\n{env.inventory()}"

        task_uid = "%s_%s"%(task_name, var)

        return {
            "task_uid": task_uid,
            "task_name": task_name,
            "goal": modified_goal,
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
        return ScienceWorldDataset(self.label_path, self.env_step_limit, selected_tasks)  # New dataset instance
