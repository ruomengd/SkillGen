# SkillGen: Learning Domain Skills for In-Context Sequential Decision Making

SkillGen is a framework for extracting, ranking, and reusing **step-wise domain skills** to enhance the performance of large language models (LLMs) in complex multi-step decision-making tasks. It leverages trajectory sampling, TD-based credit assignment, and graph-based skill extraction to build structured prompts that guide LLMs more effectively than standard prompting strategies.


> 🚧 **This repository is currently under active development. Full code and documentation will be released soon.** 🚧


## Project Structure

```
SkillGen/
├── data/                # Datasets for supported environments
├── dataset/             # Dataset loading and preprocessing scripts
├── environment/         # Environment wrappers for AlfWorld, BabyAI, ScienceWorld
├── logs/                # Output logs, extracted skills, embeddings, and results
├── models/              # Pretrained LLMs (e.g., Qwen2.5-7B-Instruct)
├── prompt/              # Prompt construction and skill prompting logic
├── scripts/             # Shell scripts for running extraction, sampling, inference
├── skill_extraction/    # Core skill extraction, embedding, and retrieval modules
├── inference_skillgen.py# Main inference script for skill-based prompting
├── sampling.py          # Script for trajectory sampling
├── utils.py             # Utility functions
└── README.md            # This file
```

## Installation

1. **Requirements**
   ```bash
    conda create -n skillgen python=3.9
    conda activate skillgen
    pip install -r requirements.txt
    ```

2. **Download or prepare datasets:**
   - Place your datasets in the `data/` directory, following the structure for AlfWorld, BabyAI, and ScienceWorld.

4. **Set the environment variable & Download LLM weights:**
   - Place your model files (e.g., Qwen2.5-7B-Instruct) in the `models/` directory.
   - Set API Keys in `SkillGen/.env`.

## Usage

### 1. Trajectory Sampling

Sample trajectories for each environment:
```bash
sh scripts/sampling.sh
```

### 2. Skill Extraction

Extract skills from sampled trajectories:
```bash
sh scripts/skill_extraction.sh
```
This will:
- Extract stwp-wise skills from sampling data
- Retrieve golden segments


### 3. Inference with SkillGen

Run inference using skill-based prompts:
```bash
sh scripts/inference_skillgen.sh
```


## Output

- Extracted skills, embeddings, and logs are saved in the `logs/` directory.
- Inference results are saved in `logs/inference_skillgen/` (or as specified by `--save_path`).
- **Note:** The logs of the main results reported in the paper are stored in `SkillGen/logs/inference_skillgen`.

