# VLM-KTAMP

This repository contains the implementation of our paper [Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling](https://arxiv.org/abs/2510.26139), accepted at _IEEE International Conference on Robotics and Automation (ICRA) 2026_.

Please visit our [project website](https://graphics.ewha.ac.kr/KinodynamicTAMP/) for more details.



## Setup

### 1. Clone the Repository with Submodules

If you are cloning the project for the first time, use:

```bash
git clone https://github.com/Minseo10/VLM-KTAMP.git
cd VLM-KTAMP
git submodule update --init --recursive
```

### 2. Create a Conda Environment

This codebase is currently used with Python 3.10 on Linux.

```bash
conda env create -f environment.yml
conda activate genesis_new
```

### 3. Install Genesis

We use [Genesis](https://genesis-world.readthedocs.io/en/latest/) for physics simulation.

```bash
git clone https://github.com/Minseo10/genesis-world.git
cd Genesis
pip install -e '.[render]'
```

### 4. Build Fast Downward Inside PDDLStream

This repository vendors `pddlstream`, and its `downward` directory must be built before running planners that depend on Fast Downward.

```bash
cd VLM-KTAMP/pddlstream
git submodule update --init --recursive
./downward/build.py
cd ..
```

### 5. Configure API Keys

The root VLM-KTAMP code reads the OpenAI key from `config.json`:

```json
{
	"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
	"org": ""
}
```

The `LLM-TAMP` submodule also expects its own key file at `LLM-TAMP/openai_keys/openai_key.json`:

```json
{
	"OPENAI_API_KEY": "YOUR_OPENAI_API_KEY",
	"org": ""
}
```

You can leave `org` blank.


## Running Experiments


```bash
python run.py --help
```

The currently supported experiment modes are:

- `ours`: our VLM-KTAMP planner
- `ours_ablation`: ablation variant of our planner without VLM backtracking
- `llm3`: an LLM-based TAMP baseline: https://arxiv.org/abs/2403.11552
- `pddlstream`: a traditional TAMP baseline: https://arxiv.org/abs/1802.08705

### Run Our Planner

```bash
python run.py \
	--domain blocksworld_pr \
	--method ours \
	--prob_complexity 3 4 5 6 \
	--prob_idx 1 2 3 4 5 \
	--trial_range 1 2 \
	--K 5 \
	--plan_number 30 \
	--timeout_seconds 600 \
	--model gpt-4o \
	--vis_sim True
```

### Run the Ablation Variant

```bash
python run.py \
	--domain blocksworld_pr \
	--method ours_ablation \
	--prob_complexity 3 4 5 6 \
	--prob_idx 1 2 3 4 5 \
	--trial_range 1 2 \
	--K 5 \
	--plan_number 30 \
	--timeout_seconds 600 \
	--model gpt-4o \
	--vis_sim True
```


### Run the LLM-TAMP Baseline

```bash
python run.py \
	--domain blocksworld_pr \
	--method llm3 \
	--prob_complexity 3 4 5 6 \
	--prob_idx 1 2 3 4 5 \
	--trial_range 1 2 \
	--timeout_seconds 600 \
	--model gpt-4o
```

### Run the PDDLStream Baseline

```bash
python run.py \
	--domain blocksworld_pr \
	--method pddlstream \
	--prob_complexity 3 4 5 6 \
	--prob_idx 1 2 3 4 5 \
	--trial_range 1 2 \
	--timeout_seconds 600
```


## Parameters

The root runner `run.py` supports the following arguments.

- `domain`: experiment domain. Current choices are `blocksworld_pr` and `kitchen`.
- `method`: experiment method. Current choices are `ours`, `ours_ablation`, `llm3`, and `pddlstream`.
- `prob_complexity`: list of problem complexities to run. The supported range is 3 to 6.
- `prob_idx`: list of problem instance indices. Supported values are 1 to 5 for `blocksworld_pr`, and 1 for `kitchen`.
- `trial_range`: list of trial indices to run. Object configurations differ across trials. Supported values are 1 to 2 for `blocksworld_pr`, and 1 to 10 for `kitchen`.
- `K`: maximum number of randomized replanning attempts. Default is 5.
- `plan_number`: number of candidate symbolic plans generated for discrete state graph generation. Default is 30.
- `timeout_seconds`: timeout for each run. Default is 600 seconds.
- `model`: VLM model name. Default is `gpt-4o`.
- `vis_sim`: whether to open the Genesis visualization during execution. Default is `True`. This is intended for `ours` and `ours_ablation`.


## Contact
Please contact `tahitiro2@gmail.com` if you have any questions.


## Citation


IEEE ICRA 2026, [Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling](https://arxiv.org/abs/2510.26139)

```
@article{kwon2025kinodynamic,
  title={Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling},
  author={Kwon, Minseo and Kim, Young J},
  journal={arXiv preprint arXiv:2510.26139},
  year={2025}
}
```