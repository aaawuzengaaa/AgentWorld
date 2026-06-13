# Evaluation Module Usage Guide

## 1. File Location

To avoid package import errors and ensure smooth operation, place the `evaluation_module` directory under `backend/agent/src/browser_agent/`.
If you want to use automated workflow execution, also place the `run_workflow.py` script from `evaluation_module` in the same `backend/agent/src/browser_agent/` directory.

---

## 2. Usage Instructions

### 1. Install Dependencies

Please install all required dependencies as specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### 2. Configure the Evaluation

- Edit the workflow configuration in the JSON file: `evaluation_config.json`.
  (You may delete the existing content and refill it, but **do not change the format**.)
- **Type field**:
  - If the workflow does **not** have a standard answer, set `"type": "open"`.
  - If the workflow **does** have a standard answer, set `"type": "standard_answer"`.
  - For standard answer tasks, place the ground truth files in the corresponding subfolder under `human_outputs/`, and add `_gt` to the filename for distinction.
- **Agent output**:
  - Output files will be automatically saved to the appropriate subfolder under `agent_outputs/`.
  - You must specify the `"output_pattern"` according to the output file type (e.g., for text files: `"*.txt"`).
  - If a workflow produces multiple output files, simply use `"*"` as the pattern.

---

### 3. Running the Scripts

#### Only Run Evaluation

To only run the evaluation (make sure the relevant files are placed in both `human_outputs/` and `agent_outputs/`):

**Usage:**
```bash
# For example, to evaluate workflow3 as defined in the config file:
python -m evaluation_module.run_evaluation --workflow workflow3

# To evaluate all workflows defined in the config file:
python -m evaluation_module.run_evaluation --all
```

---

#### Automated Workflow Execution and Evaluation

To automatically run the agent and then evaluate (ground truth files in `human_outputs/` must be placed manually; agent outputs will be generated automatically):

**Usage:**
```bash
# For example, to run workflow3 as defined in the config file:
python run_workflow.py --workflow workflow3

# To run all workflows as defined in the config file:
python run_workflow.py --workflow all
```

## Update 7.23

add agent-based evaluation

File structure

```markdonw
your workspace
    ├── backend
    │   └── agent  # agent inference engine
    ├── AgentWorld
```

1. change the file/directory path in the prompt in `generate_agent_config.py`, run the script to generate *.json cofig in `AgentWorld/config/agent_based`

```shell
python generate_agent_config.py
```

2. change the file/directory path in `run_agent_eval.py`, run the script to generate evaluation report in `AgentWorld/eval_results`

```shell
python run_agent_eval.py --model-name chatgpt
```

3. change the directory path(`/Users/baiyl/workspace/sophia/git/AgentWorld/eval_results`) in the prompt of `generate_report.py`, run the script to generate overall report in `AgentWorld/eval_results`

```shell
python generate_report.py
```


