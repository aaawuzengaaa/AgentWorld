" Useage: python run_workflow.py --workflow workflown ; python run_workflow.py --workflow all "

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import asyncio
import datetime
import traceback
import re
import glob
import shutil
import subprocess

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Path config
eval_module_path = Path(__file__).parent / "evaluation_module"
config_path = eval_module_path / "evaluation_config.json"
agent_output_path = eval_module_path / "agent_outputs"
log_path = eval_module_path / "logs"
run_eval_path = eval_module_path / "run_evaluation.py"

# Log config
log_path.mkdir(exist_ok=True)
log_file = log_path / f"run_workflow_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

def load_workflow_config():
    if not config_path.exists():
        logging.error(f"Config file does not exist: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "workflows" not in config or not isinstance(config["workflows"], dict):
            raise ValueError("Config file missing 'workflows' field or format error")
        return config["workflows"]
    except Exception as e:
        logging.error(f"Failed to read config file: {e}")
        sys.exit(1)

def check_workflow_names(config, workflow_names):
    invalid = [w for w in workflow_names if w not in config]
    if invalid:
        logging.error(f"Invalid workflow name(s): {invalid}")
        sys.exit(1)

def get_task_instruction(workflow_cfg):
    task_config = workflow_cfg.get("task_config", {})
    query = task_config.get("query", {})
    diligent = query.get("diligent", "").strip()
    lazy = query.get("lazy", "").strip()
    deliverable = task_config.get("deliverable", "").strip()
    # Prefer diligent, then lazy
    if diligent:
        instruction = diligent
    elif lazy:
        instruction = lazy
    else:
        instruction = ""
    # Append deliverable requirements
    if deliverable:
        instruction += f"\n\nDeliverable: {deliverable}"
    return instruction.strip()

def extract_actual_content(result, track_dir):
    # Try regex match for output file hints
    patterns = [
        r"saved as '([^']+\.(txt|csv|md))'",
        r"saved to '([^']+\.(txt|csv|md))'",
        r"saved at[：: ]*([\w\-\.]+\.(txt|csv|md))",
        r"see ([^\s]+\.(txt|csv|md))",
        r"filename[：: ]*([\w\-\.]+\.(txt|csv|md))"
    ]
    for pat in patterns:
        match = re.search(pat, result, re.IGNORECASE)
        if match:
            actual_file = match.group(1)
            actual_file_path = track_dir / actual_file
            if actual_file_path.exists():
                with open(actual_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logging.warning(f"Actual output file not found: {actual_file_path}")
    return result  # If not matched, return original content

def is_target_output_file(filename):
    # Only keep target output types
    return filename.endswith((".txt", ".csv", ".md")) and not (
        filename.endswith(".log") or
        filename.endswith(".json") or
        "trajectory" in filename or
        "debug" in filename
    )

async def run_agent(task_instruction, workflow_name):
    try:
        from agent.src.browser_agent.agents import BrowserAgent
        from agent.src.smolagents import OpenAIServerModel
    except ImportError as e:
        logging.error(f"Failed to import agent modules: {e}")
        sys.exit(1)

    # Log and trace files are stored separately
    track_dir = log_path / f"{workflow_name}_track"
    track_dir.mkdir(exist_ok=True)

    model = OpenAIServerModel(
        model_id="google/gemini-2.5-flash",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3,
        max_tokens=8000,
    )
    agent = BrowserAgent(
        model=model,
        headless=True,
        max_steps=80,
        use_redis=True,
        tokens_dict={},
        stream_outputs=True,
        additional_authorized_imports=["*"],
        executor_kwargs={
            "work_dir": str(track_dir)
        }
    )
    try:
        await agent.run_task_simple(task_instruction)
    except Exception as e:
        logging.error(f"Agent execution failed: {e}\n{traceback.format_exc()}")
        return None

    # Copy target output files to agent_outputs
    workflow_output_dir = agent_output_path / workflow_name
    workflow_output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for file in track_dir.iterdir():
        if file.is_file() and is_target_output_file(file.name):
            shutil.copy2(file, workflow_output_dir / file.name)
            copied_files.append(file.name)
    if copied_files:
        logging.info(f"Copied output files to {workflow_output_dir}: {copied_files}")
    else:
        logging.warning(f"No target output files found, nothing copied to {workflow_output_dir}")
    return copied_files

def run_evaluation(workflow_name):
    eval_module_dir = eval_module_path.parent  # Parent directory of evaluation_module
    cmd = [
        sys.executable,
        "-m", "evaluation_module.run_evaluation",
        "--workflow", workflow_name
    ]
    result = subprocess.run(cmd, cwd=eval_module_dir, capture_output=True, text=True)
    logging.info(f"Evaluation result:\n{result.stdout}")
    if result.returncode != 0:
        logging.error(f"Evaluation script error: {result.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Automatically run specified workflow(s) and evaluate")
    parser.add_argument("--workflow", nargs="+", help="Workflow name(s) to run (e.g. workflow1 workflow2)")
    parser.add_argument("--all", action="store_true", help="Run all workflows")
    args = parser.parse_args()

    config = load_workflow_config()
    if args.all:
        workflow_names = list(config.keys())
    elif args.workflow:
        workflow_names = args.workflow
        check_workflow_names(config, workflow_names)
    else:
        logging.error("Please specify workflow(s) with --workflow or use --all to run all workflows.")
        sys.exit(1)

    for workflow_name in workflow_names:
        logging.info(f"==== Start workflow: {workflow_name} ====")
        workflow_cfg = config[workflow_name]
        task_instruction = get_task_instruction(workflow_cfg)
        if not task_instruction:
            logging.error(f"workflow {workflow_name} missing valid query field")
            continue
        copied_files = asyncio.run(run_agent(task_instruction, workflow_name))
        if copied_files:
            run_evaluation(workflow_name)
        logging.info(f"==== Finished workflow: {workflow_name} ====")

if __name__ == "__main__":
    main() 