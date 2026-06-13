import os
import json
import asyncio
from evaluators import AgentBasedEvaluator
import argparse


def run_agent_eval(model_name: str):
    agent_output_dir = f"/Users/baiyl/workspace/sophia/git/AgentWorld/eval_results/{model_name}"
    config_dir = "/Users/baiyl/workspace/sophia/git/AgentWorld/config/agent_based_gt"
    deliverable_dir = f"/Users/baiyl/workspace/sophia/git/AgentWorld/deliverable_data/{model_name}"
    for config_file in os.listdir(config_dir):
        agent = AgentBasedEvaluator()
        config_path = os.path.join(config_dir, config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        task_name = list(config.keys())[0]
        workflow_config = config[task_name]["workflow_agent_eval"]
        # print(workflow_config)
        task_config = workflow_config.get('task_config', {})
        evaluation_config = workflow_config.get('evaluation_config', {})

        
        evaluation_config['query_id'] = task_name
        full_path = os.path.join(deliverable_dir, task_name)
        if os.path.exists(full_path):
            file_count = len(os.listdir(full_path))
        else:
            file_count = 0

        do_run = False
        if "sophia" in model_name:
            if file_count > 4:
                do_run = True
        else:
            if file_count >= 1:
                do_run = True


        if do_run:
            output = []
            for file in os.listdir(full_path):
                if "lazy_query.md" not in file and "evaluation_criteria.md" not in file and "delivrable.md" not in file and "deligent_query" not in file:
                    output.append(os.path.join(full_path, file))
            evaluation_config['output_files'] = output

            # 如果已经保存了，则不重复保存
            task_dir = os.path.join(agent_output_dir, task_name)
            if os.path.exists(task_dir):
                if len(os.listdir(task_dir)) == 3:
                    continue
            

            asyncio.run(agent.evaluate_task(task_config, agent_output_dir, evaluation_config))
            # break
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--model-name",
        type=str,
        default="manus",
        help="The model name to use for evaluation"
    )
    args = parser.parse_args()
    
    run_agent_eval(args.model_name)
