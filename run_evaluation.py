
"""
Intelligent evaluation system runner script
Usage:
    python run_evaluation.py --workflow workflow1
    python -m evaluation_module.run_evaluation --workflow workflow3
    python run_evaluation.py --all
"""

import sys
from pathlib import Path
import os

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from .evaluation_engine import EvaluationEngine

def main():
    """Main function"""
    import argparse
    try:
        parser = argparse.ArgumentParser(description="Automated Evaluation Module")
        parser.add_argument('--workflow', type=str, required=True, help='Name of the workflow to evaluate')
        parser.add_argument('--config', type=str, default=None, help='Path to the config file')
        parser.add_argument('--all', action='store_true', help='Evaluate all workflows')
        parser.add_argument('--list', action='store_true', help='List all available workflows')
        args = parser.parse_args()

        # Smartly find config file path, always find it
        if args.config:
            config_path = args.config
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.path.dirname(__file__), config_path)
        else:
            config_path = os.path.join(os.path.dirname(__file__), 'evaluation_config.json')
        if not os.path.exists(config_path):
            print(f'Error: Config file does not exist: {config_path}')
            print('Please make sure evaluation_config.json exists')
            exit(1)

        engine = EvaluationEngine(config_path)
        
        if args.list:
            # List all workflows
            print("Available workflows:")
            for workflow_name, workflow_config in engine.config['workflows'].items():
                print(f"  - {workflow_name}: {workflow_config['name']}")
                print(f"    Description: {workflow_config['description']}")
                print(f"    Type: {workflow_config.get('type', 'unknown')}")
                task_config = workflow_config.get('task_config', {})
                deliverable = task_config.get('deliverable', 'N/A')
                print(f"    Deliverable: {deliverable}")
                if 'tasks' in workflow_config:
                    print(f"    Task count: {len(workflow_config['tasks'])}")
                else:
                    print(f"    Task count: 1 (single task workflow)")
                print()
            return
        
        if args.all:
            # Evaluate all workflows
            print("Starting evaluation of all workflows...")
            result = engine.evaluate_all_workflows()
            print(f"\nEvaluation completed!")
            print(f"Overall average score: {result['overall_summary']['average_score']:.4f}")
            print(f"Successful workflows: {result['overall_summary']['successful_workflows']}")
            print(f"Failed workflows: {result['overall_summary']['failed_workflows']}")
            print(f"\nScores for each workflow:")
            for workflow_name, workflow_result in result['workflow_results'].items():
                if workflow_result.get('success', False):
                    print(f"  {workflow_name}: {workflow_result.get('final_score', 0.0):.4f}")
                else:
                    print(f"  {workflow_name}: Failed - {workflow_result.get('error', 'Unknown error')}")
            
        elif args.workflow:
            # Evaluate specified workflow
            print(f"Starting evaluation for workflow: {args.workflow}")
            result = engine.evaluate_workflow(args.workflow)
            
            # Optimize judgment logic: all tasks must be successful for overall success
            all_success = all(task['result'].get('success', False) for task in result.get('task_results', []))
            if all_success:
                print(f"\nEvaluation completed!")
                print(f"Workflow: {args.workflow}")
                print(f"Final score: {result['final_score']:.4f}")
                print(f"Task count: {result['summary']['total_tasks']}")
                print(f"Successful tasks: {result['summary']['successful_tasks']}")
                print(f"Failed tasks: {result['summary']['failed_tasks']}")
                print(f"\nScores for each task:")
                for task_result in result['task_results']:
                    task_name = task_result['task_name']
                    task_result_data = task_result['result']
                    if task_result_data['success']:
                        print(f"  {task_name}: {task_result_data['score']:.4f}")
                    else:
                        print(f"  {task_name}: Failed - {task_result_data.get('error', 'Unknown error')}")
            else:
                print(f"Evaluation failed: {result.get('error', 'Unknown error')}")
        else:
            print("Please specify --workflow or use --all to evaluate all workflows")
            print("Use --list to see all available workflows")
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 