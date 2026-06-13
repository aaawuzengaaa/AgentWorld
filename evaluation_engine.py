import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

from .evaluators import StandardAnswerEvaluator, RuleBasedEvaluator, AgentBasedEvaluator

class EvaluationEngine:
    """
    Evaluation engine, unified management of different types of evaluation tasks
    """
    
    def __init__(self, config_path: str):
        self.base_dir = Path(__file__).parent.resolve()
        self.config = self._load_config(config_path)
        self.global_settings = self.config.get('global_settings', {})
        # Optimize all directories as absolute paths
        self.agent_output_dir = self._get_abs_dir(self.global_settings.get('agent_output_dir', 'agent_outputs'))
        self.human_output_dir = self._get_abs_dir(self.global_settings.get('human_output_dir', 'human_outputs'))
        self.evaluation_output_dir = self._get_abs_dir(self.global_settings.get('evaluation_output_dir', 'evaluation_results'))
        self.logger = self._setup_logging()
        
        # Initialize evaluators
        self.standard_evaluator = StandardAnswerEvaluator(self.config)
        self.rule_evaluator = RuleBasedEvaluator()
        self.agent_evaluator = AgentBasedEvaluator()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load evaluation config
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load config file: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logger
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = Path("evaluation_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def evaluate_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """
        Evaluate the specified workflow
        """
        self.logger.info(f"Start evaluating workflow: {workflow_name}")
        
        if workflow_name not in self.config['workflows']:
            return {
                'success': False,
                'error': f'Workflow config not found: {workflow_name}'
            }
        
        workflow_config = self.config['workflows'][workflow_name]
        global_settings = self.config['global_settings']
        
        # Set directory paths
        agent_output_dir = self.agent_output_dir / workflow_name
        human_output_dir = self.human_output_dir / workflow_name
        evaluation_output_dir = self.evaluation_output_dir / workflow_name
        
        if not agent_output_dir.exists():
            return {
                'success': False,
                'error': f'Agent output directory does not exist: {agent_output_dir}'
            }
        
        # Only supports new structures. Workflow itself is a task. 
        tasks_to_evaluate = [workflow_config]
        
        # Evaluate all tasks
        task_results = []
        total_score = 0.0
        
        for task_config in tasks_to_evaluate:
            self.logger.info(f"Evaluating task: {task_config['name']}")
            
            if task_config['type'] == 'standard_answer':
                result = self.standard_evaluator.evaluate_task(
                    task_config, agent_output_dir, human_output_dir
                )
            elif task_config['type'] == 'open':
                result = self.rule_evaluator.evaluate_task(
                    task_config, agent_output_dir, human_output_dir
                )
            elif task_config['type'] == 'agent':
                result = self.agent_evaluator.evaluate_task(
                    task_config, agent_output_dir, human_output_dir
                )
            else:
                result = {
                    'success': False,
                    'error': f'Unknown task type: {task_config["type"]}',
                    'score': 0.0
                }
            
            if result['success']:
                total_score += result['score']
            task_results.append({
                'task_id': task_config.get('id', task_config['name']),
                'task_name': task_config['name'],
                'task_type': task_config['type'],
                'result': result
            })
        
        # Final score
        final_score = total_score / len(task_results) if len(task_results) > 0 else 0.0
        
        # Generate evaluation report
        evaluation_report = {
            'workflow_name': workflow_name,
            'workflow_config': workflow_config,
            'evaluation_time': datetime.now().isoformat(),
            'final_score': final_score,
            'task_results': task_results,
            'summary': {
                'total_tasks': len(task_results),
                'successful_tasks': sum(1 for tr in task_results if tr['result']['success']),
                'failed_tasks': sum(1 for tr in task_results if not tr['result']['success']),
                'average_score': final_score
            }
        }
        
        # Save evaluation result
        self._save_evaluation_report(evaluation_report, workflow_name)
        
        return evaluation_report
    
    def evaluate_all_workflows(self) -> Dict[str, Any]:
        """
        Evaluate all configured workflows
        """
        self.logger.info("Start evaluating all workflows")
        
        all_results = {}
        overall_summary = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_score': 0.0
        }
        
        total_score = 0.0
        workflow_count = 0
        
        for workflow_name in self.config['workflows']:
            self.logger.info(f"Evaluating workflow: {workflow_name}")
            result = self.evaluate_workflow(workflow_name)
            all_results[workflow_name] = result
            
            if result.get('success', False):
                overall_summary['successful_workflows'] += 1
                total_score += result.get('final_score', 0.0)
            else:
                overall_summary['failed_workflows'] += 1
            
            workflow_count += 1
        
        overall_summary['total_workflows'] = workflow_count
        overall_summary['average_score'] = total_score / workflow_count if workflow_count > 0 else 0.0
        
        # Save overall evaluation report
        overall_report = {
            'evaluation_time': datetime.now().isoformat(),
            'overall_summary': overall_summary,
            'workflow_results': all_results
        }
        
        self._save_overall_report(overall_report)
        
        return overall_report
    
    def _save_evaluation_report(self, report: Dict[str, Any], workflow_name: str):
        """
        Save evaluation report for a single workflow
        """
        output_dir = self.evaluation_output_dir
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_file = output_dir / f"{workflow_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Generate readable text report
        text_file = output_dir / f"{workflow_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._generate_text_report(report, text_file)
        
        self.logger.info(f"Evaluation report saved: {json_file}, {text_file}")
    
    def _save_overall_report(self, report: Dict[str, Any]):
        """
        Save overall evaluation report
        """
        output_dir = self.evaluation_output_dir
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_file = output_dir / f"overall_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Generate readable text report
        text_file = output_dir / f"overall_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._generate_overall_text_report(report, text_file)
        
        self.logger.info(f"Overall evaluation report saved: {json_file}, {text_file}")
    
    def _generate_text_report(self, report: Dict[str, Any], output_file: Path):
        """
        Generate readable text report for a workflow
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Workflow Evaluation Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Workflow Name: {report['workflow_name']}\n")
            f.write(f"Evaluation Time: {report['evaluation_time']}\n")
            f.write(f"Final Score: {report['final_score']:.4f}\n")
            f.write(f"\n")
            
            f.write(f"Task Evaluation Details:\n")
            f.write(f"-" * 30 + "\n")
            
            for task_result in report['task_results']:
                f.write(f"Task: {task_result['task_name']}\n")
                f.write(f"Type: {task_result['task_type']}\n")
                
                result = task_result['result']
                if result['success']:
                    f.write(f"Score: {result['score']:.4f}\n")
                    if 'summary' in result:
                        for key, value in result['summary'].items():
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"Error: {result['error']}\n")
                
                f.write(f"\n")
            
            f.write(f"Overall Statistics:\n")
            f.write(f"-" * 20 + "\n")
            summary = report['summary']
            f.write(f"Total Tasks: {summary['total_tasks']}\n")
            f.write(f"Successful Tasks: {summary['successful_tasks']}\n")
            f.write(f"Failed Tasks: {summary['failed_tasks']}\n")
            f.write(f"Average Score: {summary['average_score']:.4f}\n")
    
    def _generate_overall_text_report(self, report: Dict[str, Any], output_file: Path):
        """
        Generate overall evaluation text report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Overall Evaluation Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Evaluation Time: {report['evaluation_time']}\n")
            f.write(f"\n")
            
            f.write(f"Overall Statistics:\n")
            f.write(f"-" * 20 + "\n")
            summary = report['overall_summary']
            f.write(f"Total Workflows: {summary['total_workflows']}\n")
            f.write(f"Successful Workflows: {summary['successful_workflows']}\n")
            f.write(f"Failed Workflows: {summary['failed_workflows']}\n")
            f.write(f"Average Score: {summary['average_score']:.4f}\n")
            f.write(f"\n")
            
            f.write(f"Workflow Details:\n")
            f.write(f"-" * 30 + "\n")
            
            for workflow_name, workflow_result in report['workflow_results'].items():
                f.write(f"Workflow: {workflow_name}\n")
                if workflow_result.get('success', False):
                    f.write(f"  Score: {workflow_result.get('final_score', 0.0):.4f}\n")
                    f.write(f"  Task Count: {workflow_result.get('summary', {}).get('total_tasks', 0)}\n")
                else:
                    f.write(f"  Error: {workflow_result.get('error', 'Unknown error')}\n")
                f.write(f"\n")

    def _get_abs_dir(self, dir_path):
        p = Path(dir_path)
        if not p.is_absolute():
            return self.base_dir / p
        return p

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Intelligent Evaluation Engine")
    parser.add_argument('--workflow', help='Specify the workflow name to evaluate')
    parser.add_argument('--config', default='evaluation_config.json', help='Evaluation config file path')
    parser.add_argument('--all', action='store_true', help='Evaluate all workflows')
    
    args = parser.parse_args()
    
    try:
        # Create evaluation engine
        engine = EvaluationEngine(args.config)
        
        if args.all:
            # Evaluate all workflows
            result = engine.evaluate_all_workflows()
            print(f"Overall evaluation completed, average score: {result['overall_summary']['average_score']:.4f}")
        elif args.workflow:
            # Evaluate specified workflow
            result = engine.evaluate_workflow(args.workflow)
            if result.get('success', False):
                print(f"Workflow {args.workflow} evaluation completed, score: {result['final_score']:.4f}")
            else:
                print(f"Workflow {args.workflow} evaluation failed: {result.get('error', 'Unknown error')}")
        else:
            print("Please specify --workflow or use --all to evaluate all workflows")
            
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
