import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Any
import logging
import re

# Add smolagents to Python path for imports
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent  # From evaluators -> evaluation_module -> browser_agent -> src
smolagents_dir = src_dir / "smolagents"
if smolagents_dir.exists() and str(smolagents_dir) not in sys.path:
    sys.path.insert(0, str(smolagents_dir))

class RuleBasedEvaluator:
    """
    Universal LLM-based evaluator for tasks without standard answers
    Supports various output formats including text, CSV, JSON, etc.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize LLM model for evaluation
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            try:
                from smolagents import OpenAIServerModel
                self.model = OpenAIServerModel(
                    model_id="google/gemini-2.5-pro",
                    api_base="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                    temperature=0.1,
                    max_tokens=6000,
                )
            except ImportError as e:
                self.model = None
                self.logger.error(f"Failed to import OpenAIServerModel: {e}")
        else:
            self.model = None
            self.logger.error("OPENROUTER_API_KEY not set, cannot perform LLM evaluation")
    
    def evaluate_task(self, task_config: Dict[str, Any], agent_output_dir: Path, human_output_dir: Path) -> Dict[str, Any]:
        """
        Universal evaluator for open-ended tasks (type=open)
        Supports all output formats with automatic format detection by LLM
        Now supports multiple files evaluation
        """
        self.logger.info(f"Starting LLM evaluation for task: {task_config['name']}")
        
        # Support both old and new JSON structure
        eval_config = task_config.get('evaluation_config', {})
        file_matching = eval_config.get('file_matching', {})
        if file_matching:
            # New structure with auto file detection - now supports multiple files
            agent_files = self._find_all_files_by_pattern(agent_output_dir, file_matching.get('output_pattern', ''))
        else:
            # Old structure with explicit file name
            output_file = task_config.get('output_file', '')
            agent_file = self._find_agent_output_file(agent_output_dir, output_file)
            agent_files = [agent_file] if agent_file else []
        
        if not agent_files:
            return {
                'success': False,
                'error': f'No agent output files found',
                'score': 0.0,
                'summary': {}
            }
        
        # Only process open-ended tasks (type=open)
        if task_config.get('type') != 'open':
            return {
                'success': False,
                'error': f'This task type ({task_config.get("type", "unknown")}) is not supported. Only type=open tasks are supported.',
                'score': 0.0,
                'summary': {}
            }
        
        # For open-ended tasks, use LLM evaluation with all files
        if self.model:
            evaluation_result = self._evaluate_with_llm_universal_multiple_files(task_config, agent_files)
            if not evaluation_result["success"]:
                return evaluation_result
        else:
            return {
                'success': False,
                'error': 'LLM not available, cannot perform evaluation',
                'score': 0.0,
                'summary': {}
            }

        return {
            'success': True,
            'task_name': task_config['name'],
            'output_files': [str(f) for f in agent_files],
            'score': evaluation_result['score'],
            'details': evaluation_result['details'],
            'summary': evaluation_result['summary']
        }
    
    def _find_agent_output_file(self, agent_output_dir: Path, output_file: str) -> Path:
        """
        Find agent output file with fuzzy matching
        """
        # Directly match file name
        file_path = agent_output_dir / output_file
        if file_path.exists():
            return file_path
        
        # Fuzzy match (ignore case and spaces)
        output_file_lower = output_file.lower().replace(' ', '')
        for file_path in agent_output_dir.glob("*"):
            if file_path.is_file():
                file_name_lower = file_path.name.lower().replace(' ', '')
                if output_file_lower in file_name_lower or file_name_lower in output_file_lower:
                    return file_path
        return None
    
    def _find_file_by_pattern(self, directory: Path, pattern: str) -> Path:
        """
        Find a file in the given directory that matches a given pattern.
        The pattern can be a simple filename or a more complex glob pattern.
        """
        if not directory.exists():
            return None
            
        if '*' in pattern or '?' in pattern:
            # Handle glob patterns
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    return file_path
        else:
            # Handle simple filename matching
            file_path = directory / pattern
            if file_path.exists() and file_path.is_file():
                    return file_path
        
            # Fallback to fuzzy matching
            pattern_lower = pattern.lower().replace(' ', '')
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_name_lower = file_path.name.lower().replace(' ', '')
                    if pattern_lower in file_name_lower or file_name_lower in pattern_lower:
                        return file_path
        return None
    
    def _find_all_files_by_pattern(self, directory: Path, pattern: str) -> List[Path]:
        """
        Find all files in the given directory that match a given pattern.
        The pattern can be a simple filename or a more complex glob pattern.
        Returns a list of all matching files.
        """
        if not directory.exists():
            return []
            
        matching_files = []
        
        if '*' in pattern or '?' in pattern:
            # Handle glob patterns
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    matching_files.append(file_path)
        else:
            # Handle simple filename matching
            file_path = directory / pattern
            if file_path.exists() and file_path.is_file():
                matching_files.append(file_path)
        
            # Fallback to fuzzy matching
            pattern_lower = pattern.lower().replace(' ', '')
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_name_lower = file_path.name.lower().replace(' ', '')
                    if pattern_lower in file_name_lower or file_name_lower in pattern_lower:
                        matching_files.append(file_path)
        
        # Sort files by name for consistent ordering
        matching_files.sort(key=lambda x: x.name)
        return matching_files
    
    def _read_output_file(self, file_path: Path) -> str:
        """
        Read file content with multiple encoding support
        """
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Failed to read file {file_path}: {e}")
                return ""
        self.logger.error(f"Failed to read file {file_path}: all encoding attempts failed")
        return ""
    
    def _read_csv_file(self, file_path: Path) -> str:
        """
        Read CSV file and convert to readable text format
        """
        try:
            content = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:  # Header row
                        content.append("Headers: " + " | ".join(row))
                    else:
                        content.append("Row " + str(i) + ": " + " | ".join(row))
            return "\n".join(content)
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {file_path}: {e}")
            return self._read_output_file(file_path)  # Fallback to raw text
    
    def _get_file_content_for_evaluation(self, file_path: Path) -> str:
        """
        Get file content in appropriate format for LLM evaluation
        Let LLM automatically detect and handle different formats
        """
        file_extension = file_path.suffix.lower()
        
        # Read file content
        if file_extension == '.csv':
            # For CSV, provide both raw content and structured view
            try:
                raw_content = self._read_output_file(file_path)
                structured_content = self._read_csv_file(file_path)
                return f"File type: CSV\nRaw content:\n{raw_content}\n\nStructured view:\n{structured_content}"
            except Exception as e:
                self.logger.error(f"Failed to process CSV file {file_path}: {e}")
                return f"File type: CSV\nContent:\n{self._read_output_file(file_path)}"
        else:
            # For all other files, provide raw content and let LLM detect format
            content = self._read_output_file(file_path)
            return f"File type: {file_extension.upper() if file_extension else 'UNKNOWN'}\nContent:\n{content}"
    
    def _evaluate_with_llm_universal(self, task_config: Dict[str, Any], agent_file: Path) -> Dict[str, Any]:
        """
        Universal LLM evaluation for any task without standard answer
        """
        try:
            # Get file content in appropriate format
            file_content = self._get_file_content_for_evaluation(agent_file)
            if not file_content:
                return {
                    'success': False,
                    'error': 'File read failed or is empty',
                    'score': 0.0,
                    'summary': {}
                }
            
            # Build evaluation prompt based on task configuration
            prompt = self._build_universal_evaluation_prompt(task_config, file_content, agent_file)
            
            # Call LLM for evaluation
            # Use direct model.generate call like in run_agent.py
            from smolagents.models import ChatMessage
            message = ChatMessage(role="user", content=prompt)
            response = self.model.generate(
                messages=[message]
            )
            evaluation_text = response.content
            
            # Parse evaluation result
            evaluation_result = self._parse_llm_evaluation_result(evaluation_text)
            if not evaluation_result["success"]:
                return evaluation_result
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            return {
                'success': False,
                'error': f'LLM evaluation failed: {e}',
                'score': 0.0,
                'summary': {}
            }
    
    def _evaluate_with_llm_universal_multiple_files(self, task_config: Dict[str, Any], agent_files: List[Path]) -> Dict[str, Any]:
        """
        Universal LLM evaluation for multiple files without standard answer
        """
        try:
            if not agent_files:
                return {
                    'success': False,
                    'error': 'No files provided for evaluation',
                    'score': 0.0,
                    'summary': {}
                }
            
            # Combine all file contents
            all_file_contents = []
            for i, agent_file in enumerate(agent_files, 1):
                file_content = self._get_file_content_for_evaluation(agent_file)
                if file_content:
                    all_file_contents.append(f"=== File {i}: {agent_file.name} ===\n{file_content}")
                else:
                    self.logger.warning(f"Failed to read file: {agent_file}")
            
            if not all_file_contents:
                return {
                    'success': False,
                    'error': 'All files failed to read or are empty',
                    'score': 0.0,
                    'summary': {}
                }
            
            # Combine all file contents into one evaluation content
            combined_content = "\n\n".join(all_file_contents)
            
            # Build evaluation prompt for multiple files
            prompt = self._build_universal_evaluation_prompt_multiple_files(task_config, combined_content, agent_files)
            
            # Call LLM for evaluation
            from smolagents.models import ChatMessage
            message = ChatMessage(role="user", content=prompt)
            response = self.model.generate(
                messages=[message]
            )
            evaluation_text = response.content
            
            # Parse evaluation result
            evaluation_result = self._parse_llm_evaluation_result(evaluation_text)
            if not evaluation_result["success"]:
                return evaluation_result
            
            # Add file count information to the result
            evaluation_result['file_count'] = len(agent_files)
            evaluation_result['evaluated_files'] = [str(f) for f in agent_files]
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            return {
                'success': False,
                'error': f'LLM evaluation failed: {e}',
                'score': 0.0,
                'summary': {}
            }
    
    def _build_universal_evaluation_prompt(self, task_config: Dict[str, Any], file_content: str, agent_file: Path) -> str:
        """
        Build evaluation prompt with automatic format detection
        """
        task_name = task_config.get('name', 'Unknown Task')
        task_description = task_config.get('description', '')
        file_extension = agent_file.suffix.lower()
        
        # Build scoring criteria from task config
        scoring_criteria = self._extract_scoring_criteria(task_config)
        
        prompt = f"""
You are an intelligent evaluator. Please evaluate the following output based on the task requirements and scoring criteria.

Task Name: {task_name}
Task Description: {task_description}

IMPORTANT: First, automatically identify the file format and content type, then evaluate accordingly.

Scoring Criteria:
{scoring_criteria}

Output to evaluate:
{file_content}

EVALUATION INSTRUCTIONS:
1. First, identify the file format and content type (e.g., CSV data table, JSON configuration, Markdown report, text analysis, etc.)
2. Based on the identified format, apply appropriate evaluation criteria:
   - For data files (CSV, Excel, etc.): Check data completeness, structure, quality, and relevance
   - For structured files (JSON, XML, etc.): Verify format validity, field completeness, and data types
   - For documents (Markdown, text, etc.): Assess content completeness, clarity, organization, and professionalism
   - For any other format: Evaluate based on content quality, relevance, and task requirements
3. Consider the specific task requirements when evaluating

Please evaluate the quality of this output and provide a score between 0-1 (0 = worst, 1 = best) based on the requirements and criteria. 

Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "Detailed explanation of the score",
  "score": 0.85,
  "detected_format": "CSV data table",
  "strengths": ["List of strengths"],
  "weaknesses": ["List of areas for improvement"],
  "completeness": 0.9,
  "accuracy": 0.8,
  "relevance": 0.9,
  "format_quality": 0.85
}}

Please ensure the return is valid JSON format.
"""
        return prompt
    
    def _build_universal_evaluation_prompt_multiple_files(self, task_config: Dict[str, Any], combined_content: str, agent_files: List[Path]) -> str:
        """
        Build evaluation prompt for multiple files with automatic format detection
        """
        task_name = task_config.get('name', 'Unknown Task')
        task_description = task_config.get('description', '')
        
        # Build scoring criteria from task config
        scoring_criteria = self._extract_scoring_criteria(task_config)
        
        # Get file information for context
        file_info = []
        for i, file_path in enumerate(agent_files, 1):
            file_info.append(f"File {i}: {file_path.name} ({file_path.suffix})")
        files_summary = "\n".join(file_info)
        
        prompt = f"""
You are an intelligent evaluator. Please evaluate the following multiple outputs based on the task requirements and scoring criteria.

Task Name: {task_name}
Task Description: {task_description}

Files to evaluate:
{files_summary}

IMPORTANT: Evaluate all files together as a complete deliverable. Consider how the files work together to fulfill the task requirements.

Scoring Criteria:
{scoring_criteria}

Combined outputs to evaluate:
{combined_content}

EVALUATION INSTRUCTIONS:
1. First, identify the format and content type of each file (e.g., CSV data table, JSON configuration, Markdown report, text analysis, etc.)
2. Evaluate how well the combination of files fulfills the task requirements:
   - For data files (CSV, Excel, etc.): Check data completeness, structure, quality, and relevance across all files
   - For structured files (JSON, XML, etc.): Verify format validity, field completeness, and data types
   - For documents (Markdown, text, etc.): Assess content completeness, clarity, organization, and professionalism
   - For any other format: Evaluate based on content quality, relevance, and task requirements
3. Consider how the files complement each other and work together as a complete deliverable
4. Consider the specific task requirements when evaluating the overall output

Please evaluate the quality of this combined output and provide a score between 0-1 (0 = worst, 1 = best) based on the requirements and criteria. 

Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "Detailed explanation of the score considering all files",
  "score": 0.85,
  "detected_formats": ["CSV data table", "JSON configuration"],
  "strengths": ["List of strengths across all files"],
  "weaknesses": ["List of areas for improvement"],
  "completeness": 0.9,
  "accuracy": 0.8,
  "relevance": 0.9,
  "format_quality": 0.85,
  "file_coordination": 0.9
}}

Please ensure the return is valid JSON format.
"""
        return prompt
    
    def _extract_scoring_criteria(self, task_config: Dict[str, Any]) -> str:
        """
        Extract scoring criteria from task configuration
        Enhanced to use new evaluation_criteria structure
        """
        eval_config = task_config.get('evaluation_config', {})
        evaluation_criteria = eval_config.get('criteria', {})
        
        if evaluation_criteria:
            # Use new structured evaluation criteria
            criteria_parts = []
            
            # Add description
            if 'description' in evaluation_criteria:
                criteria_parts.append(f"Task Requirements: {evaluation_criteria['description']}")
            
            # Add structure requirements for documents
            if 'structure' in evaluation_criteria:
                structure = evaluation_criteria['structure']
                if 'description' in structure:
                    criteria_parts.append(f"\nStructure Requirements: {structure['description']}")
                if 'sections' in structure:
                    sections = structure['sections']
                    criteria_parts.append("\nRequired Sections:")
                    for section in sections:
                        section_name = section.get('name', 'Unknown')
                        section_desc = section.get('description', '')
                        section_req = section.get('requirement', '')
                        criteria_parts.append(f"- {section_name}: {section_desc}")
                        if section_req:
                            criteria_parts.append(f"  Requirement: {section_req}")
            
            # Add column requirements for CSV files
            if 'columns' in evaluation_criteria:
                columns = evaluation_criteria['columns']
                criteria_parts.append("\nRequired Columns:")
                for column in columns:
                    col_name = column.get('name', 'Unknown')
                    col_index = column.get('column_index', '')
                    col_checkpoint = column.get('checkpoint', '')
                    criteria_parts.append(f"- Column {col_index}: {col_name}")
                    if col_checkpoint:
                        criteria_parts.append(f"  Checkpoint: {col_checkpoint}")
            
            # Add word count requirements
            if 'word_count' in evaluation_criteria:
                word_count = evaluation_criteria['word_count']
                if 'requirement' in word_count:
                    criteria_parts.append(f"\nWord Count: {word_count['requirement']}")
                if 'reason' in word_count:
                    criteria_parts.append(f"Reason: {word_count['reason']}")
            
            # Add tone requirements
            if 'tone' in evaluation_criteria:
                tone = evaluation_criteria['tone']
                if 'description' in tone:
                    criteria_parts.append(f"\nTone Requirements: {tone['description']}")
                if 'requirements' in tone:
                    criteria_parts.append("Tone Guidelines:")
                    for req in tone['requirements']:
                        criteria_parts.append(f"- {req}")
            
            # Add informativeness requirements
            if 'informativeness' in evaluation_criteria:
                informativeness = evaluation_criteria['informativeness']
                if 'description' in informativeness:
                    criteria_parts.append(f"\nInformativeness: {informativeness['description']}")
                if 'requirements' in informativeness:
                    criteria_parts.append("Information Requirements:")
                    for req in informativeness['requirements']:
                        criteria_parts.append(f"- {req}")
            
            # Add key aspects for general evaluation
            if 'key_aspects' in evaluation_criteria:
                key_aspects = evaluation_criteria['key_aspects']
                criteria_parts.append("\nKey Evaluation Aspects:")
                for aspect in key_aspects:
                    criteria_parts.append(f"- {aspect}")
            
            return "\n".join(criteria_parts)
        else:
            # Fallback to basic criteria based on task description
            description = task_config.get('description', '')
            
            # Basic criteria based on common evaluation needs
            criteria = f"""
1. Completeness (30%): Does the output contain all required information and elements?
2. Accuracy (25%): Is the information correct and reliable?
3. Relevance (20%): Does the output address the task requirements?
4. Structure/Format (15%): Is the output well-organized and properly formatted?
5. Quality (10%): Overall quality and professionalism of the output

Task-specific requirements: {description}
"""
            return criteria
    
    def _parse_llm_evaluation_result(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Parse LLM evaluation result with enhanced robustness
        """
        try:
            # Extract JSON part using regex
            match = re.search(r'\{[\s\S]*\}', evaluation_text)
            if match:
                json_str = match.group(0)
                result = json.loads(json_str)
                
                # Extract and validate score
                raw_score = result.get('score', 0.0)
                score = max(0.0, min(1.0, float(raw_score)))
                
                # Build summary
                summary = {
                    'score': score,
                    'reason': result.get('reason', ''),
                    'detected_format': result.get('detected_format', 'Unknown'),
                    'strengths': result.get('strengths', []),
                    'weaknesses': result.get('weaknesses', []),
                    'completeness': result.get('completeness', 0.0),
                    'accuracy': result.get('accuracy', 0.0),
                    'relevance': result.get('relevance', 0.0),
                    'format_quality': result.get('format_quality', 0.0)
                }
                
                return {
                    'success': True,
                    'score': score,
                    'details': result,
                    'summary': summary
                }
            else:
                return {
                    'success': False,
                    'error': 'LLM returned content could not be parsed as JSON',
                    'score': 0.0,
                    'details': {'raw_llm_output': evaluation_text},
                    'summary': {}
                }
        except Exception as e:
            self.logger.error(f"Failed to parse LLM evaluation result: {e}")
            return {
                'success': False,
                'error': f'Failed to parse LLM evaluation result: {e}',
                'score': 0.0,
                'details': {'raw_llm_output': evaluation_text},
                'summary': {}
            } 