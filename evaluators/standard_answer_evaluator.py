import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import logging
import os

# Add smolagents to Python path for imports
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent  # From evaluators -> evaluation_module -> browser_agent -> src
smolagents_dir = src_dir / "smolagents"
if smolagents_dir.exists() and str(smolagents_dir) not in sys.path:
    sys.path.insert(0, str(smolagents_dir))

class StandardAnswerEvaluator:
    """
    Standard answer evaluator, supports intelligent scoring for CSV and Markdown files
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # LLM initialization
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
            except Exception as e:
                self.model = None
                self.logger.error(f"Failed to create OpenAIServerModel: {e}")
        else:
            self.model = None
            self.logger.error("OPENROUTER_API_KEY not set, cannot perform LLM evaluation")
    
    def evaluate_task(self, task_config: Dict[str, Any], agent_output_dir: Path, human_output_dir: Path) -> Dict[str, Any]:
        self.logger.info(f"Starting evaluation for task: {task_config['name']}")
        
        # Support both old and new JSON structure
        eval_config = task_config.get('evaluation_config', {})
        file_matching = eval_config.get('file_matching', {})
        if file_matching:
            # New structure with auto file detection
            agent_file = self._find_file_by_pattern(agent_output_dir, file_matching.get('output_pattern', ''))
            gt_file = self._find_file_by_pattern(human_output_dir, file_matching.get('ground_truth_pattern', ''))
        else:
            # Old structure with explicit file names
            output_file = task_config.get('output_file', '')
            ground_truth = task_config.get('ground_truth', {})
            agent_file = self._find_agent_output_file(agent_output_dir, output_file)
            gt_file = human_output_dir / ground_truth.get('file_name', '') if ground_truth else None
        
        if not agent_file:
            return {
                'success': False,
                'task_name': task_config['name'],
                'output_file': str(agent_file) if agent_file else 'unknown',
                'error': f'Agent output file not found',
                'score': 0.0,
                'details': {},
                'summary': {}
            }
        
        if not gt_file or not gt_file.exists():
            return {
                'success': False,
                'task_name': task_config['name'],
                'output_file': str(agent_file),
                'error': f'Ground truth file not found: {gt_file}',
                'score': 0.0,
                'details': {},
                'summary': {}
            }
        
        agent_suffix = agent_file.suffix.lower()
        gt_suffix = gt_file.suffix.lower()
        
        # Handle CSV files
        if agent_suffix == '.csv' and gt_suffix == '.csv':
            agent_data = self._read_csv_file(agent_file)
            gt_data = self._read_csv_file(gt_file)
            if not agent_data or not gt_data:
                # If structured comparison fails, automatically switch to LLM subjective scoring
                if self.model:
                    return self._evaluate_csv_with_llm(task_config, agent_file, gt_file)
                else:
                    return {
                        'success': False,
                        'task_name': task_config['name'],
                        'output_file': str(agent_file),
                        'error': 'CSV file read failed or is empty, and LLM not available',
                        'score': 0.0,
                        'details': {},
                        'summary': {}
                    }
            # Use new scoring configuration if available
            scoring_config = eval_config.get('scoring', {})
            key_columns = scoring_config.get('key_columns', [])
            expected_columns = scoring_config.get('expected_columns', [])
            if expected_columns:
                # Use new scoring method with expected columns
                score, match, total, details = self._evaluate_csv_with_expected_columns(
                    agent_data, gt_data, key_columns, expected_columns
                )
            else:
                # Use old generic method
                score, match, total, details = self._evaluate_table_content(agent_data, gt_data)
            # If structured comparison fails (e.g. no matching columns, or match rate is very low)
            if total == 0 or score < 0.1:
                if self.model:
                    return self._evaluate_csv_with_llm(task_config, agent_file, gt_file)
                else:
                    return {
                        'success': False,
                        'task_name': task_config['name'],
                        'output_file': str(agent_file),
                        'error': 'Structured comparison failed and LLM not available',
                        'score': 0.0,
                        'details': {},
                        'summary': {}
                    }
            evaluation_result = {
                'score': score,
                'details': {
                    'evaluation_type': 'csv_cellwise_generic',
                    'matched_cells': match,
                    'total_cells': total,
                    'match_rate': score,
                    'unmatched': details[:10]
                },
                'summary': {
                    'matched_cells': match,
                    'total_cells': total,
                    'match_rate': score
                }
            }
            return {
                'success': True,
                'task_name': task_config['name'],
                'output_file': str(agent_file),
                'score': evaluation_result['score'],
                'details': evaluation_result['details'],
                'summary': evaluation_result['summary']
            }
        
        # Handle Markdown files
        elif agent_suffix == '.md' and gt_suffix == '.md':
            agent_content = self._read_text_file(agent_file)
            gt_content = self._read_text_file(gt_file)
            if not agent_content or not gt_content:
                return {
                    'success': False,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'error': 'File read failed or is empty',
                    'score': 0.0,
                    'details': {},
                    'summary': {}
                }
            if self.model:
                # Use new evaluation criteria if available
                evaluation_criteria = eval_config.get('criteria', {})
                prompt = self._build_markdown_evaluation_prompt(
                    task_config, agent_content, gt_content, evaluation_criteria
                )
                
                # Use direct model.generate call like in run_agent.py
                response = self.model.generate(
                    messages=[{"role": "user", "content": prompt}]
                )
                evaluation_text = response.content
                match = re.search(r'\{[\s\S]*\}', evaluation_text)
                if match:
                    json_str = match.group(0)
                    try:
                        result = json.loads(json_str)
                        raw_score = result.get('score', 0.0)
                        # Ensure score is within valid range
                        mapped_score = max(0.0, min(1.0, float(raw_score)))
                        return {
                            'success': True,
                            'task_name': task_config['name'],
                            'output_file': str(agent_file),
                            'score': mapped_score,
                            'details': result,
                            'summary': {'reason': result.get('reason', '')}
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'task_name': task_config['name'],
                            'output_file': str(agent_file),
                            'error': f'LLM output could not be parsed as JSON: {e}',
                            'score': 0.0,
                            'details': {'raw_llm_output': evaluation_text},
                            'summary': {}
                        }
                else:
                    return {
                        'success': False,
                        'task_name': task_config['name'],
                        'output_file': str(agent_file),
                        'error': 'LLM output could not be parsed as JSON',
                        'score': 0.0,
                        'details': {'raw_llm_output': evaluation_text},
                        'summary': {}
                    }
            else:
                return {
                    'success': False,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'error': 'LLM not available, cannot perform evaluation',
                    'score': 0.0,
                    'details': {},
                    'summary': {}
                }
        
        # Handle other file formats with LLM auto-detection
        else:
            agent_content = self._read_text_file(agent_file)
            gt_content = self._read_text_file(gt_file)
            if not agent_content or not gt_content:
                return {
                    'success': False,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'error': 'File read failed or is empty',
                    'score': 0.0,
                    'details': {},
                    'summary': {}
                }
            
            # Use LLM for other formats with auto-detection
            if self.model:
                evaluation_result = self._evaluate_with_llm_auto_detection(
                    task_config, agent_content, gt_content, agent_file, gt_file
                )
                if evaluation_result["success"]:
                    return {
                        'success': True,
                        'task_name': task_config['name'],
                        'output_file': str(agent_file),
                        'score': evaluation_result['score'],
                        'details': evaluation_result['details'],
                        'summary': evaluation_result['summary']
                    }
                else:
                    return evaluation_result
            else:
                # Fallback to text comparison if LLM not available
                text_result = self._evaluate_text_data(agent_content, gt_content)
                return {
                    'success': True,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'score': text_result['score'],
                    'details': text_result['details'],
                    'summary': {'reason': text_result['summary']} if isinstance(text_result['summary'], str) else text_result['summary']
                }
    
    def _find_agent_output_file(self, agent_output_dir: Path, output_file: str) -> Path:
        file_path = agent_output_dir / output_file
        if file_path.exists():
            return file_path
        output_file_lower = output_file.lower().replace(' ', '')
        for file_path in agent_output_dir.glob("*"):
            if file_path.is_file():
                file_name_lower = file_path.name.lower().replace(' ', '')
                if output_file_lower in file_name_lower or file_name_lower in output_file_lower:
                    return file_path
        return None
    
    def _find_file_by_pattern(self, directory: Path, pattern: str) -> Path:
        """
        Finds a file in the given directory that matches a given pattern.
        The pattern can be a simple filename or a more complex glob pattern.
        """
        if '*' in pattern or '?' in pattern:
            # Handle glob patterns
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    return file_path
        else:
            # Handle simple filename matching
            file_path = directory / pattern
            if file_path.exists():
                return file_path
        return None
    
    def _read_csv_file(self, file_path: Path) -> List[Dict[str, str]]:
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Failed to read CSV file {file_path}: {e}")
                return []
        self.logger.error(f"Failed to read CSV file {file_path}: Tried multiple encodings but failed")
        return []
    
    def _read_text_file(self, file_path: Path) -> str:
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Failed to read text file {file_path}: {e}")
                return ""
        self.logger.error(f"Failed to read text file {file_path}: Tried multiple encodings but failed")
        return ""
    
    def _evaluate_csv_data(self, agent_data: List[Dict], gt_data: List[Dict], key_columns: List[str], expected_columns: List[str]) -> Dict[str, Any]:
        """CSV file: score based on cell matching rate"""
        if not agent_data or not gt_data:
            return {'score': 0.0, 'details': {'reason': 'Data is empty'}, 'summary': {'total_cells': 0, 'matched_cells': 0, 'match_rate': 0.0}}
        
        # Check column structure
        agent_columns = set(agent_data[0].keys())
        expected_columns_set = set(expected_columns)
        
        # Only compare expected columns
        columns_to_compare = list(agent_columns & expected_columns_set)
        if not columns_to_compare:
            return {'score': 0.0, 'details': {'reason': 'No matching columns'}, 'summary': {'total_cells': 0, 'matched_cells': 0, 'match_rate': 0.0}}
        
        total_cells = 0
        matched_cells = 0
        cell_details = []
        
        if key_columns:
            # Use primary key matching
            gt_dict = {}
            for row in gt_data:
                try:
                    key = tuple(row.get(k, '') for k in key_columns)
                    gt_dict[key] = row
                except Exception:
                    continue
            
            for row in agent_data:
                try:
                    key = tuple(row.get(k, '') for k in key_columns)
                    if key in gt_dict:
                        gt_row = gt_dict[key]
                        for col in columns_to_compare:
                            total_cells += 1
                            agent_value = row.get(col, '').strip()
                            gt_value = gt_row.get(col, '').strip()
                            if agent_value == gt_value:
                                matched_cells += 1
                            else:
                                cell_details.append({
                                    'key': key,
                                    'column': col,
                                    'agent_value': agent_value,
                                    'gt_value': gt_value
                                })
                    else:
                        # Primary key not matched, all expected columns are considered unmatched
                        for col in columns_to_compare:
                            total_cells += 1
                            cell_details.append({
                                'key': key,
                                'column': col,
                                'agent_value': row.get(col, ''),
                                'gt_value': 'Primary key not matched'
                            })
                except Exception:
                    continue
        else:
            # Row-by-row matching
            min_rows = min(len(agent_data), len(gt_data))
            for i in range(min_rows):
                for col in columns_to_compare:
                    total_cells += 1
                    agent_value = agent_data[i].get(col, '').strip()
                    gt_value = gt_data[i].get(col, '').strip()
                    if agent_value == gt_value:
                        matched_cells += 1
                    else:
                        cell_details.append({
                            'row': i,
                            'column': col,
                            'agent_value': agent_value,
                            'gt_value': gt_value
                        })
        
        # Calculate match rate
        match_rate = matched_cells / total_cells if total_cells > 0 else 0.0
        
        return {
            'score': match_rate,
            'details': {
                'evaluation_type': 'csv_cell_based',
                'total_cells': total_cells,
                'matched_cells': matched_cells,
                'match_rate': match_rate,
                'unmatched_cells': cell_details[:10],  # Only show the first 10 unmatched cells
                'columns_compared': columns_to_compare
            },
            'summary': {
                'total_cells': total_cells,
                'matched_cells': matched_cells,
                'match_rate': match_rate,
                'score': match_rate
            }
        }
    
    def _evaluate_text_data(self, agent_content: str, gt_content: str) -> Dict[str, Any]:
        """
        Other files: exact match evaluation
        """
        # Remove leading/trailing spaces and newlines for comparison
        agent_clean = agent_content.strip()
        gt_clean = gt_content.strip()
        
        is_exact_match = agent_clean == gt_clean
        
        return {
            'score': 1.0 if is_exact_match else 0.0,
            'details': {
                'evaluation_type': 'text_exact_match',
                'reason': 'Exact match' if is_exact_match else 'Content not exactly the same',
                'agent_length': len(agent_clean),
                'gt_length': len(gt_clean)
            },
            'summary': {
                'exact_match': is_exact_match,
                'score': 1.0 if is_exact_match else 0.0
            }
        } 

    def _flatten_csv_content(self, data):
        flat = []
        if isinstance(data, list):
            if isinstance(data[0], dict):
                # Horizontal header
                for row in data:
                    flat.extend([str(v).strip().lower() for v in row.values()])
            elif isinstance(data[0], list):
                # Vertical header
                for row in data:
                    flat.extend([str(v).strip().lower() for v in row])
        return flat

    def _evaluate_csv_content_flat(self, agent_flat, gt_flat):
        min_len = min(len(agent_flat), len(gt_flat))
        match = sum(1 for i in range(min_len) if agent_flat[i] == gt_flat[i])
        total = max(len(agent_flat), len(gt_flat))
        return (match / total if total > 0 else 0.0, match, total) 

    def _dict_from_csv(self, data):
        # Only supports horizontal header, Name as key, Content as value
        d = {}
        for row in data:
            if 'Name' in row and 'Content' in row:
                k = str(row['Name']).strip().lower()
                v = str(row['Content']).strip().lower()
                d[k] = v
        return d

    def _evaluate_table_content(self, agent_data, gt_data):
        # Supports List[Dict] (horizontal header) and List[List] (vertical header)
        match, total = 0, 0
        details = []
        # Auto-adapt to vertical header
        if isinstance(agent_data[0], list) and isinstance(gt_data[0], list):
            agent_data = self._transpose_table(agent_data)
            gt_data = self._transpose_table(gt_data)
            # After transpose, it becomes horizontal header
            agent_data = self._listlist_to_listdict(agent_data)
            gt_data = self._listlist_to_listdict(gt_data)
        if not (isinstance(agent_data[0], dict) and isinstance(gt_data[0], dict)):
            return 0.0, 0, 0, []
        agent_cols = list(agent_data[0].keys())
        gt_cols = list(gt_data[0].keys())
        min_cols = min(len(agent_cols), len(gt_cols))
        i, j = 0, 0
        while i < len(agent_data) and j < len(gt_data):
            row_match = True
            for col in range(min_cols):
                a_val = str(agent_data[i][agent_cols[col]]).strip()
                g_val = str(gt_data[j][gt_cols[col]]).strip()
                if not self._cell_content_equal(a_val, g_val):
                    row_match = False
                    break
            if row_match:
                match += min_cols
                i += 1
                j += 1
            else:
                # Try to skip gt current row
                skip_gt = False
                if j+1 < len(gt_data):
                    skip_gt = True
                    for col in range(min_cols):
                        a_val = str(agent_data[i][agent_cols[col]]).strip()
                        g_val = str(gt_data[j+1][gt_cols[col]]).strip()
                        if not self._cell_content_equal(a_val, g_val):
                            skip_gt = False
                            break
                # Try to skip agent current row
                skip_agent = False
                if i+1 < len(agent_data):
                    skip_agent = True
                    for col in range(min_cols):
                        a_val = str(agent_data[i+1][agent_cols[col]]).strip()
                        g_val = str(gt_data[j][gt_cols[col]]).strip()
                        if not self._cell_content_equal(a_val, g_val):
                            skip_agent = False
                            break
                if skip_gt:
                    j += 1
                elif skip_agent:
                    i += 1
                else:
                    # Record all inconsistent cells
                    for col in range(min_cols):
                        a_val = str(agent_data[i][agent_cols[col]]).strip()
                        g_val = str(gt_data[j][gt_cols[col]]).strip()
                        if not self._cell_content_equal(a_val, g_val):
                            details.append({'row': i, 'col': col, 'agent_value': a_val, 'gt_value': g_val})
                    i += 1
                    j += 1
            total += min_cols
        return (match / total if total else 0.0, match, total, details)

    def _transpose_table(self, data):
        # List[List] transpose
        return [list(row) for row in zip(*data)]

    def _listlist_to_listdict(self, data):
        # List[List] to List[Dict], first row as header
        if not data or not isinstance(data[0], list):
            return data
        header = data[0]
        return [dict(zip(header, row)) for row in data[1:]]

    def _cell_content_equal(self, v1, v2):
        # Ignore case and spaces, highly tolerant of common date and number formats
        import re
        try:
            from dateutil.parser import parse as date_parse
        except ImportError:
            date_parse = None
        def normalize(s):
            s = str(s).strip().lower()
            s = s.replace('-', '').replace('/', '').replace(' ', '').replace('.', '')
            return s
        n1, n2 = normalize(v1), normalize(v2)
        if n1 == n2:
            return True
        # Try to parse as date
        if date_parse:
            try:
                d1 = date_parse(v1, fuzzy=True)
                d2 = date_parse(v2, fuzzy=True)
                if d1.date() == d2.date():
                    return True
            except Exception:
                pass
        # Further tolerant: numbers and month order difference are also considered equal
        def date_tokens(s):
            return set(re.findall(r'[0-9]{1,2}|[a-z]+', s))
        if date_tokens(n1) == date_tokens(n2) and len(date_tokens(n1)) > 1:
            return True
        # Pure number content tolerant
        if n1.isdigit() and n2.isdigit() and int(n1) == int(n2):
            return True
        return False 

    def _evaluate_with_llm_auto_detection(self, task_config: Dict[str, Any], agent_content: str, gt_content: str, agent_file: Path, gt_file: Path) -> Dict[str, Any]:
        """
        Use LLM to automatically detect file format and evaluate against standard answer
        """
        try:
            task_name = task_config.get('name', 'Unknown Task')
            task_description = task_config.get('description', '')
            agent_suffix = agent_file.suffix.lower()
            gt_suffix = gt_file.suffix.lower()
            
            prompt = f"""
You are a professional evaluator. You need to compare the AI-generated output with the standard answer (ground truth) and provide a score.

Task Name: {task_name}
Task Description: {task_description}

IMPORTANT: First, automatically identify the file format and content type of both files, then evaluate accordingly.

Standard Answer (Ground Truth):
File type: {gt_suffix.upper() if gt_suffix else 'UNKNOWN'}
Content:
{gt_content}

AI Generated Output:
File type: {agent_suffix.upper() if agent_suffix else 'UNKNOWN'}
Content:
{agent_content}

EVALUATION INSTRUCTIONS:
1. First, identify the file format and content type of both files (e.g., JSON configuration, XML data, text analysis, code, etc.)
2. Compare the AI output with the standard answer based on the identified format:
   - For structured data (JSON, XML, etc.): Compare structure, field completeness, data accuracy, and format validity
   - For documents (text, reports, etc.): Compare content completeness, accuracy, structure, and quality
   - For code files: Compare functionality, structure, and implementation correctness
   - For any other format: Compare based on content quality, accuracy, and task requirements
3. Consider the specific task requirements when evaluating
4. Provide a score between 0-1 (0 = completely wrong, 1 = perfect match)

Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "Detailed explanation of the score",
  "score": 0.85,
  "detected_agent_format": "JSON configuration",
  "detected_gt_format": "JSON configuration",
  "strengths": ["List of strengths"],
  "weaknesses": ["List of areas for improvement"],
  "completeness": 0.9,
  "accuracy": 0.8,
  "relevance": 0.9,
  "format_quality": 0.85
}}

Please ensure the return is valid JSON format.
"""
            
            # Use direct model.generate call like in run_agent.py
            try:
                # Create proper ChatMessage object for smolagents
                from smolagents.models import ChatMessage
                message = ChatMessage(role="user", content=prompt)
                response = self.model.generate(
                    messages=[message]
                )
            except Exception as e:
                raise Exception(f"LLM evaluation failed: {e}")
            evaluation_text = response.content
            
            # Parse evaluation result
            match = re.search(r'\{[\s\S]*\}', evaluation_text)
            if match:
                json_str = match.group(0)
                try:
                    result = json.loads(json_str)
                    raw_score = result.get('score', 0.0)
                    score = max(0.0, min(1.0, float(raw_score)))
                    
                    # Build summary
                    summary = {
                        'score': score,
                        'reason': result.get('reason', ''),
                        'detected_agent_format': result.get('detected_agent_format', 'Unknown'),
                        'detected_gt_format': result.get('detected_gt_format', 'Unknown'),
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
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'LLM output could not be parsed as JSON: {e}',
                        'score': 0.0,
                        'details': {'raw_llm_output': evaluation_text},
                        'summary': {}
                    }
            else:
                return {
                    'success': False,
                    'error': 'LLM output could not be parsed as JSON',
                    'score': 0.0,
                    'details': {'raw_llm_output': evaluation_text},
                    'summary': {}
                }
                
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            return {
                'success': False,
                'error': f'LLM evaluation failed: {e}',
                'score': 0.0,
                'summary': {}
            } 

    def _build_markdown_evaluation_prompt(self, task_config: Dict[str, Any], agent_content: str, gt_content: str, evaluation_criteria: Dict[str, Any]) -> str:
        """
        Builds a prompt for LLM to evaluate Markdown content based on criteria.
        """
        task_name = task_config.get('name', 'Unknown Task')
        task_description = task_config.get('description', '')
        
        prompt_template = """
You are a professional document reviewer. Based on the standard answer (reference answer) below, subjectively score the AI-generated Markdown report and provide a brief reason.
Standard answer:
{gt_content}

AI generated content:
{agent_content}

Please subjectively give a score between 0-1 (0 means worst, 1 means best) based on structure completeness, analysis depth, and recommendation feasibility, and provide a reason. Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "The content structure is complete, analysis is in-depth, and recommendations are specific.",
  "score": 0.85
}}
"""
        
        # Override criteria if provided in task_config
        if evaluation_criteria:
            if 'score_criteria' in evaluation_criteria:
                prompt_template = f"""
You are a professional document reviewer. Based on the standard answer (reference answer) below, subjectively score the AI-generated Markdown report and provide a brief reason.
Standard answer:
{gt_content}

AI generated content:
{agent_content}

Please subjectively give a score between 0-1 (0 means worst, 1 means best) based on structure completeness, analysis depth, and recommendation feasibility, and provide a reason. Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "The content structure is complete, analysis is in-depth, and recommendations are specific.",
  "score": 0.85
}}
"""
            if 'prompt_template' in evaluation_criteria:
                prompt_template = evaluation_criteria['prompt_template']
        
        return prompt_template.format(gt_content=gt_content, agent_content=agent_content) 

    def _evaluate_csv_with_expected_columns(self, agent_data: List[Dict], gt_data: List[Dict], key_columns: List[str], expected_columns: List[str]) -> Tuple[float, int, int, List]:
        """
        Evaluate CSV data using expected columns from new JSON structure
        """
        if not agent_data or not gt_data:
            return 0.0, 0, 0, []
        
        # Check column structure
        agent_columns = set(agent_data[0].keys()) if agent_data else set()
        expected_columns_set = set(expected_columns)
        
        # Only compare expected columns
        columns_to_compare = list(agent_columns & expected_columns_set)
        if not columns_to_compare:
            return 0.0, 0, 0, []
        
        total_cells = 0
        matched_cells = 0
        cell_details = []
        
        if key_columns:
            # Use primary key matching
            gt_dict = {}
            for row in gt_data:
                try:
                    key = tuple(row.get(k, '') for k in key_columns)
                    gt_dict[key] = row
                except Exception:
                    continue
            
            # Calculate total cells based on ground truth
            total_cells = len(gt_data) * len(columns_to_compare)
            
            for row in agent_data:
                try:
                    key = tuple(row.get(k, '') for k in key_columns)
                    if key in gt_dict:
                        gt_row = gt_dict[key]
                        for col in columns_to_compare:
                            agent_value = row.get(col, '').strip()
                            gt_value = gt_row.get(col, '').strip()
                            if agent_value == gt_value:
                                matched_cells += 1
                            else:
                                cell_details.append({
                                    'key': key,
                                    'column': col,
                                    'agent_value': agent_value,
                                    'gt_value': gt_value
                                })
                except Exception:
                    continue
            
            # Count missing rows in agent output
            agent_keys = set()
            for row in agent_data:
                try:
                    key = tuple(row.get(k, '') for k in key_columns)
                    agent_keys.add(key)
                except Exception:
                    continue
            
            for gt_key in gt_dict.keys():
                if gt_key not in agent_keys:
                    # This row is missing in agent output
                    for col in columns_to_compare:
                        cell_details.append({
                            'key': gt_key,
                            'column': col,
                            'agent_value': 'Missing row',
                            'gt_value': gt_dict[gt_key].get(col, '')
                        })
        else:
            # Row-by-row matching
            # Calculate total cells based on ground truth (standard answer)
            total_cells = len(gt_data) * len(columns_to_compare)
            
            min_rows = min(len(agent_data), len(gt_data))
            for i in range(min_rows):
                for col in columns_to_compare:
                    agent_value = agent_data[i].get(col, '').strip()
                    gt_value = gt_data[i].get(col, '').strip()
                    if agent_value == gt_value:
                        matched_cells += 1
                    else:
                        cell_details.append({
                            'row': i,
                            'column': col,
                            'agent_value': agent_value,
                            'gt_value': gt_value
                        })
            
            # Count missing rows in agent output
            if len(agent_data) < len(gt_data):
                for i in range(len(agent_data), len(gt_data)):
                    for col in columns_to_compare:
                        cell_details.append({
                            'row': i,
                            'column': col,
                            'agent_value': 'Missing row',
                            'gt_value': gt_data[i].get(col, '')
                        })
        
        # Calculate match rate
        match_rate = matched_cells / total_cells if total_cells > 0 else 0.0
        
        return match_rate, matched_cells, total_cells, cell_details 

    # New: When structured CSV comparison fails, use LLM subjective scoring
    def _evaluate_csv_with_llm(self, task_config: Dict[str, Any], agent_file: Path, gt_file: Path) -> Dict[str, Any]:
        """
        When structured CSV comparison fails, use LLM subjective scoring. Output 0-1 score in JSON.
        """
        agent_content = self._read_text_file(agent_file)
        gt_content = self._read_text_file(gt_file)
        if not agent_content or not gt_content:
            return {
                'success': False,
                'task_name': task_config['name'],
                'output_file': str(agent_file),
                'error': 'CSV file content read failed',
                'score': 0.0,
                'details': {},
                'summary': {}
            }
        if not self.model:
            return {
                'success': False,
                'task_name': task_config['name'],
                'output_file': str(agent_file),
                'error': 'LLM not available, cannot perform subjective scoring',
                'score': 0.0,
                'details': {},
                'summary': {}
            }
        # Build prompt, refer to Markdown scoring style
        prompt = f"""
You are a professional table evaluator. Based on the standard answer CSV and the AI output CSV below, subjectively give a score between 0-1 (0 means worst, 1 means best) and briefly explain the reason. Strictly output in the following JSON format and do not output any extra content:
{{
  "reason": "The content structure is complete, data is accurate, and the format is standard.",
  "score": 0.85
}}
Standard answer CSV:
{gt_content[:2000]}

AI output CSV:
{agent_content[:2000]}

Please strictly output JSON, do not output any extra content.
"""
        # Use direct model.generate call like in run_agent.py
        from smolagents.models import ChatMessage
        message = ChatMessage(role="user", content=prompt)
        response = self.model.generate(
            messages=[message]
        )
        import re, json
        evaluation_text = response.content
        match = re.search(r'\{[\s\S]*\}', evaluation_text)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                raw_score = result.get('score', 0.0)
                mapped_score = max(0.0, min(1.0, float(raw_score)))
                return {
                    'success': True,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'score': mapped_score,
                    'details': result,
                    'summary': {'reason': result.get('reason', '')}
                }
            except Exception as e:
                return {
                    'success': False,
                    'task_name': task_config['name'],
                    'output_file': str(agent_file),
                    'error': f'LLM output could not be parsed as JSON: {e}',
                    'score': 0.0,
                    'details': {'raw_llm_output': evaluation_text},
                    'summary': {}
                }
        else:
            return {
                'success': False,
                'task_name': task_config['name'],
                'output_file': str(agent_file),
                'error': 'LLM output could not be parsed as JSON',
                'score': 0.0,
                'details': {'raw_llm_output': evaluation_text},
                'summary': {}
            } 