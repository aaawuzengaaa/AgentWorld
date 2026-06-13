import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import uuid
from datetime import datetime

import sys

# 路径常量，与run_agent_eval.py保持一致
BASE_PATH = "/Users/baiyl/workspace/sophia/git/AgentWorld"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
backend_root = os.path.join(project_root, 'backend')
sys.path.insert(0, project_root)
sys.path.insert(0, backend_root)

class AgentBasedEvaluator:
    """
    Agent-based evaluator using BrowserAgent for evaluation tasks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.browser_agent = None
        
    def _initialize_browser_agent(self):
        """Initialize browser agent for evaluation"""
        if self.browser_agent is not None:
            return
            
        try:
            from agent.src.smolagents import OpenAIServerModel
            from agent.src.browser_agent.agents import BrowserAgent
            
            # Use same model configuration as other evaluators
            # openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_api_key = "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"
            if not openrouter_api_key:
                raise Exception("OPENROUTER_API_KEY not set")
                
            model = OpenAIServerModel(
                model_id="anthropic/claude-sonnet-4",
                api_base="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                temperature=0.3,
                max_tokens=8000,
            )
            agent_uuid = str(uuid.uuid4())
            work_root = "./data"
            work_dir = os.path.join(work_root, agent_uuid)
            self.browser_agent = BrowserAgent(
                model=model,
                headless=True,  # 无头模式
                max_steps=30,  # 自定义最大步数
                use_redis=False,
                stream_outputs=True,
                tokens_dict = {
                    "google": {
                        "access_token": "xxx",
                        "refresh_token": "xxx",
                    },
                    "airtable": {"api_key": "airtable_api_key_value"},
                },
                additional_authorized_imports=["*"],  # 允许导入所有包
                executor_kwargs={
                    "work_dir": work_dir
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser agent: {e}")
            raise
    
    async def evaluate_task(self, task_config: Dict[str, Any], agent_output_dir: Path, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate task using browser agent
        """
        # try:
        self._initialize_browser_agent()
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(task_config, agent_output_dir, evaluation_config)
        
        try:
            # Run browser agent evaluation
            result = await self.browser_agent.run_task_simple(prompt)
            
            # Parse evaluation result
            evaluation_result = self._parse_agent_result(result)
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Agent-based evaluation failed: {e}")
            return {
                'success': False,
                'error': f'Agent-based evaluation failed: {e}',
                'score': 0.0,
                'summary': {}
            }
    
    def _build_evaluation_prompt(self, task_config: Dict[str, Any], agent_output_dir: Path, evaluation_config: Dict[str, Any]) -> str:
        """Build evaluation prompt for browser agent"""
        
        # Build dynamic prompt template based on evaluation configuration
        prompt_template = self._build_dynamic_prompt_template(evaluation_config)
        
        # Extract task ID from eval_results path (e.g., eval_results/manus/task_name/ -> task_name)
        task_id = self._extract_task_id(agent_output_dir)
        self.logger.info(f"Extracted task ID from eval_results path: {task_id}")
        
        # Get task from any txt file in query directory
        task = self._get_task_from_query_file(task_id)
        
        # Fallback to config if query file not found
        if not task:
            self.logger.info(f"Query file not found for task {task_id}, falling back to config")
            task = task_config.get('query', {}).get('diligent', '')
            if task.startswith('/'):
                task = task_config.get('query', {}).get('lazy', '')
        else:
            self.logger.info(f"Successfully loaded task query from file for task {task_id}")

        # Get ground truth data if available
        ground_truth_data = self._get_ground_truth_data(evaluation_config, agent_output_dir)
        
        # Get agent output files
        agent_output_files = self._get_agent_output_files(agent_output_dir, evaluation_config)
        
        # Always use criteria from evaluation_config for pairwise comparison
        criteria_content = self._format_criteria(evaluation_config)
        
        # Format prompt with task information
        formatted_prompt = prompt_template.format(
            query_id=evaluation_config.get('query_id', ''),
            task=task,
            criteria=criteria_content,
            output=agent_output_files,
            ground_truth=ground_truth_data,
            metrics=self._format_metrics(evaluation_config, agent_output_dir),
            output_dir=agent_output_dir
        )
        
        return formatted_prompt
    
    def _build_dynamic_prompt_template(self, evaluation_config: Dict[str, Any]) -> str:
        """Build dynamic prompt template based on evaluation configuration"""
        metrics = evaluation_config.get('metrics', [])
        
        # Base prompt template
        base_prompt = """你是一个严谨、苛刻、专业的达人营销与增长领域交付物评审专家。

我将提供你以下内容：
1.	任务描述（task）

{task}

2.	评估标准（criteria）

{criteria}

3.	Agent的交付物（output）

{output}

4.	Ground Truth数据（ground_truth）

{ground_truth}

5.	评估指标（metrics）

{metrics}

请你以公正的评判者身份，采用pairwise对比方式评估Agent交付物的质量。评估流程如下：

**评估方法说明：**
1. **多文件对比**：如果存在多个GT文件或多个Agent输出文件，请分别进行对比分析
2. **字段完整性对比**：检查Agent输出是否包含所有必需的字段，与Ground Truth的字段结构进行对比
3. **数值量级对比**：对于数值型数据，允许50%的误差范围。例如：
   - 如果GT中某个数值是100，Agent输出在50-150范围内都算正确
   - 如果GT中某个数值是1000，Agent输出在500-1500范围内都算正确
   - 百分比、比率等相对数值也适用此规则
4. **指标计算**：根据提供的metrics配置，计算相应的评估指标

请按照以下流程进行评估：

1. **将Agent的交付物与Ground Truth进行比较**：
   - 如果存在多个GT文件或多个Agent输出文件，请分别进行对比分析
   - 指出Agent交付物有哪些不足，并进一步解释
   - 检查字段完整性：Agent输出是否包含所有必需的字段（不需要精确匹配column名称，允许不完全一致）
   - 检查数值量级：数值型数据是否在合理范围内（±50%误差）
   - 对于非数值型字段，检查语义匹配度
   - 对于多文件情况，分析文件间的关联性和一致性

2. **从不同维度对Agent交付物进行评价**：
   - 基于评估标准中定义的每一条criterion，进行逐条判断
   - 对每个criterion进行pairwise对比：与GT对比，表现更好就加分，表现不好就减分
   - 在每个维度的评价之后，给每一个维度一个1～10的分数
   - 必须为每一项优点与缺点提供具体示例，严禁泛泛而谈
   - 示例应明确引用交付物中的字段、行、截图片段或关键内容进行说明

3. **计算评估指标**：
   - 根据提供的metrics配置，计算相应的评估指标
   - 提供详细的指标计算过程和结果

4. **综合评分**：
   - 综合每个维度的评估，对Agent的交付物给出一个1～10的综合分数
   - 打分需要尽可能严格，遵守评分规则

**评分规则：**
- 当交付物存在与任务不相关，或者有本质性的事实错误时，总分必须是1到2分
- 当交付物没有严重错误，但是质量较低，没有满足任务需求，总分为3到4分
- 当交付物基本满足任务要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分
- 当交付物质量与Ground Truth相近，在所有维度上表现良好，总分得7到8分
- 只有当交付物质量显著超过Ground Truth，充分地解决了任务问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分

**Pairwise对比要求：**
- 对于每个字段，明确对比GT和Agent的输出
- 对于数值型字段，计算相对误差：|Agent值 - GT值| / GT值 × 100%
- 如果相对误差 ≤ 50%，则认为该数值正确
- 对于非数值型字段，检查语义匹配度
- 对于多文件情况，分别进行文件级别的对比，然后进行整体分析
- 提供详细的对比表格，显示每个字段的GT值、Agent值、是否匹配、误差百分比等
- 如果存在多个GT文件，请说明每个GT文件的用途和重要性

5. **评分输出**（必须输出为JSON格式）：
   - 完成度（Completeness）：若交付物完全覆盖所有任务要求，则为 1 分；否则按交付情况严格计算覆盖率
   - 质量评分：每个criterion请单独给出得分（1-10分），基于pairwise对比结果，表现更好就加分，表现不好就减分
   - 评估指标：根据metrics配置计算相应的指标值
   
   评分输出格式如下（务必按JSON输出）：
   {{
       "Completeness": 完成度分数（满分1分）,
       "criteria_scores": {{
           "criterion1 name": 分数（1-10分）,
           "criterion2 name": 分数（1-10分）,
           "criterion3 name": 分数（1-10分）
       }},
       "metrics_scores": {{
           // 根据实际metrics配置动态生成，具体格式见下方metrics部分
       }},
       "overall": 综合总评分（1-10分）
   }}

特别要求：
- 评分必须严格，形式完成 ≠ 实质完成（图像或展示类交付物：只要能体现所要求功能或效果，即可认定为有效成果，不要求展示代码或运行日志）
- 不评估是否调用了某个工具或平台（如是否使用Instagram API）
- 不允许因"可能是测试"、"还没完善"而放宽标准
- 所有结论都必须有明确理由支持，不能主观含糊
- 对于存在明显虚构、重复或批量模板生成痕迹的数据或内容，必须明确指出
- **数值对比采用±50%误差范围，而不是严格匹配**

最终你需要保存三个文件：
1. 报告文件：f"{output_dir}/{query_id}/eval_report_{query_id}.md"
    报告格式为：

    一、优点
    ...(具体优点，要showcase具体内容)
    二、缺点
    ...(具体缺点，要showcase具体内容)
    三、暴露的Agent问题
    ...(具体问题，要showcase具体内容)
    四、完成度分析
    ...(具体完成度分析)
    五、质量评分分析
    ...(具体质量分析)"""

        # 动态添加评估指标分析部分
        if metrics:
            base_prompt += "\n    六、评估指标分析\n    ...(详细的指标计算过程和结果，包括具体数值和计算过程)"
            
            # 为每个指标添加具体的分析要求
            for i, metric in enumerate(metrics, 1):
                metric_name = metric.get('name', '')
                if metric_name == 'recall':
                    base_prompt += f"\n    六-{i}、召回率（Recall）分析\n    ...(召回率计算过程和结果)"
                elif metric_name == 'precision':
                    base_prompt += f"\n    六-{i}、精确率（Precision）分析\n    ...(精确率计算过程和结果)"
                elif metric_name == 'f1_score':
                    base_prompt += f"\n    六-{i}、F1分数分析\n    ...(F1分数计算过程和结果)"
                elif metric_name == 'pairwise_comparison':
                    base_prompt += f"\n    六-{i}、逐对对比（Pairwise Comparison）分析\n    ...(逐对对比计算过程和结果)"
                elif metric_name == 'completeness':
                    base_prompt += f"\n    六-{i}、完整性（Completeness）分析\n    ...(完整性计算过程和结果)"
                elif metric_name == 'accuracy':
                    base_prompt += f"\n    六-{i}、准确性（Accuracy）分析\n    ...(准确性计算过程和结果)"
                elif metric_name == 'quality':
                    base_prompt += f"\n    六-{i}、质量（Quality）分析\n    ...(质量评估过程和结果)"
                else:
                    base_prompt += f"\n    六-{i}、{metric_name}分析\n    ...({metric_name}计算过程和结果)"
        
        # 添加Pairwise对比详情（如果存在相关指标）
        has_pairwise = any(metric.get('name') in ['pairwise_comparison', 'field_comparison', 'numerical_accuracy'] for metric in metrics)
        if has_pairwise:
            base_prompt += "\n    七、Pairwise对比详情\n    ...(详细的字段对比表格，包含GT值、Agent值、误差百分比等)"
        
        base_prompt += """

2. 完成度评分文件：f"{output_dir}/{query_id}/completeness_{query_id}.json"
    评分文件格式为：    
    {{
        "Completeness": 完成度分数（满分1分）,
    }}

3. 综合评分文件：f"{output_dir}/{query_id}/evaluation_{query_id}.json"
    评分文件格式为：
    {{
        "criteria_scores": {{
            "criterion1 name": 分数（1-10分）,
            "criterion2 name": 分数（1-10分）,
            "criterion3 name": 分数（1-10分）
        }},
        "metrics_scores": {{
            // 根据实际metrics配置动态生成
        }},
        "overall": 综合总评分（1-10分）
    }}
"""

        return base_prompt
    
    # def _format_criteria(self, criteria: Dict[str, Any]) -> str:
    #     """Format evaluation criteria for prompt"""
    #     if not criteria:
    #         return "No specific criteria provided"
            
    #     formatted = []
    #     for key, criterion in criteria.items():
    #         if isinstance(criterion, dict):
    #             name = criterion.get('name', key)
    #             description = criterion.get('description', '')
    #             points = criterion.get('points', 1)
    #             formatted.append(f"- {name} ({points} points): {description}")
    #         else:
    #             formatted.append(f"- {key}: {criterion}")
        
    #     return "\n".join(formatted)

    def _format_criteria(self, evaluation_config: Dict[str, Any]) -> str:
        """Format evaluation criteria for prompt"""
        criteria = evaluation_config.get('criteria', {})
        if criteria:
            criteria_prompt = "**请严格遵守以下评估标准，用于pairwise对比评分**：\n"
            criteria_prompt += "**评分规则：根据每个criterion的要求进行对比，表现更好就加分，表现不好就减分**\n\n"
            for criterion in criteria:
                name = criterion.get('name', '')
                description = criterion.get('description', '')
                points = criterion.get('points', 1)
                criteria_prompt += f"- {name} ({points} points): {description}\n"
            return criteria_prompt
        else:
            return ""
            
    
    def _get_ground_truth_data(self, evaluation_config: Dict[str, Any], agent_output_dir: Path) -> str:
        """Get and format ground truth data for comparison"""
        # Check if there are metrics with GT file paths
        metrics = evaluation_config.get('metrics', [])
        task_id = None
        gt_columns = []
        
        # Extract task ID from agent_output_dir path
        # Expected path format: eval_results/{model_name}/{task_name}/
        task_id = self._extract_task_id(agent_output_dir)
        
        # Collect all GT files from metrics
        gt_files = []
        for metric in metrics:
            gt_config = metric.get('gt', {})
            gt_filename = gt_config.get('file', '')
            if gt_filename:
                # Construct GT file path based on task ID
                gt_file_path = Path(f"{BASE_PATH}/Ground Truth/{task_id}/{gt_filename}")
                if gt_file_path.exists():
                    gt_files.append({
                        'path': gt_file_path,
                        'columns': gt_config.get('column', []),
                        'metric_name': metric.get('name', 'unknown')
                    })
                    self.logger.info(f"Found GT file: {gt_file_path} for metric: {metric.get('name', 'unknown')}")
                else:
                    self.logger.warning(f"GT file not found: {gt_file_path}")
                
                # Collect GT columns from all metrics
                metric_columns = gt_config.get('column', [])
                if metric_columns:
                    gt_columns.extend(metric_columns)
        
        # Fallback to file_matching if no metrics GT path
        if not gt_files:
            file_matching = evaluation_config.get('file_matching', {})
            ground_truth_pattern = file_matching.get('ground_truth_pattern', '')
            
            if not ground_truth_pattern or ground_truth_pattern == 'none':
                return "无Ground Truth数据提供"
            
            # Try to find ground truth files
            gt_files = self._find_gt_files_by_pattern(agent_output_dir, ground_truth_pattern)
        
        if not gt_files:
            return f"Ground Truth文件未找到"
        
        # Read and format all GT files
        gt_data_sections = []
        for i, gt_file_info in enumerate(gt_files):
            gt_file_path = gt_file_info['path']
            file_columns = gt_file_info.get('columns', gt_columns)
            metric_name = gt_file_info.get('metric_name', f'GT文件{i+1}')
            
            try:
                # Read ground truth file content based on file type
                file_extension = gt_file_path.suffix.lower()
                self.logger.info(f"Reading GT file {i+1}: {gt_file_path} (extension: {file_extension})")
                
                file_content = self._read_file_by_type(gt_file_path, file_columns)
                gt_data_sections.append(f"**{metric_name}**：\n{file_content}")
                
            except Exception as e:
                self.logger.error(f"Failed to read GT file {gt_file_path}: {e}")
                gt_data_sections.append(f"**{metric_name}**：读取失败 - {str(e)}")
        
        # Combine all GT data
        if len(gt_data_sections) == 1:
            return gt_data_sections[0]
        else:
            combined_gt = "**多个Ground Truth文件**：\n\n" + "\n\n".join(gt_data_sections)
            return combined_gt
    
    def _read_file_by_type(self, file_path: Path, gt_columns: list) -> str:
        """Read file based on its type"""
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.xlsx', '.xls']:
            self.logger.info("Processing Excel file")
            return self._read_excel_file(file_path, gt_columns)
        elif file_extension == '.csv':
            self.logger.info("Processing CSV file")
            return self._read_csv_file(file_path, gt_columns)
        elif file_extension == '.json':
            self.logger.info("Processing JSON file")
            return self._read_json_file(file_path, gt_columns)
        elif file_extension in ['.txt', '.md']:
            self.logger.info("Processing text file")
            return self._read_text_file(file_path)
        elif file_extension in ['.tsv', '.tab']:
            self.logger.info("Processing TSV file")
            return self._read_tsv_file(file_path, gt_columns)
        elif file_extension in ['.parquet', '.pq']:
            self.logger.info("Processing Parquet file")
            return self._read_parquet_file(file_path, gt_columns)
        elif file_extension in ['.feather', '.ftr']:
            self.logger.info("Processing Feather file")
            return self._read_feather_file(file_path, gt_columns)
        elif file_extension in ['.pickle', '.pkl']:
            self.logger.info("Processing Pickle file")
            return self._read_pickle_file(file_path, gt_columns)
        else:
            self.logger.warning(f"Unknown file extension: {file_extension}, trying to read as text")
            return self._read_text_file(file_path)
    
    def _find_gt_files_by_pattern(self, directory: Path, pattern: str) -> list:
        """Find multiple GT files by pattern"""
        gt_files = []
        
        if not directory.exists():
            return gt_files
            
        if '*' in pattern or '?' in pattern:
            # Handle glob patterns
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    gt_files.append({
                        'path': file_path,
                        'columns': [],
                        'metric_name': f'GT文件_{file_path.name}'
                    })
        else:
            # Handle simple filename matching
            file_path = directory / pattern
            if file_path.exists() and file_path.is_file():
                gt_files.append({
                    'path': file_path,
                    'columns': [],
                    'metric_name': f'GT文件_{file_path.name}'
                })
            else:
                # Fallback to fuzzy matching
                pattern_lower = pattern.lower().replace(' ', '')
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        file_name_lower = file_path.name.lower().replace(' ', '')
                        if pattern_lower in file_name_lower or file_name_lower in pattern_lower:
                            gt_files.append({
                                'path': file_path,
                                'columns': [],
                                'metric_name': f'GT文件_{file_path.name}'
                            })
        
        self.logger.info(f"Found {len(gt_files)} GT files by pattern: {pattern}")
        return gt_files
    
    def _get_agent_output_files(self, agent_output_dir: Path, evaluation_config: Dict[str, Any]) -> str:
        """Get and format agent output files for comparison"""
        try:
            # Get output files from evaluation_config (set by run_agent_eval.py)
            output_files = evaluation_config.get('output_files', [])
            
            if not output_files:
                return "Agent输出文件未指定"
            
            # Format output files information
            output_sections = []
            for i, file_path_str in enumerate(output_files):
                file_path = Path(file_path_str)
                file_name = file_path.name
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_extension = file_path.suffix.lower()
                
                try:
                    # Read file content based on type
                    if file_extension in ['.xlsx', '.xls', '.csv', '.json', '.tsv', '.tab', '.parquet', '.pq', '.feather', '.ftr', '.pickle', '.pkl']:
                        # For data files, read and format content
                        file_content = self._read_file_by_type(file_path, [])
                        output_sections.append(f"**输出文件 {i+1}: {file_name}** (大小: {file_size} bytes)\n{file_content}")
                    elif file_extension in ['.txt', '.md']:
                        # For text files, read content directly
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        output_sections.append(f"**输出文件 {i+1}: {file_name}** (大小: {file_size} bytes)\n{content}")
                    else:
                        # For other file types, just show file info
                        output_sections.append(f"**输出文件 {i+1}: {file_name}** (大小: {file_size} bytes, 类型: {file_extension})\n*二进制文件，内容不显示*")
                
                except Exception as e:
                    self.logger.error(f"Failed to read output file {file_path}: {e}")
                    output_sections.append(f"**输出文件 {i+1}: {file_name}** (大小: {file_size} bytes)\n*读取失败: {str(e)}*")
            
            # Combine all output files
            if len(output_sections) == 1:
                return output_sections[0]
            else:
                combined_output = "**多个Agent输出文件**：\n\n" + "\n\n".join(output_sections)
                return combined_output
                
        except Exception as e:
            self.logger.error(f"Failed to get agent output files: {e}")
            return f"获取Agent输出文件失败：{str(e)}"
    
    def _read_excel_file(self, file_path: Path, gt_columns: list) -> str:
        """Read Excel file (.xlsx, .xls)"""
        try:
            import pandas as pd
            
            df = pd.read_excel(file_path)
            self.logger.info(f"Excel file loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Filter columns if specified
            if gt_columns:
                available_columns = df.columns.tolist()
                # Find matching columns (case-insensitive)
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False)
                    self.logger.info(f"Filtered to {len(matching_columns)} columns: {matching_columns}")
                    return f"Ground Truth Excel数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    self.logger.warning(f"Specified columns not found: {gt_columns}")
                    return f"Ground Truth Excel数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False)
                return f"Ground Truth Excel数据（转换为CSV格式）：\n{gt_content}"
        except Exception as e:
            self.logger.error(f"Failed to read Excel file {file_path}: {e}")
            return f"读取Excel文件失败：{str(e)}"
    
    def _read_csv_file(self, file_path: Path, gt_columns: list) -> str:
        """Read CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            if gt_columns:
                available_columns = df.columns.tolist()
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth CSV数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth CSV数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False)
                return f"Ground Truth CSV数据：\n{gt_content}"
        except Exception as e:
            return f"读取CSV文件失败：{str(e)}"
    
    def _read_json_file(self, file_path: Path, gt_columns: list) -> str:
        """Read JSON file"""
        try:
            import pandas as pd
            import json
            
            # Try to read as JSON first
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                

                
                if gt_columns:
                    available_columns = df.columns.tolist()
                    matching_columns = []
                    for target_col in gt_columns:
                        for col in available_columns:
                            if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                                matching_columns.append(col)
                                break
                    
                    if matching_columns:
                        df = df[matching_columns]
                        gt_content = df.to_csv(index=False)
                        return f"Ground Truth JSON数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                    else:
                        gt_content = df.to_csv(index=False)
                        return f"Ground Truth JSON数据（未找到指定列，显示所有列）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth JSON数据（转换为CSV格式）：\n{gt_content}"
            else:
                # Return raw JSON if not a list of dictionaries
                return f"Ground Truth JSON数据：\n{json.dumps(data, indent=2, ensure_ascii=False)}"
        except Exception as e:
            return f"读取JSON文件失败：{str(e)}"
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read text file (.txt, .md)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"Ground Truth文本数据：\n{content}"
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return f"Ground Truth文本数据（GBK编码）：\n{content}"
            except Exception as e:
                return f"读取文本文件失败：{str(e)}"
        except Exception as e:
            return f"读取文本文件失败：{str(e)}"
    
    def _read_tsv_file(self, file_path: Path, gt_columns: list) -> str:
        """Read TSV file (.tsv, .tab)"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, sep='\t')
            
            if gt_columns:
                available_columns = df.columns.tolist()
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False, sep='\t')
                    return f"Ground Truth TSV数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False, sep='\t')
                    return f"Ground Truth TSV数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False, sep='\t')
                return f"Ground Truth TSV数据：\n{gt_content}"
        except Exception as e:
            return f"读取TSV文件失败：{str(e)}"
    
    def _read_parquet_file(self, file_path: Path, gt_columns: list) -> str:
        """Read Parquet file (.parquet, .pq)"""
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            
            if gt_columns:
                available_columns = df.columns.tolist()
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Parquet数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Parquet数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False)
                return f"Ground Truth Parquet数据（转换为CSV格式）：\n{gt_content}"
        except Exception as e:
            return f"读取Parquet文件失败：{str(e)}"
    
    def _read_feather_file(self, file_path: Path, gt_columns: list) -> str:
        """Read Feather file (.feather, .ftr)"""
        try:
            import pandas as pd
            df = pd.read_feather(file_path)
            
            if gt_columns:
                available_columns = df.columns.tolist()
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Feather数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Feather数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False)
                return f"Ground Truth Feather数据（转换为CSV格式）：\n{gt_content}"
        except Exception as e:
            return f"读取Feather文件失败：{str(e)}"
    
    def _read_pickle_file(self, file_path: Path, gt_columns: list) -> str:
        """Read Pickle file (.pickle, .pkl)"""
        try:
            import pandas as pd
            import pickle
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert to DataFrame if it's a DataFrame or list of dictionaries
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:
                return f"Ground Truth Pickle数据（非表格格式）：\n{str(data)}"
            
            if gt_columns:
                available_columns = df.columns.tolist()
                matching_columns = []
                for target_col in gt_columns:
                    for col in available_columns:
                        if target_col.lower() in col.lower() or col.lower() in target_col.lower():
                            matching_columns.append(col)
                            break
                
                if matching_columns:
                    df = df[matching_columns]
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Pickle数据（重点关注列：{', '.join(matching_columns)}）：\n{gt_content}"
                else:
                    gt_content = df.to_csv(index=False)
                    return f"Ground Truth Pickle数据（未找到指定列，显示所有列）：\n{gt_content}"
            else:
                gt_content = df.to_csv(index=False)
                return f"Ground Truth Pickle数据（转换为CSV格式）：\n{gt_content}"
        except Exception as e:
            return f"读取Pickle文件失败：{str(e)}"
    
    def _find_file_by_pattern(self, directory: Path, pattern: str) -> Optional[Path]:
        """Find file by pattern in directory"""
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

    def _extract_task_id(self, agent_output_dir: Path) -> str:
        """Extract task ID from agent output directory path
        
        Expected path format: eval_results/{model_name}/{task_name}/
        Task ID is the task_name from the path
        """
        if not agent_output_dir:
            return "1"  # Default fallback
        
        path_parts = agent_output_dir.parts
        
        # The task_id is the last directory name in the path
        # Expected: eval_results/model_name/task_name/
        if len(path_parts) >= 2:
            task_id = path_parts[-1]  # Get the last part as task_id
            if task_id and task_id != ".":
                return task_id
        
        # Fallback: try to extract from path string
        path_str = str(agent_output_dir)
        import re
        
        # Look for patterns like "task_name" in the path
        # This handles cases where the path might be nested differently
        task_patterns = [
            r'/([^/]+)/$',  # Last directory name
            r'\\\\([^\\]+)\\\\$',  # Windows path last directory
        ]
        
        for pattern in task_patterns:
            match = re.search(pattern, path_str)
            if match:
                return match.group(1)
        
        # If still no task ID found, return default
        return "1"  # Default fallback

    def _get_task_from_query_file(self, task_id: str) -> str:
        """Get task query from any txt file in the query directory for the given task ID"""
        try:
            # Look for any txt file in Query/{task_id}/ directory
            query_dir = Path(f"{BASE_PATH}/Query/{task_id}/")
            
            if not query_dir.exists():
                self.logger.warning(f"Query directory not found: {query_dir}")
                return ""
            
            # Find any txt file in the directory
            txt_files = list(query_dir.glob("*.txt"))
            
            if not txt_files:
                self.logger.warning(f"No txt files found in query directory: {query_dir}")
                return ""
            
            # Use the first txt file found
            query_file_path = txt_files[0]
            self.logger.info(f"Found query file: {query_file_path}")
            
            # If multiple txt files exist, log a warning
            if len(txt_files) > 1:
                self.logger.warning(f"Multiple txt files found in {query_dir}: {[f.name for f in txt_files]}")
                self.logger.info(f"Using first file: {query_file_path.name}")
            
            # Read query content from file
            with open(query_file_path, 'r', encoding='utf-8') as f:
                query_content = f.read().strip()
            
            if not query_content:
                self.logger.warning(f"Query file is empty: {query_file_path}")
                return ""
            
            self.logger.info(f"Successfully loaded query from: {query_file_path}")
            
            # Validate query content
            if self._validate_query_content(query_content):
                return query_content
            else:
                self.logger.warning(f"Query content validation failed for task {task_id}")
                return query_content  # Still return content even if validation fails
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error reading query file for task {task_id}: {e}")
            # Try with different encoding
            try:
                with open(query_file_path, 'r', encoding='gbk') as f:
                    query_content = f.read().strip()
                self.logger.info(f"Successfully loaded query with GBK encoding from: {query_file_path}")
                return query_content
            except Exception as e2:
                self.logger.error(f"Failed to read query file with GBK encoding: {e2}")
                return ""
        except Exception as e:
            self.logger.error(f"Failed to read query file for task {task_id}: {e}")
            return ""

    def _validate_query_content(self, query_content: str) -> bool:
        """Validate query content structure"""
        if not query_content:
            return False
        
        # Check if content has minimum required structure
        lines = query_content.split('\n')
        if len(lines) < 3:
            return False
        
        # Check for common query patterns
        has_task_description = any('collect' in line.lower() or 'extract' in line.lower() or 'find' in line.lower() for line in lines[:5])
        has_data_fields = any('field' in line.lower() or 'data' in line.lower() for line in lines)
        
        if not has_task_description:
            self.logger.warning("Query content lacks task description")
            return False
        
        if not has_data_fields:
            self.logger.warning("Query content lacks data field specifications")
            return False
        
        return True

    def _get_evaluation_instruction(self, metric_name: str, gt_file: str, gt_columns: list) -> str:
        """Get specific evaluation instruction based on metric name"""
        # 动态生成字段名称
        if gt_columns:
            field_names = ", ".join(gt_columns)
            field_description = f"{field_names}字段"
        else:
            field_names = "所有字段"
            field_description = "所有字段"
        
        instructions = {
            'recall': f"- 召回率（Recall）：计算Agent输出中正确匹配的{field_description}数量与Ground Truth中总{field_description}数量的比率\n  计算公式：Recall = 正确匹配数 / GT总数",
            'precision': f"- 精确率（Precision）：计算Agent输出中正确匹配的{field_description}数量与Agent输出中总{field_description}数量的比率\n  计算公式：Precision = 正确匹配数 / Agent输出总数",
            'f1_score': f"- F1分数：精确率和召回率的调和平均数\n  计算公式：F1 = 2 × (Precision × Recall) / (Precision + Recall)",
            'pairwise_comparison': f"- 逐对对比（Pairwise Comparison）：将Agent输出与Ground Truth进行逐行逐字段对比，计算匹配率和误差分析" + 
                                 (f"\n  重点关注列：{', '.join(gt_columns)}" if gt_columns else "\n  进行全字段对比"),
            'completeness': f"- 完整性评估（Completeness）：检查Agent输出是否包含所有必需的字段和数据，计算字段覆盖率" +
                           (f"\n  必需字段：{', '.join(gt_columns)}" if gt_columns else "\n  检查所有字段的完整性"),
            'accuracy': f"- 准确性评估（Accuracy）：评估Agent输出数据的准确性和可靠性，计算数据准确率" +
                        (f"\n  关键字段：{', '.join(gt_columns)}" if gt_columns else "\n  评估所有字段的准确性"),
            'quality': f"- 质量评估（Quality）：综合评估Agent输出的整体质量，包括数据完整性、准确性、一致性等维度",
            'field_comparison': f"- 字段对比（Field Comparison）：对比Agent输出与GT的字段完整性和准确性，计算字段匹配率" +
                               (f"\n  关键字段：{', '.join(gt_columns)}" if gt_columns else "\n  进行全字段对比"),
            'numerical_accuracy': f"- 数值准确性（Numerical Accuracy）：评估数值型数据的准确性，允许±50%误差范围，计算数值准确率",
            'semantic_similarity': f"- 语义相似性（Semantic Similarity）：评估文本内容的语义匹配度，计算语义相似度分数"
        }
        
        return instructions.get(metric_name, f"- {metric_name}：根据Ground Truth文件 {gt_file} 的列 {gt_columns} 进行评估")

    def _format_metrics(self, evaluation_config: Dict[str, Any], agent_output_dir: Path = None) -> str:
        """Format metrics for prompt"""
        metrics = evaluation_config.get('metrics', [])
        if not metrics:
            return "无特定评估指标要求"
        
        # Extract task ID for dynamic path construction
        task_id = self._extract_task_id(agent_output_dir) if agent_output_dir else "1"
        
        metrics_prompt = "**评估指标要求：**\n"
        for metric in metrics:
            metric_name = metric.get('name', '')
            gt_config = metric.get('gt', {})
            gt_file = gt_config.get('file', '')
            gt_columns = gt_config.get('column', [])
            
            # 使用专门的指令生成方法
            instruction = self._get_evaluation_instruction(metric_name, gt_file, gt_columns)
            metrics_prompt += instruction + "\n"
        
        metrics_prompt += "\n**计算说明：**\n"
        
        # 动态生成匹配字段说明
        all_columns = set()
        for metric in metrics:
            gt_config = metric.get('gt', {})
            columns = gt_config.get('column', [])
            all_columns.update(columns)
        
        if all_columns:
            columns_str = ", ".join(all_columns)
            metrics_prompt += f"- 使用以下字段进行匹配：{columns_str}，允许不完全精确匹配（如大小写差异、空格差异等）\n"
        else:
            metrics_prompt += "- 使用GT文件中的所有字段进行匹配，允许不完全精确匹配（如大小写差异、空格差异等）\n"
        
        # 明确说明不使用evaluation_config中的columns字段
        metrics_prompt += "- **注意：请根据GT文件的实际column结构进行对比，不需要精确匹配column名称**\n"
        metrics_prompt += "- **允许column名称的语义匹配，如'Creator social id'与'username'、'social_id'等**\n"
        
        metrics_prompt += f"- Ground Truth文件路径：{BASE_PATH}/Ground Truth/{task_id}/{gt_file}\n"
        if any(metric.get('gt', {}).get('column') for metric in metrics):
            metrics_prompt += "- 重点关注GT文件中指定的关键列进行对比\n"
        metrics_prompt += "- 请提供详细的计算过程和结果\n"
        metrics_prompt += "- 对于数值型数据，允许±50%误差范围\n"
        metrics_prompt += "- 对于文本型数据，允许语义匹配，不要求完全一致\n"
        metrics_prompt += "- 所有计算结果必须保留小数点后2位\n"
        
        # 动态生成metrics_scores格式
        metrics_scores_format = []
        for metric in metrics:
            metric_name = metric.get('name', '')
            if metric_name == 'recall':
                metrics_scores_format.append('"recall": 召回率分数')
            elif metric_name == 'precision':
                metrics_scores_format.append('"precision": 精确率分数')
            elif metric_name == 'f1_score':
                metrics_scores_format.append('"f1_score": F1分数')
            elif metric_name == 'pairwise_comparison':
                metrics_scores_format.append('"pairwise_comparison": 对比分数')
            elif metric_name == 'completeness':
                metrics_scores_format.append('"completeness": 完整性分数')
            elif metric_name == 'accuracy':
                metrics_scores_format.append('"accuracy": 准确性分数')
            elif metric_name == 'quality':
                metrics_scores_format.append('"quality": 质量分数')
            else:
                metrics_scores_format.append(f'"{metric_name}": {metric_name}分数')
        
        metrics_prompt += f"\n**metrics_scores格式要求：**\n"
        metrics_prompt += "{\n"
        for format_item in metrics_scores_format:
            metrics_prompt += f"    {format_item},\n"
        metrics_prompt += "}\n"
        
        return metrics_prompt

    def _format_columns(self, evaluation_config: Dict[str, Any]) -> str:
        """Format columns for prompt"""
        columns = evaluation_config.get('columns', {})
        columns_list = [col["name"] for col in columns]
        if columns:
            columns_prompt = "输出的表格必须包含以下列，但名称不一定完全一致：\n"
            return columns_prompt + ", ".join(columns_list) + "\n"
        else:
            return ""

    def _parse_agent_result(self, result: str) -> Dict[str, Any]:
        """Parse browser agent evaluation result"""
        try:
            # Try to extract JSON from result
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    evaluation_data = json.loads(json_match.group())
                    
                    return {
                        'success': True,
                        'score': float(evaluation_data.get('overall', evaluation_data.get('score', 0.0))),
                        'summary': {
                            'completeness': evaluation_data.get('Completeness', 0.0),
                            'criteria_scores': evaluation_data.get('criteria_scores', {}),
                            'metrics_scores': evaluation_data.get('metrics_scores', {}),
                            'overall': evaluation_data.get('overall', 0.0),
                            'raw_result': result
                        }
                    }
                except json.JSONDecodeError as json_e:
                    self.logger.warning(f"JSON parsing failed, trying text extraction: {json_e}")
                    # Continue to text extraction
            else:
                self.logger.warning("No JSON found in result, trying text extraction")
            
            # Fallback: try to extract score from text
            score_match = re.search(r'score[:\s]+([0-9.]+)', result, re.IGNORECASE)
            overall_match = re.search(r'overall[:\s]+([0-9.]+)', result, re.IGNORECASE)
            
            score = 0.0
            if overall_match:
                score = float(overall_match.group(1))
            elif score_match:
                score = float(score_match.group(1))
            
            return {
                'success': True,
                'score': score,
                'summary': {
                    'reasoning': result,
                    'raw_result': result
                }
            }
                
        except Exception as e:
            self.logger.error(f"Failed to parse agent result: {e}")
            return {
                'success': False,
                'error': f'Failed to parse evaluation result: {e}',
                'score': 0.0,
                'summary': {'raw_result': result}
            }