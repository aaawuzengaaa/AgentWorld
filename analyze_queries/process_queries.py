import pandas as pd
import requests
import json
import time
import os
from typing import Dict, Any, Tuple
import re

class QueryProcessor:
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AgentWorld", 
            "X-Title": "AgentWorld Query Analysis"
        }
    
    def create_prompt(self, query: str) -> str:
        """Create a comprehensive prompt for translation, difficulty, and executability assessment"""
        prompt = f"""
请分析以下查询并提供综合评估，返回JSON格式：

查询内容："{query}"

请按照以下JSON格式回复：
{{
    "chinese_translation": "您的中文翻译",
    "difficulty_level": 1-5,
    "difficulty_reasoning": "简要说明为什么给出这个难度等级",
    "is_executable": true/false,
    "executability_reason": "说明为什么可执行或不可执行，或执行所缺少的信息"
}}

评估指南：
1. **中文翻译**：提供准确自然的中文翻译，保持原意和意图。

2. **难度等级 (1-5)**：基于大语言模型能力评估难度：
   - 1: 非常简单 - 简单的信息检索或基础文本处理
   - 2: 简单 - 需要基础推理的直接任务
   - 3: 中等 - 需要适度推理、规划或多步骤过程的任务
   - 4: 困难 - 需要高级推理、领域专业知识或复杂规划的任务
   - 5: 非常困难 - 极其复杂的任务，挑战当前大语言模型能力极限

3. **可执行性评估**：判断任务描述是否足够清晰，人类是否能够理解并执行：
   - 任务目标是否明确具体
   - 是否提供了必要的背景信息和约束条件
   - 输入输出要求是否清楚
   - 执行步骤或期望结果是否明确
   - 如果让人来执行这个任务，是否有足够的信息来完成

请仅提供有效的JSON格式回复，不要包含任何额外的文字或markdown格式。
"""
        return prompt
    
    def call_openrouter_api(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call OpenRouter API with retry logic"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"API Error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        
            except Exception as e:
                print(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, int, str, bool, str]:
        """Parse the API response and extract structured data"""
        try:
            content = response['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                chinese_translation = data.get('chinese_translation', '')
                difficulty_level = int(data.get('difficulty_level', 3))
                difficulty_reasoning = data.get('difficulty_reasoning', '')
                is_executable = data.get('is_executable', True)
                executability_reason = data.get('executability_reason', '')
                
                return chinese_translation, difficulty_level, difficulty_reasoning, is_executable, executability_reason
            else:
                print(f"No JSON found in response: {content}")
                return '', 3, 'Parse error', True, 'Parse error'
                
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return '', 3, 'Parse error', True, 'Parse error'
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query through the complete pipeline"""
        if pd.isna(query) or not query.strip():
            return {
                'chinese_translation': '',
                'difficulty_level': 0,
                'difficulty_reasoning': 'Empty query',
                'is_executable': False,
                'executability_reason': 'Empty query'
            }
        
        prompt = self.create_prompt(query)
        response = self.call_openrouter_api(prompt)
        
        if response:
            chinese_translation, difficulty_level, difficulty_reasoning, is_executable, executability_reason = self.parse_response(response)
            return {
                'chinese_translation': chinese_translation,
                'difficulty_level': difficulty_level,
                'difficulty_reasoning': difficulty_reasoning,
                'is_executable': is_executable,
                'executability_reason': executability_reason
            }
        else:
            return {
                'chinese_translation': 'API Error',
                'difficulty_level': 0,
                'difficulty_reasoning': 'API call failed',
                'is_executable': False,
                'executability_reason': 'API call failed'
            }

def main():
    # Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    if not OPENROUTER_API_KEY:
        print("Please set your OPENROUTER_API_KEY environment variable")
        print("You can get an API key from: https://openrouter.ai/")
        return
    
    # File paths
    excel_file = "Workflow Evaluation_Table_Grid.xlsx"
    output_file = "Workflow_Evaluation_Processed.csv"
    
    # Load the Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name='Table')
        print(f"Loaded {len(df)} rows from {excel_file}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Find the column with queries (assumed to be 'Diligent Query' based on the column names)
        query_column = None
        for col in df.columns:
            if 'diligent' in col.lower() and 'query' in col.lower():
                query_column = col
                break
        
        if not query_column:
            print("Could not find 'Diligent Query' column. Available columns:")
            print(df.columns.tolist())
            return
        
        print(f"Processing queries from column: '{query_column}'")
        
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        return
    
    # Initialize processor
    processor = QueryProcessor(OPENROUTER_API_KEY)
    
    # Add new columns for results
    df['Chinese_Translation'] = ''
    df['Difficulty_Level'] = 0
    df['Difficulty_Reasoning'] = ''
    df['Is_Executable'] = False
    df['Executability_Reason'] = ''
    
    # Process each query
    total_queries = len(df)
    for index, row in df.iterrows():
        query = row[query_column]
        print(f"\nProcessing row {index + 1}/{total_queries}")
        print(f"Query: {str(query)[:100]}...")
        
        result = processor.process_query(str(query))
        
        # Update the dataframe
        df.at[index, 'Chinese_Translation'] = result['chinese_translation']
        df.at[index, 'Difficulty_Level'] = result['difficulty_level']
        df.at[index, 'Difficulty_Reasoning'] = result['difficulty_reasoning']
        df.at[index, 'Is_Executable'] = result['is_executable']
        df.at[index, 'Executability_Reason'] = result['executability_reason']
        
        print(f"Chinese: {result['chinese_translation'][:50]}...")
        print(f"Difficulty: {result['difficulty_level']}")
        print(f"Executable: {result['is_executable']}")
        
        # Add a small delay to be respectful to the API
        time.sleep(1)
        
        # Save progress every 10 rows
        if (index + 1) % 10 == 0:
            df.to_csv(f"temp_progress_{index + 1}.csv", index=False, encoding='utf-8-sig')
            print(f"Progress saved at row {index + 1}")
    
    # Save final results
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total queries processed: {total_queries}")
    print(f"Difficulty distribution:")
    print(df['Difficulty_Level'].value_counts().sort_index())
    print(f"Executable queries: {df['Is_Executable'].sum()}")
    print(f"Non-executable queries: {(~df['Is_Executable']).sum()}")

if __name__ == "__main__":
    main() 