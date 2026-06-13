import asyncio
import os
import uuid
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
backend_root = os.path.join(project_root, 'backend')
sys.path.insert(0, project_root)
sys.path.insert(0, backend_root)

from agent.src.browser_agent.agents import BrowserAgent
from agent.src.smolagents import OpenAIServerModel


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

async def main():
    agent_uuid = str(uuid.uuid4())
    work_root = "./data"
    work_dir = os.path.join(work_root, agent_uuid)
    # 或者使用自定义模型
    model = OpenAIServerModel(
        model_id="anthropic/claude-sonnet-4",
        api_base="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.3,
        max_tokens=10000,
    )

    tokens_dict = {
        "google": {
            "access_token": "xxx",
            "refresh_token": "xxx",
        },
        "airtable": {"api_key": "airtable_api_key_value"},
    }

    agent = BrowserAgent(
        model=model,
        headless=True,  # 无头模式
        max_steps=50,  # 自定义最大步数
        use_redis=False,
        stream_outputs=True,
        tokens_dict=tokens_dict,
        additional_authorized_imports=["*"],  # 允许导入所有包
        executor_kwargs={
            "work_dir": work_dir
        }
        # use_structured_outputs_internally=True,  # 启用结构化输出
        # use_xml_format=False,
        # prompt_templates=custom_prompts,
    )

    await agent.run_task_simple("""
帮我生成包含所有评估任务的agent_evaluation_config.json。
所有任务的文件夹在/Users/baiyl/workspace/sophia/git/AgentWorld/deliverable_data/sophia，其中每个任务在一个子文件夹，例如/Users/baiyl/workspace/sophia/git/AgentWorld/deliverable_data/sophia/1/，文件夹名称是任务的id。
每个子文件夹有四个文件你能用到，一个是lazy_query.md，一个是deligent_query.md，一个是evaluation_criteria.md，一个是delivrable.md，其中lazy_query.md是任务的描述，deligent_query.md是任务的描述，evaluation_criteria.md是评估标准，delivrable.md交代了交付物是什么。
你需要参考这个config结构，帮我生成所有任务的config，最后保存到/Users/baiyl/workspace/sophia/git/AgentWorld/config/agent_based，用任务id命名。


你需要把criteria结合deligent query拆分成得分点，一共5分，每个得分点0.5-1分。

由于你的上下文长度有限，你可以使用call_llm来生成config（请调用anthropic/claude-sonnet-4，是目前最强大的），在调用之前，请使用以下这个prompt让LLM生成config，并封装成一个函数以便你高效调用。

[Config生成prompt开始]                 
You are tasked with generating an agent evaluation configuration JSON for task {task_id}.                                                        
                                                                                                                                                
Here are the task details:                                                                                                                       
- Task ID: {task_id}                                                                                                                             
- Lazy Query: {lazy_query}                                                                                                                       
- Diligent Query: {diligent_query}         
- Delivrable: {delivrable}                                                                                                     
- Evaluation Criteria: {evaluation_criteria}                                                                                                     
                                                                                                                                                
Please generate a JSON configuration following this exact structure:                                                                             
{{                                                                                                                                               
    "{task_id}": {{                                                                                                                              
    "workflow_agent_eval": {{                                                                                                                  
        "name": "A concise, descriptive name for this workflow",                                                                                 
        "description": "A brief description of what this workflow accomplishes",                                                                 
        "type": "agent_based",                                                                                                                   
        "task_config": {{                                                                                                                        
        "initial_environment": "Browser",                                                                                                      
        "query": {{                                                                                                                            
            "lazy": "{lazy_query}",                                                                                                              
            "diligent": "{diligent_query}"                                                                                                       
        }},                                                                                                                                    
        "deliverable": "{delivrable}"                                                                                                     
        }},                                                                                                                                      
        "evaluation_config": {{                                                                                                                  
        "agent_evaluation": {{                                                                                                                 
            "prompt_template": "A comprehensive prompt template for evaluating the agent's performance on this task"                             
        }},                 
        "columns": [
            {"id": "1", "name": "Column Name 1"},
            {"id": "2", "name": "Column Name 2"},
            // For any task involving structured CSV-like data, field-level validation must be explicitly stated
            // If the task is not about structured CSV-like data, this section can be omitted
        ],
        "criteria": [
            {
                "name": "Criterion Name 1",
                "description": "A brief explanation of what this criterion assesses and how it is judged.",
                "points": 1
            },
            {
                "name": "Criterion Name 2",
                "description": "Another measurable and specific aspect of the task to be evaluated.",
                "points": 0.5
            }
            // Include 3–10 total criteria, each 0.5–1 point, summing to exactly 5
            // Base these criteria on the evaluation_criteria provided
        ]                                                                                                                                
        "file_matching": {{                                                                                                                    
            "output_pattern": "*.csv", # Output file is strictly according to the "delivrable"                                
            "auto_detect": true                                                                                                                  
        }}                                                                                                                                     
        }}                                                                                                                                       
    }}                                                                                                                                         
    }}                                                                                                                                           
}}                                                                                                                                               
                                                                                                                                    
Important guidelines:                                                                                                                            
1. The "name" field should be a concise and informative title that summarizes the purpose of the workflow.                   
2. The "description" should clearly state the goal of the workflow in one sentence, highlighting what the agent is expected to accomplish.
3. The "deliverable" should specify the format and nature of the final output (e.g., “A CSV file listing 100 Instagram influencers with specified attributes”).             
4. The "columns" section are ONLY required for deliverable including CSV or Excel data, and must list all the columns that are expected in the output file, the expected columns is provided in the evaluation criteria.
5. The "criteria" section must break down the overall evaluation into 3 to 10 specific and measurable components, totaling exactly 5 points.
    - Each criterion must directly reflect the provided evaluation_criteria.
    - Each criterion should be worth between 0.5 and 3 points, depending on its complexity and importance.
    - For example, if the task requires collecting a CSV with 100 specific fields, include a criterion like: "Successfully collects all 100 required fields, including username, profile link, engagement rate, etc. Evaluator must check completeness and accuracy."
    - Make sure the criteria are actionable, and easy to verify, so they can be evaluated objectively by an agent evaluator.                      
6. The prompt_template should be a comprehensive instruction for evaluating the agent's work                                                                                                                                                         
Please return ONLY the JSON configuration, no additional text or explanation.
[Config生成prompt结束]                                 
    
""")

if __name__ == "__main__":    
    asyncio.run(main())
