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


# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_api_key = "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"

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
你是一位严谨、结构化思维强、语言表达专业的达人营销与增长领域专家，也具备Agent系统评估与算法能力分析经验。
你将从指定目录中读取多个结构相似的任务评估报告和配套评分文件，提取核心信息，并撰写一份专业、结构清晰的综合评估报告，格式为Markdown，保存到本地路径。

输入目录说明

这些评估报告的路径如下：

/Users/baiyl/workspace/sophia/git/AgentWorld/eval_results

该路径下包含4个子文件夹，分别代表4个模型（chatgpt、comet、manus、sophia）。每个模型文件夹下含若干数字编号的子目录（如 2/），每个子目录包含：
                                
1. eval_report_{n}.md：主评估报告，结构一致，包含五部分：
    - 一、优点
    - 二、缺点
    - 三、暴露的Agent问题
    - 四、完成度分析
    - 五、质量评分分析
                                
2. completeness_{n}.json：完成度得分，格式为：
{{
  "Completeness": 完成度分数（满分1分）
}}

3. quality_{n}.json：质量得分，格式为：
{{
  "criteria_scores": {{
    "criterion1 name": 分数,
    "criterion2 name": 分数,
    "criterion3 name": 分数
  }},
  "overall": 总评分（满分5分）
}}



你的任务目标：

请读取所有报告内容和得分文件，基于以下结构撰写一份《Sophia Agent 评估与竞品对比综合分析报告》，要求内容逻辑清晰、结论可信、语言专业。

目标报告结构（请严格遵守）：

一、评估概况
	概述本次评估情况，可用文字和表格总结。包括评估基本信息（模型、任务、纬度、来源），各模型进本表现概览表
    - 你需要用代码计算每个模型的完成度得分和质量得分平均值，并在**写入报告前核实你的结果**。
    - 你要在表格中标注出每个模型在完成度和质量上分别排名第几

二、整体结果概览
主要以美观的图表形式呈现，包括：

1. 完成度区间分布对比柱状图（纵向）
	•	将任务按完成度划分为四个区间：
        •	未完成：[0, 0.2)
        •	低完成：[0.2, 0.5)
        •	中完成：[0.5, 0.8)
        •	高完成：[0.8, 1]
	•	每个区间为一组，共四组；每组内含四个模型（共 16 根柱子）。
	•	每组之间保留视觉间距；组内柱子紧凑排列。

2. 质量得分区间分布对比柱状图（纵向）
	•	将任务按质量得分划分为四个区间：
        •	极低表现：[0, 2)
        •	低表现：[2, 3)
        •	中表现：[3, 4)
        •	高表现：[4, 5]
	•	同样为四组，每组内四个模型，共 16 根柱子。

3. 完成度 win-tie-lose 横向对比柱状图
	•	统计 Sophia 与每个竞品在任务完成度上的对比结果（Win / Tie / Lose）。
	•	图表为 3 个横向柱状条（每个条代表与一个竞品的对比）。
	•	每条内部由三部分组成：Win / Tie / Lose。
	•	横条需加边框，三条长度一致，强调比例对比。
    •   要用百分比表示win/tie/lose情况

4. 质量得分 win-tie-lose 横向对比柱状图
	•	与图 3 类似，但比较维度为“质量得分”。

图表风格要求：
	•	Sophia 色系固定为红色调 #fc7176，务必在所有图中显眼突出。
	•	其余三个竞品色系选择青绿色系，分别为：#71d7cf、#6ac5da、#87ceeb。
	•	横向 win-tie-lose 图中配色如下：
        •	Sophia Win：#fc7176
        •	Tie：浅黄色（如 #fff4c2）
        •	Lose：#87ceeb
	•	所有图表需保持统一美观风格（字体、线条、颜色、背景网格、图例样式等）。
	•	横向柱图需添加边框增强可读性。
	•	每张图下方需添加简洁但有洞察力的数据分析文字说明（详见下方格式）。

请为每张图撰写一段简要的数据洞察说明，例如：

完成度分布图分析：
Sophia 在高完成度区间（0.8–1）中任务占比为 XX%，Comet 在...

或：

完成度 Win-Tie-Lose 对比结果：
	•	vs Comet：胜出占比...
	•	vs Manus：...

附加说明：
	•	不可遗漏任何模型的任务数据，确保图中任务全量统计。
	•	如有其他可补充说明的结构性信息，可考虑使用简洁的表格或额外图表呈现。

         
三、Sophia Agent优点总结

请提取所有sophia的评估报告中反复出现的正面特征和成功因素，可以对优点进行分类，请分成5类，每类附典型示例（引用原文中的语句）。

四、Sophia Agent共性问题归纳

请归纳在多数任务中反复出现的问题类型，如数据造假、字段缺失、结构混乱、目标理解偏差等，指出频率。请分成10类，每类提供频率统计及典型任务中的原文示例。频率百分比也放到标题中，方便一眼就能看到。

五、综合评价与建议
结合专业角度，给出对 Sophia Agent 的系统总结，包括：
    - 模型能力评估：在哪些类型任务表现突出？在哪些任务存在明显短板？
    - 优化建议：例如提示工程（prompt）、工具接口改造、边界收紧、数据过滤等。
    - 评估流程反思：自动化评估是否覆盖全面？主观判断是否仍占主导？是否可进一步标准化？
            

你的输出要求
    - 报告格式：Markdown
    - 文件路径：/Users/baiyl/workspace/sophia/git/AgentWorld/eval_results/Sophia_评估综合分析报告.md
    - 风格要求：语言专业、结构清晰、避免随意语气，可用列表/表格/图表辅助表达
    - 示例引用：适量引用原报告中的语句以增强说服力
    - 图表制作：使用 matplotlib，每个图独立展示


你不需要提前加载并解析所有 eval_report_{n}.md 文件，但是你需要意识到你的总共只有30个步骤可以执行，所以你需要合理规划你的步骤。
如果你要使用matplotlib画图，请使用matplotlib.use('Agg')  # 必须放在 import pyplot 之前，因为我是mac系统！！

另外，图表我喜欢下面这个风格，总结如下：

🔤 字体设置

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

	•	使用了中文字体支持（适配不同系统的字体），以确保图表标题、坐标轴标签中的中文能正常显示。
	•	设置了负号正常显示。

🎨 配色方案

1. 固定颜色（用于柱状图 - 完成度和质量）

colors = ['#2E8B57', '#4169E1', '#FF6347', '#DC143C']  # 完成度
quality_colors = ['#32CD32', '#4169E1', '#FF6347', '#8B0000']  # 质量

	•	使用了自定义的颜色序列，每种等级一个色块，视觉清晰。
	•	色彩鲜明，区分度高，有助于分辨不同类别。

2. 渐变色（用于横向柱状图 - 优点和问题）

adv_colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(adv_categories)))
prob_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(prob_categories)))

	•	利用 matplotlib 的 colormap 进行渐变填色。
	•	渐变色范围选在中间值段（0.4 - 0.8），避免过于鲜艳或暗淡。
	•	绿色代表优点，红色代表问题，符合认知习惯。


🧱 图表类型
	•	使用了 bar()（纵向柱状图）和 barh()（横向柱状图）两种方式：
	•	bar()：用于“完成度”和“质量”。
	•	barh()：用于“优点”和“问题”。


📏 线条样式

edgecolor='black', linewidth=1

	•	所有柱子都有黑色边框，提升视觉边界清晰度。
	•	边框宽度设置为 1，适中不突兀。


📊 标签与注释

ax.text(..., f'{value}', ...)

	•	每个柱子上方（或右侧）都显示具体的数值，提升信息可读性。
	•	使用 fontsize=11, fontweight='bold'，保证数值醒目。

🗂 标题与坐标轴

ax.set_title(..., fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(..., fontsize=12)
ax.set_ylabel(..., fontsize=12)

	•	标题使用加粗、较大字体。
	•	pad=20 为标题与图之间预留间距，排版更整洁。
	•	坐标轴标签简洁明了。

🧱 背景网格

ax.grid(axis='y' or 'x', alpha=0.3)

	•	添加了背景参考网格，仅在主要方向上（纵向柱图使用 axis='y'，横向柱图用 axis='x'）。
	•	网格线透明度设置为 0.3，不干扰主图但能辅助读数。


💾 输出图表

plt.savefig(f'{output_dir}xxx.png', dpi=300, bbox_inches='tight')

	•	每个图表单独保存为 .png，分辨率 300dpi，适合高清打印或嵌入报告。
	•	bbox_inches='tight' 去除多余空白边缘。

                                
""")

if __name__ == "__main__":    
    asyncio.run(main())
