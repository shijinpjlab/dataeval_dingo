"""
RAG Context Recall (上下文召回) Prompt模板

评估检索到的上下文是否完整地支持了答案中的信息。
"""

from dingo.model.model import Model
from dingo.model.prompt.base import BasePrompt


@Model.prompt_register("QUALITY_BAD_CONTEXT_RECALL", ["rag"], ["LLMRAGContextRecall"])
class PromptRAGContextRecall(BasePrompt):
    """
    RAG上下文召回评估Prompt

    输入参数:
    - {0}: 问题 (question)
    - {1}: 答案/期望输出 (expected_output)
    - {2}: 上下文 (contexts，已拼接)

    基于 Ragas 和 DeepEval 的设计
    """

    _metric_info = {
        "category": "RAG Evaluation Metrics",
        "metric_name": "PromptRAGContextRecall",
        "description": "评估检索上下文的完整性，判断上下文是否能支持答案中的所有陈述",
        "paper_title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "paper_url": "https://arxiv.org/abs/2309.15217",
        "source_frameworks": "Ragas + DeepEval"
    }

    content = """你是一个严格的事实核查专家。你的任务是评估检索到的上下文是否完整地支持了给定答案中的所有信息。

**评估目标**:
判断答案中的每个陈述是否能从上下文中找到支持证据

**评估流程**:
1. 从答案中提取独立的事实陈述
2. 对每个陈述，判断是否能从上下文中归因（找到支持证据）
3. 计算上下文召回率 = 可归因陈述数 / 总陈述数

**判断标准**:
- attributed (可归因): 陈述可以从上下文中直接找到或合理推导出
- not attributed (不可归因): 陈述在上下文中没有支持证据

**问题**:
{0}

**答案**:
{1}

**检索到的上下文**:
{2}

**任务要求**:
1. 提取答案中的所有独立陈述（每个陈述应该是完整的、可独立验证的事实）
2. 对每个陈述判断是否可以从上下文归因
3. 计算召回率分数 = (可归因陈述数 / 总陈述数) × 10
4. 以JSON格式返回结果，不要输出其他内容

**输出格式**:
```json
{{
    "score": 0-10,
    "reason": "评估理由，说明有多少陈述可以归因，有多少不能归因"
}}
```

其中score为0-10之间的整数，10表示所有陈述都能归因（完美召回），0表示所有陈述都不能归因。
"""
