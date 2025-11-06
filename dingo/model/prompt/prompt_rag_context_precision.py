"""
RAG Context Precision (上下文精度) Prompt模板

评估检索到的上下文的精确度，即相关上下文的比例和排序质量。
"""

from dingo.model.model import Model
from dingo.model.prompt.base import BasePrompt


@Model.prompt_register("QUALITY_BAD_CONTEXT_PRECISION", ["rag"], ["LLMRAGContextPrecision"])
class PromptRAGContextPrecision(BasePrompt):
    """
    RAG上下文精度评估Prompt

    输入参数:
    - %s[0]: 问题 (question)
    - %s[1]: 答案 (answer)
    - %s[2]: 上下文列表 (contexts，每行一个)
    """

    _metric_info = {
        "category": "RAG Evaluation Metrics",
        "metric_name": "PromptRAGContextPrecision",
        "description": "评估检索上下文的精确度，包括相关性和排序质量",
        "paper_title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "paper_url": "https://arxiv.org/abs/2309.15217",
        "source_frameworks": "Ragas"
    }

    content = """你是一个信息检索专家。你的任务是评估检索到的上下文是否对回答问题有帮助。

**评估目标**:
- 判断每个上下文是否与问题和答案相关
- 评估上下文的排序质量（相关的应该排在前面）

**判断标准**:
- relevant (相关): 上下文包含有助于回答问题的信息
- not_relevant (不相关): 上下文与问题无关或不包含有用信息

**问题**:
{0}

**答案**:
{1}

**检索到的上下文**:
{2}

**任务要求**:
1. 按顺序评估每个上下文的相关性
2. 计算平均精度（Average Precision），考虑排序质量
3. 相关上下文排在前面会得到更高分数
4. 以JSON格式返回结果，不要输出其他内容

**输出格式**:
```json
{{
    "score": 0-10,
    "reason": "评估理由，说明各上下文的相关性"
}}
```

其中score为0-10之间的整数，10表示所有上下文相关且排序完美，0表示所有上下文都不相关。
"""
