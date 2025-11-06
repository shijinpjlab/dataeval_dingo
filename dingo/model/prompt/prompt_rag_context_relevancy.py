"""
RAG Context Relevancy (上下文相关性) Prompt模板

评估检索到的上下文是否与问题相关。
"""

from dingo.model.model import Model
from dingo.model.prompt.base import BasePrompt


@Model.prompt_register("QUALITY_BAD_CONTEXT_RELEVANCY", ["rag"], ["LLMRAGContextRelevancy"])
class PromptRAGContextRelevancy(BasePrompt):
    """
    RAG上下文相关性评估Prompt

    输入参数:
    - {0}: 问题 (question)
    - {1}: 上下文 (contexts，已拼接)

    基于 Ragas、DeepEval 和 TruLens 的设计
    """

    _metric_info = {
        "category": "RAG Evaluation Metrics",
        "metric_name": "PromptRAGContextRelevancy",
        "description": "评估检索上下文与问题的相关性，检测噪声信息",
        "paper_title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "paper_url": "https://arxiv.org/abs/2309.15217",
        "source_frameworks": "Ragas + DeepEval + TruLens"
    }

    content = """你是一个信息相关性评估专家。你的任务是评估检索到的上下文是否与给定问题相关。

**评估目标**:
判断每个上下文是否包含与问题相关的信息

**评估流程**:
1. 理解问题的核心意图
2. 对每个上下文判断是否包含与问题相关的信息
3. 计算相关性分数 = (相关上下文数 / 总上下文数) × 10

**判断标准**:
- relevant (相关): 上下文包含与问题相关的信息，有助于回答问题
- irrelevant (不相关): 上下文与问题无关，或者是噪声信息、冗余信息

**问题**:
{0}

**检索到的上下文**:
{1}

**任务要求**:
1. 分析每个上下文是否与问题相关
2. 计算相关性分数
3. 以JSON格式返回结果，不要输出其他内容

**输出格式**:
```json
{{
    "score": 0-10,
    "reason": "评估理由，说明有多少上下文相关，有多少不相关"
}}
```

其中score为0-10之间的整数，10表示所有上下文都相关，0表示所有上下文都不相关。

**注意**: 不要考虑答案，只关注上下文与问题的相关性。
"""
