"""
RAG Answer Relevancy (答案相关性) Prompt模板

评估答案是否直接回答了用户的问题，检测无关和冗余信息。
"""

from dingo.model.model import Model
from dingo.model.prompt.base import BasePrompt


@Model.prompt_register("QUALITY_BAD_ANSWER_RELEVANCY", ["rag"], ["LLMRAGAnswerRelevancy"])
class PromptRAGAnswerRelevancy(BasePrompt):
    """
    RAG答案相关性评估Prompt

    输入参数:
    - %s[0]: 问题 (question)
    - %s[1]: 答案 (answer)
    """

    _metric_info = {
        "category": "RAG Evaluation Metrics",
        "metric_name": "PromptRAGAnswerRelevancy",
        "description": "评估答案是否直接回答问题，检测无关和冗余信息",
        "paper_title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "paper_url": "https://arxiv.org/abs/2309.15217",
        "source_frameworks": "Ragas + DeepEval + TruLens"
    }

    content = """你是一个问答质量评估专家。你的任务是评估答案是否直接、完整地回答了用户的问题。

**评估目标**:
- 答案是否回答了问题
- 答案是否包含无关或冗余信息
- 答案的针对性和完整性

**判断标准**:
- 高分(8-10): 答案直接回答问题，信息准确且简洁
- 中分(4-7): 答案回答了问题但包含一些无关信息
- 低分(0-3): 答案大部分内容与问题无关或答非所问

**问题**:
{0}

**答案**:
{1}

**任务要求**:
1. 分析答案中的每个陈述是否与问题相关
2. 识别无关、冗余或偏题的内容
3. 评估答案的针对性和完整性
4. 计算相关性分数
5. 以JSON格式返回结果，不要输出其他内容

**输出格式**:
```json
{{
    "score": 0-10,
    "reason": "评估理由，指出相关和不相关的部分"
}}
```

其中score为0-10之间的整数，10表示答案完全相关，0表示答案完全不相关。
"""
