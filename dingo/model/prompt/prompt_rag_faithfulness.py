"""
RAG Faithfulness (忠实度) Prompt模板

评估生成的答案是否忠实于给定的上下文，检测幻觉和编造信息。
"""

from dingo.model.model import Model
from dingo.model.prompt.base import BasePrompt


@Model.prompt_register("QUALITY_BAD_FAITHFULNESS", ["rag"], ["LLMRAGFaithfulness"])
class PromptRAGFaithfulness(BasePrompt):
    """
    RAG忠实度评估Prompt

    输入参数:
    - %s[0]: 问题 (question)
    - %s[1]: 答案 (answer)
    - %s[2]: 上下文 (contexts，已拼接)
    """

    _metric_info = {
        "category": "RAG Evaluation Metrics",
        "metric_name": "PromptRAGFaithfulness",
        "description": "评估生成答案是否忠实于给定上下文，检测幻觉和编造信息",
        "paper_title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "paper_url": "https://arxiv.org/abs/2309.15217",
        "source_frameworks": "Ragas + DeepEval"
    }

    content = """你是一个严格的事实验证专家。你的任务是评估一个答案是否忠实于给定的上下文。

**评估流程**:
1. 从答案中提取独立的事实陈述
2. 对每个陈述验证是否能从上下文推导
3. 计算忠实陈述的比例

**判断标准**:
- faithful (忠实): 陈述可以从上下文中直接推导或明确支持
- unfaithful (不忠实): 陈述无法从上下文推导，或与上下文矛盾，或包含上下文中没有的信息

**问题**:
{0}

**答案**:
{1}

**上下文**:
{2}

**任务要求**:
1. 提取答案中的独立陈述（每个陈述应该是完整的、可独立验证的事实）
2. 对每个陈述判断是否忠实于上下文
3. 计算忠实度分数 = 忠实陈述数量 / 总陈述数量
4. 以JSON格式返回结果，不要输出其他内容

**输出格式**:
```json
{{
    "score": 0-10,
    "reason": "评估理由说明"
}}
```

其中score为0-10之间的整数，10表示完全忠实，0表示完全不忠实。
"""
