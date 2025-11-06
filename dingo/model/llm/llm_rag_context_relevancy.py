"""
RAG Context Relevancy (上下文相关性) LLM评估器

基于LLM评估检索上下文与问题的相关性。
"""

import json
from typing import List

from dingo.io import Data
from dingo.model import Model
from dingo.model.llm.base_openai import BaseOpenAI
from dingo.model.modelres import ModelRes
from dingo.model.prompt.prompt_rag_context_relevancy import PromptRAGContextRelevancy
from dingo.model.response.response_class import ResponseScoreReason
from dingo.utils import log
from dingo.utils.exception import ConvertJsonError


@Model.llm_register("LLMRAGContextRelevancy")
class LLMRAGContextRelevancy(BaseOpenAI):
    """
    RAG上下文相关性评估LLM

    输入要求:
    - input_data.prompt 或 raw_data['question']: 用户问题
    - input_data.context 或 raw_data['contexts']: 检索到的上下文列表

    注意: Context Relevancy 只需要问题和上下文，不需要答案
    """

    prompt = PromptRAGContextRelevancy

    @classmethod
    def build_messages(cls, input_data: Data) -> List:
        """
        构建LLM输入消息

        Args:
            input_data: 输入数据

        Returns:
            消息列表
        """
        # 提取字段
        question = input_data.prompt or input_data.raw_data.get("question", "")

        # 处理contexts
        contexts = None
        if input_data.context:
            if isinstance(input_data.context, list):
                contexts = input_data.context
            else:
                contexts = [input_data.context]
        elif "contexts" in input_data.raw_data:
            raw_contexts = input_data.raw_data["contexts"]
            if isinstance(raw_contexts, list):
                contexts = raw_contexts
            else:
                contexts = [raw_contexts]

        if not question:
            raise ValueError("Context Relevancy评估需要question字段")
        if not contexts:
            raise ValueError("Context Relevancy评估需要contexts字段")

        # 拼接上下文
        combined_contexts = "\n\n".join([f"上下文{i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # 构建prompt内容
        prompt_content = cls.prompt.content.format(question, combined_contexts)

        messages = [{"role": "user", "content": prompt_content}]

        return messages

    @classmethod
    def process_response(cls, response: str) -> ModelRes:
        """
        处理LLM响应

        Args:
            response: LLM原始响应

        Returns:
            ModelRes对象
        """
        log.info(f"RAG Context Relevancy response: {response}")

        # 清理响应
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            response_json = json.loads(response.strip())
        except json.JSONDecodeError:
            raise ConvertJsonError(f"Convert to JSON format failed: {response}")

        # 解析响应
        response_model = ResponseScoreReason(**response_json)

        result = ModelRes()
        result.score = response_model.score

        # 根据分数判断是否通过（默认阈值5，满分10分）
        threshold = 5
        if hasattr(cls, 'dynamic_config') and cls.dynamic_config.parameters:
            threshold = cls.dynamic_config.parameters.get('threshold', 5)

        if response_model.score >= threshold:
            result.error_status = False
            result.type = "QUALITY_GOOD"
            result.name = "CONTEXT_RELEVANCY_PASS"
            result.reason = [f"上下文相关性评估通过 (分数: {response_model.score}/10)\n{response_model.reason}"]
        else:
            result.error_status = True
            result.type = cls.prompt.metric_type
            result.name = cls.prompt.__name__
            result.reason = [f"上下文相关性评估未通过 (分数: {response_model.score}/10)\n{response_model.reason}"]

        return result
