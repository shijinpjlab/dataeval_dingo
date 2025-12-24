import os

from dingo.config.input_args import EvaluatorLLMArgs
from dingo.io.input import Data
from dingo.model.llm.text_quality.llm_text_quality_v5 import LLMTextQualityV5
from dingo.model.rule.rule_common import RuleEnterAndSpace

# Configure LLM (set your API key via environment variable OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
OPENAI_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")  # Set OPENAI_API_KEY env var


def llm():
    data = Data(
        data_id='123',
        prompt="hello, introduce the world",
        content="Hello! The world is a vast and diverse place, full of wonders, cultures, and incredible natural beauty."
    )

    LLMTextQualityV5.dynamic_config = EvaluatorLLMArgs(
        model=OPENAI_MODEL,
        key=OPENAI_KEY,
        api_url=OPENAI_URL,
    )
    res = LLMTextQualityV5.eval(data)
    print(res)


def rule():
    data = Data(
        data_id='123',
        prompt="hello, introduce the world",
        content="Hello! The world is a vast and diverse place, full of wonders, cultures, and incredible natural beauty."
    )

    res = RuleEnterAndSpace().eval(data)
    print(res)


if __name__ == "__main__":
    llm()
    rule()
