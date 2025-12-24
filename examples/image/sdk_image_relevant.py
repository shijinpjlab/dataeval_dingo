import os
from pathlib import Path

from dingo.config import InputArgs
from dingo.exec import Executor

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


def image_relevant():
    # 从环境变量获取 API 配置
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_url = os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")

    input_data = {
        "input_path": str(PROJECT_ROOT / "test/data/test_img_jsonl.jsonl"),
        "output_path": "output/hallucination_evaluation/",
        "dataset": {
            "source": "local",
            "format": "jsonl",
        },
        "executor": {
            "result_save": {
                "bad": True,
                "good": True
            }
        },
        "evaluator": [
            {
                "fields": {"id": "id", "prompt": "url_1", "content": "url_2"},
                "evals": [
                    {"name": "VLMImageRelevant", "config": {"model": model, "key": api_key, "api_url": api_url}},
                ]
            }
        ]
    }
    input_args = InputArgs(**input_data)
    executor = Executor.exec_map["local"](input_args)
    result = executor.execute()
    print(result)


if __name__ == '__main__':
    image_relevant()
