from pathlib import Path

from dingo.config import InputArgs
from dingo.exec import Executor
from dingo.model.rule.rule_sciencemetabench import write_similarity_to_excel

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

if __name__ == '__main__':
    input_data = {
        "input_path": str(PROJECT_ROOT / "test/data/sciencemetabench/paper.jsonl"),
        "dataset": {
            "source": "local",
            "format": "jsonl",
        },
        "executor": {
            "result_save": {
                "merge": True,
            }
        },
        "evaluator": [
            {
                # "fields": {"content": "content"},
                "evals": [
                    {"name": "RuleMetadataMatchPaper", "config": {"threshold": 0.8}}
                ]
            }
        ]
    }
    input_args = InputArgs(**input_data)
    executor = Executor.exec_map["local"](input_args)
    result = executor.execute()
    print(result)

    write_similarity_to_excel("paper", result.output_path)
