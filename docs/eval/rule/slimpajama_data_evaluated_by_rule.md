# Slimpajama Dataset

## Dataset Introduction
This dataset aims to evaluate the accuracy of the built-in rules in dingo. Therefore, the open-source dataset Slimpajama was selected, and data was extracted from it to construct the test set.

| Field Name   | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| data_id      | Data ID, without special meaning, can be modified according to user needs     |
| content      | Data to be tested                                                             |
| language     | Language type                                                                 |
| eval_status | Data status, True for negative examples, False for positive examples          |
| type_list    | Negative example types for negative data, empty list for positive data        |
| name_list    | Negative example names for negative data, empty list for positive data        |
| reason_list  | Negative example descriptions for negative data, empty list for positive data |

Links:
https://huggingface.co/datasets/chupei/slimpajama_badcase_rule
https://huggingface.co/datasets/chupei/slimpajama_goodcase_rule

### Dataset Composition
| Type                                            | Count |
|-------------------------------------------------|-------|
| Positive examples                               | 82    |
| Negative examples: RuleAlphaWords               | 27    |
| Negative examples: RuleCapitalWords             | 26    |
| Negative examples: RuleCharNumber               | 5     |
| Negative examples: RuleDocRepeat                | 17    |
| Negative examples: RuleHtmlEntity               | 3     |
| Negative examples: RuleLineEndWithEllipsis      | 5     |
| Negative examples: RuleLineEndWithTerminal      | 5     |
| Negative examples: RuleLineStartWithBulletpoint | 6     |
| Negative examples: RuleLoremIpsum               | 5     |
| Negative examples: RuleMeanWordLength           | 12    |
| Negative examples: RuleNoPunc                   | 7     |
| Negative examples: RuleSentenceNumber           | 8     |
| Negative examples: RuleSpecialCharacter         | 4     |
| Negative examples: RuleStopWord                 | 24    |
| Negative examples: RuleSymbolWordRatio          | 5     |
| Negative examples: RuleUniqueWords              | 7     |
| Negative examples: RuleWordNumber               | 7     |

## Rules Introduction
This test uses the **pretrain** group rules explicitly configured in the evaluator. For specific rules included, please refer to: [Group Introduction](../../groups.md).<br>
For rules within the group, please refer to: [Rules Introduction](../../rules.md).

## Evaluation Results
### Definitions
After evaluation, both positive and negative data will generate corresponding summary files. Therefore, the results need to be defined with clear concepts.

| Term     | Description                                                                    |
|----------|--------------------------------------------------------------------------------|
| TP       | True Positive: Number of positive examples correctly identified                |
| FP       | False Positive: Number of negative examples incorrectly identified as positive |
| TN       | True Negative: Number of negative examples correctly identified                |
| FN       | False Negative: Number of positive examples incorrectly identified as negative |
| Accuracy | TP / (TP + FP) Ratio of positive examples in the identified positives          |
| Recall   | TP / (TP + FN) Ratio of positive examples correctly identified                 |
| F1       | (Accuracy + Recall) / 2                                                        |

### Results Display
| Dataset Name | TP | FP | TN  | FN | Accuracy% | Recall% | F1   |
|--------------|----|----|-----|----|-----------|---------|------|
| slimpajama   | 78 | 5  | 103 | 4  | 94        | 95      | 94.5 |

## Evaluation Method

```python
from dingo.config import InputArgs
from dingo.exec import Executor

input_data = {
    "input_path": "chupei/slimpajama_badcase_rule",
    "dataset": {
        "source": "huggingface",
        "format": "jsonl",
    },
    "executor": {
        "max_workers": 10,
        "batch_size": 10,
        "result_save": {
            "bad": True,
            "good": True,
            "raw": True
        }
    },
    "evaluator": [
        {
            "fields": {"content": "content"},
            "evals": [
                # Pretrain group rules - see docs/groups.md for full list
                {"name": "RuleAlphaWords"},
                {"name": "RuleCapitalWords"},
                {"name": "RuleCharNumber"},
                {"name": "RuleColonEnd"},
                {"name": "RuleContentNull"},
                {"name": "RuleDocRepeat"},
                {"name": "RuleHtmlEntity"},
                {"name": "RuleIDCard"},
                {"name": "RuleLineEndWithEllipsis"},
                {"name": "RuleLineEndWithTerminal"},
                {"name": "RuleLineStartWithBulletpoint"},
                {"name": "RuleLineJavascriptCount"},
                {"name": "RuleLoremIpsum"},
                {"name": "RuleMeanWordLength"},
                {"name": "RuleNoPunc"},
                {"name": "RuleSentenceNumber"},
                {"name": "RuleSpecialCharacter"},
                {"name": "RuleStopWord"},
                {"name": "RuleSymbolWordRatio"},
                {"name": "RuleUniqueWords"},
                {"name": "RuleWordNumber"},
            ]
        }
    ]
}
input_args = InputArgs(**input_data)
executor = Executor.exec_map["local"](input_args)
result = executor.execute()
print(result)
```
