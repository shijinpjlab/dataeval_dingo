# Dataset Kaoti

## Dataset Introduction
This dataset aims to evaluate the accuracy of the built-in kaoti prompt words in dingo, therefore, the test question data was selected to construct the test set.

| Field Name   | Description                                                                        |
|--------------|------------------------------------------------------------------------------------|
| id           | DATA id, without special meaning, users can modify it according to their own needs |
| grade_class  | The classification of students based on their academic grade levels                |
| major        | Main area of knowledge and skills                                                  |
| content      | Data to be tested                                                                  |                                                  |



### Dataset Composition
| Type                                                                                  | Count |
|---------------------------------------------------------------------------------------|-------|
| Positive Examples                                                                     | 100   |
| Negative Examples: <br/>1. ineffectiveness<br/>2. dissimilarity<br/>3. incompleteness | 100   |


## LLM Evaluator Introduction
The built-in **LLMTextQualityKaoti** is used as the LLM evaluator for this test.<br>
Specific content can be referred to: [Introduction to LLMTextQualityKaoti](../../../dingo/model/llm/llm_text_quality_kaoti.py)<br>
The built-in LLM evaluator collection can be referred to: [LLM Collection](../../../dingo/model/llm)

## Evaluation Results
### Concept Introduction
Both positive and negative examples will generate corresponding summary files after evaluation, so the results need to be defined and the concepts clarified.

| Name      | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| TP        | True Positive: Number of positive examples evaluated as positive            |
| FP        | False Positive: Number of negative examples evaluated as positive           |
| TN        | True Negative: Number of negative examples evaluated as negative            |
| FN        | False Negative: Number of positive examples evaluated as negative           |
| Precision | TP / (TP + FP) Ratio of positive examples among those evaluated as positive |
| Recall    | TP / (TP + FN) Ratio of positive examples correctly evaluated as positive   |
| F1        | 2 * Accuracy * Recall /  (Accuracy + Recall)                                |

### Result Display
| Dataset Name | TP  | FP  | TN  | FN  | Precision% | Recall% | F1   |
|--------------|-----|-----|-----|-----|------------|---------|------|
| redpajama    | 86  | 15  | 85  | 14  | 85         | 86      | 0.856|
## Evaluation Method

```python
from dingo.config import InputArgs
from dingo.exec import Executor

input_data = {
    "input_path": "/your/dataset/path",  # s3 path: qa-huawei
    "dataset": {
        "source": "local",
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
                {"name": "LLMTextQualityKaoti", "config": {
                    "key": "Your Key",
                    "api_url": "Your Url"
                }}
            ]
        }
    ]
}
input_args = InputArgs(**input_data)
executor = Executor.exec_map["local"](input_args)
result = executor.execute()
print(result)
```
