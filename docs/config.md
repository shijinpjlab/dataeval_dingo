# Config

`Dingo` 为不同模块设置了各自的配置，让用户可以更加自由地使用项目完成自身的质检需求。

## CLI Config

用户在命令行输入指令启动项目时只需要指定配置文件路径：

| Parameter | Type | Default | Required | Description           |
|-----------|------|---------|----------|-----------------------|
| --input / -i | str  | None    | Yes      | 配置文件路径           |

## 配置文件结构

配置文件采用 JSON 格式，包含以下主要结构：

### InputArgs 根配置

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| task_name | str | "dingo" | No | 任务名称 |
| input_path | str | "test/data/test_local_json.json" | Yes | 要检查的文件或目录路径 |
| output_path | str | "outputs/" | No | 结果输出路径 |
| log_level | str | "WARNING" | No | 日志级别，可选值：['DEBUG', 'INFO', 'WARNING', 'ERROR'] |
| use_browser | bool | false | No | 是否使用浏览器进行可视化 |

### Dataset 配置 (dataset)

数据集相关配置：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| source | str | "hugging_face" | Yes | 数据源类型，可选值：['hugging_face', 'local'] |
| format | str | "json" | Yes | 数据格式，可选值：['json', 'jsonl', 'plaintext', 'listjson', 'image', 'multi_turn_dialog'] |
| hf_config | object | - | No | HuggingFace 特定配置 |
| s3_config | object | - | No | S3 存储配置 |
| sql_config | object | - | No | SQL 数据库配置 |
| excel_config | object | - | No | Excel 文件配置 |

#### DatasetHFConfig 配置 (dataset.hf_config)

HuggingFace 特定配置：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| huggingface_split | str | "" | No | HuggingFace 数据集分割 |
| huggingface_config_name | str | null | No | HuggingFace 配置名称 |

### Executor 配置 (executor)

执行器相关配置：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| start_index | int | 0 | No | 开始检查的数据索引 |
| end_index | int | -1 | No | 结束检查的数据索引 |
| max_workers | int | 1 | No | 最大并发工作线程数 |
| batch_size | int | 1 | No | 并发检查的最大数据量 |
| multi_turn_mode | str | null | No | 多轮对话解析模式 |
| result_save | object | - | No | 结果保存配置 |

#### ExecutorResultSave 配置 (executor.result_save)

结果保存配置：

| Parameter  | Type | Default | Required | Description |
|------------|------|---------|----------|-------------|
| bad        | bool | true    | No       | 是否保存错误结果    |
| good       | bool | false   | No       | 是否保存正确结果    |
| all_labels | bool | false   | No       | 是否保存所有标签    |
| raw        | bool | false   | No       | 是否保存原始数据    |

### Evaluator 配置 (evaluator)

评估器配置采用数组形式，支持多个评估管道（EvalPipline）：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| evaluator | array | [] | Yes | 评估管道数组 |

#### EvalPipline 配置 (evaluator[])

每个评估管道包含字段映射和评估器列表：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| fields | object | {} | Yes | 字段映射配置，将数据字段映射到评估器需要的字段 |
| evals | array | [] | Yes | 评估器列表 |

**fields 字段映射说明**：

| 映射字段 | Description |
|----------|-------------|
| id | 数据 ID 字段名 |
| prompt | prompt/问题字段名 |
| content | 内容字段名（必需） |
| context | 上下文字段名 |
| image | 图像字段名 |
| reference | 参考答案字段名 |

#### EvalPiplineConfig 配置 (evaluator[].evals[])

单个评估器配置：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| name | str | - | Yes | 评估器名称（Rule 或 LLM 类名） |
| config | object | null | No | 评估器配置参数 |

#### Rule 评估器配置 (config)

规则类评估器的配置参数：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| threshold | float | null | No | 规则决策阈值 |
| pattern | str | null | No | 匹配模式字符串 |
| key_list | list | null | No | 匹配关键词列表 |
| refer_path | list | null | No | 参考文件路径或模型路径 |
| parameters | object | null | No | 其他自定义参数 |

#### LLM 评估器配置 (config)

LLM 类评估器的配置参数：

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| model | str | null | No | 使用的模型名称 |
| key | str | null | Yes | API 密钥 |
| api_url | str | null | Yes | API URL |
| parameters | object | null | No | LLM 调参配置 |
| embedding_config | object | null | No | Embedding 模型配置 |

##### LLM Parameters 配置

LLM 调参配置：

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| temperature | number | 1 | 采样温度，0-2之间 |
| top_p | number | 1 | 核心采样概率 |
| max_tokens | number | 4000 | 最大生成token数 |
| presence_penalty | number | 0 | 存在惩罚，-2.0到2.0之间 |
| frequency_penalty | number | 0 | 频率惩罚，-2.0到2.0之间 |

## 配置文件示例

### 基础示例（仅使用规则评估器）

```json
{
  "task_name": "dingo",
  "input_path": "test/data/test_local_jsonl.jsonl",
  "output_path": "outputs/",
  "log_level": "WARNING",
  "use_browser": false,

  "dataset": {
    "source": "local",
    "format": "jsonl"
  },

  "executor": {
    "start_index": 0,
    "end_index": -1,
    "max_workers": 1,
    "batch_size": 1,
    "result_save": {
      "bad": true,
      "good": false
    }
  },

  "evaluator": [
    {
      "fields": {"content": "content"},
      "evals": [
        {"name": "RuleColonEnd"},
        {"name": "RuleAbnormalChar"}
      ]
    }
  ]
}
```

### 使用 LLM 评估器

```json
{
  "task_name": "llm_evaluation",
  "input_path": "test/data/test_local_jsonl.jsonl",
  "output_path": "outputs/",

  "dataset": {
    "source": "local",
    "format": "jsonl"
  },

  "executor": {
    "result_save": {
      "bad": true,
      "good": true
    }
  },

  "evaluator": [
    {
      "fields": {"content": "content"},
      "evals": [
        {"name": "LLMTextQualityV4", "config": {
          "model": "deepseek-chat",
          "key": "your-api-key",
          "api_url": "https://api.deepseek.com/v1"
        }}
      ]
    }
  ]
}
```

### 混合使用规则和 LLM 评估器

```json
{
  "task_name": "mixed_evaluation",
  "input_path": "test/data/test_local_jsonl.jsonl",

  "dataset": {
    "source": "local",
    "format": "jsonl"
  },

  "executor": {
    "max_workers": 4,
    "batch_size": 10,
    "result_save": {
      "bad": true,
      "good": true
    }
  },

  "evaluator": [
    {
      "fields": {"content": "content"},
      "evals": [
        {"name": "RuleColonEnd"},
        {"name": "RuleAbnormalChar"},
        {"name": "LLMTextQualityV4", "config": {
          "model": "deepseek-chat",
          "key": "your-api-key",
          "api_url": "https://api.deepseek.com/v1"
        }}
      ]
    }
  ]
}
```

### 多字段评估示例

```json
{
  "task_name": "multi_field_evaluation",
  "input_path": "path/to/your/data.jsonl",
  "dataset": {
    "source": "local",
    "format": "jsonl"
  },
  "evaluator": [
    {
      "fields": {"prompt": "question", "content": "answer", "context": "context"},
      "evals": [
        {"name": "LLMHallucination", "config": {
          "key": "your-api-key",
          "api_url": "https://api.openai.com/v1"
        }}
      ]
    }
  ]
}
```

## 使用方式

### CLI 方式
```bash
dingo --input config.json
```

### SDK 方式
```python
from dingo.config import InputArgs
from dingo.exec import Executor

# 从字典创建配置
input_data = {
    "task_name": "my_task",
    "input_path": "data.jsonl",
    "dataset": {
        "source": "local",
        "format": "jsonl"
    },
    "executor": {
        "result_save": {"bad": True, "good": True}
    },
    "evaluator": [
        {
            "fields": {"content": "content"},
            "evals": [
                {"name": "RuleColonEnd"}
            ]
        }
    ]
}

input_args = InputArgs(**input_data)
executor = Executor.exec_map["local"](input_args)
result = executor.execute()
print(result)
```

## 多轮对话模式

`Dingo` 支持多轮对话数据质检，如MT-Bench、MT-Bench++和MT-Bench101：

| Mode | Description |
|------|-------------|
| all | 拼接多轮对话中的所有回合 |

具体使用方法请参考相关示例文件:
[sdk_mtbench101_rule_all.py](../examples/multi_turn_dialogues/sdk_mtbench101_rule_all.py)、[sdk_mtbench101_llm.py](../examples/multi_turn_dialogues/sdk_mtbench101_llm.py)、
[sdk_mtbench_rule_all.py](../examples/multi_turn_dialogues/sdk_mtbench_rule_all.py)、[sdk_mtbench_llm.py](../examples/multi_turn_dialogues/sdk_mtbench_llm.py)。
