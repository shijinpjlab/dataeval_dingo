# Dingo: Executor的Local模式介绍

## 一、模块定位与作用

`dingo.exec.local` 主要实现了本地执行器（LocalExecutor），用于在本地环境下对数据集进行评测任务的自动化批量处理。它负责数据加载、模型配置、评测调度、结果统计与输出，是 Dingo 评测系统的核心执行组件之一。

---

## 二、核心类与结构

### 1. LocalExecutor

#### 继承关系

- 继承自 `ExecProto`，并通过 `@Executor.register("local")` 装饰器注册为 "local" 类型的执行器。

#### 主要属性

- `input_args`: 评测任务的输入参数（InputArgs 实例）。
- `llm`: 当前使用的大语言模型（BaseLLM 实例）。
- `summary`: 评测任务的统计与汇总信息（SummaryModel 实例）。

#### 主要方法

##### 1. 数据加载

- `load_data() -> Generator[Data]`
  - 根据输入参数中的数据集名称，动态加载数据源和数据集，返回数据生成器。

##### 2. 执行主流程

- `execute() -> SummaryModel`
  - 设置日志等级，应用模型配置，创建输出目录。
  - 根据配置选择 LLM，初始化 SummaryModel。
  - 调用 `evaluate()` 进行批量评测。
  - 汇总并写出 summary，返回最终统计结果。

##### 3. 评测主循环

- `evaluate()`
  - 支持多线程并发处理。
  - 按 batch_size 分批处理数据，调度评估管道（EvalPipline）下的评测任务。
  - 聚合每条数据的评测结果，实时更新 summary，并写出单条数据和 summary。

##### 4. 单条数据评测

- `evaluate_single_data(evaluator, data: Data) -> ResultInfo`
  - 根据 EvalPipline 配置，依次调用规则或 LLM 评估器。
  - 聚合每个评估器的评测结果，区分好坏类型、名称、原因。

- 评估器调用：
  - 规则评估器：直接调用规则的 `eval` 方法
  - LLM 评估器：调用 LLM 的 `eval` 方法（内置提示词）

##### 5. 结果写出与汇总

- `write_single_data(path, input_args, result_info)`
  - 按类型-名称分目录写出每条数据的评测结果（jsonl 格式），支持保存原始数据。

- `write_summary(path, input_args, summary)`
  - 写出当前评测任务的 summary（summary.json）。

- `summarize(summary: SummaryModel) -> SummaryModel`
  - 计算得分、类型/名称分布比例，补充完成时间。

##### 6. 结果查询

- `get_info_list(high_quality: bool) -> list`
  - 读取输出目录下所有结果，按 eval_status 区分高/低质量数据。

- `get_bad_info_list()`, `get_good_info_list()`
  - 分别获取低质量/高质量数据列表。

##### 7. 结果合并

- `merge_result_info(existing_list, new_item)`
  - 合并同一 data_id 的评测结果，去重类型、名称、原因。

---

## 三、执行流程（有序步骤）

1. **初始化 LocalExecutor**
   创建 LocalExecutor 实例，传入 InputArgs 参数。

2. **加载数据**
   调用 `load_data` 方法，根据输入参数加载数据集，返回数据生成器。

3. **应用模型配置**
   调用 `Model.apply_config`，根据配置文件和分组名应用评测规则、Prompt、LLM 等配置。

4. **创建输出目录**
   根据当前时间和 UUID 生成唯一输出目录，并在需要时创建。

5. **初始化 SummaryModel**
   创建 SummaryModel 实例，用于统计和汇总评测任务信息。

6. **批量评测**
   调用 `evaluate` 方法，按 batch_size 分批调度线程池，对每批数据进行评测。

7. **对每条数据进行评测**
   针对每条数据，按照 evaluator 配置中的 EvalPipline 依次调用评估器。

8. **聚合结果，写出单条数据**
   聚合每条数据的评测结果，写出到对应的输出文件。

9. **实时更新 summary**
    在评测过程中，实时更新 SummaryModel 的统计信息。

10. **写出 summary.json**
    评测结束后，将 summary 信息写出为 summary.json 文件。

11. **返回 SummaryModel**
    返回最终的 SummaryModel 结果，供后续分析或展示使用。

---

## 四、设计亮点

- **高并发支持**：支持线程池并发处理，兼容本地多核部署。
- **评估管道**：支持 evaluator 数组配置，灵活组合规则和 LLM 评估器。
- **动态模型配置**：与 Model 配合，支持按配置文件动态切换评测规则、LLM。
- **结果结构化输出**：单条数据与 summary 分别输出，便于后续分析与复现。
- **高/低质量数据筛选**：内置高低质量数据快速检索接口。

---

## 五、注意事项

- 需保证输入参数（InputArgs）和配置文件格式正确。
- 评测规则、LLM 需提前注册并实现对应接口。
- 输出目录需有写权限，且不会与历史任务冲突。

---

## 六、典型用法

```python
from dingo.config import InputArgs
from dingo.exec import Executor

input_data = {
    "input_path": "test/data/test_local_jsonl.jsonl",
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
            "fields": {"content": "content"},
            "evals": [
                {"name": "RuleColonEnd"},
                {"name": "RuleAbnormalChar"}
            ]
        }
    ]
}

input_args = InputArgs(**input_data)
executor = Executor.exec_map["local"](input_args)
result = executor.execute()
print(result)
```

---

## 七、总结

`dingo.exec.local` 是 Dingo 评测系统的本地执行核心，具备高并发、灵活评估管道配置、动态配置、结构化输出等特性，适合大规模自动化评测任务。其设计充分考虑了扩展性与易用性，是构建智能评测流水线的重要基础模块。
