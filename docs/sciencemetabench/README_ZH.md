# ScienceMetaBench 使用指南

[English](README.md) | [中文](README_ZH.md)

## 📖 简介

ScienceMetaBench 是一个用于评估科学文献 PDF 元数据提取准确性的基准测试数据集。本文档介绍如何使用 Dingo 框架对提取的元数据进行质量评估。

**数据集地址**: 🤗 [HuggingFace - ScienceMetaBench](https://huggingface.co/datasets/opendatalab/ScienceMetaBench)

### 支持的数据类型

- **学术论文 (Paper)**: 主要来自学术期刊和会议论文
- **教科书 (Textbook)**: 正式出版的教科书
- **电子书 (Ebook)**: 数字化的历史文献和书籍

## 🚀 快速开始

### 1. 安装依赖

#### 安装 Dingo

**方式一：安装指定版本**

```bash
pip install dingo-python
```

**方式二：开发模式安装（推荐用于本地开发）**

```bash
git clone https://github.com/MigoXLab/dingo.git
cd dingo
git checkout dev
pip install -e .
```


### 2. 准备数据

数据格式为 JSONL，每条记录包含以下字段：

#### 字段说明

- **`sha256`**: 唯一标识符，用于追溯源文件

- **`benchmark`**: 标准答案（ground truth）
  - **来源**: 从 [ScienceMetaBench 数据集](https://huggingface.co/datasets/opendatalab/ScienceMetaBench)获取
  - **包含字段**：
    - 学术论文 (Paper): `doi`, `title`, `author`, `keyword`, `abstract`, `pub_time`
    - 教科书/电子书 (Textbook/Ebook): `isbn`, `title`, `author`, `abstract`, `category`, `pub_time`, `publisher`

- **`product`**: 待评估的提取结果
  - **来源**: 从 PDF 中提取
  - **包含字段**: 与上面 `benchmark` 对应，字段相同

### 3. 运行评估

完整的示例代码请参考：`examples/sciencemetabench/paper.py`

**配置参数说明**：

- `input_path`: 输入 JSONL 文件路径
- `dataset.source`: 数据源类型，本地文件使用 `"local"`
- `dataset.format`: 数据格式，使用 `"jsonl"`
- `executor.result_save.merge`: 是否将所有结果保存在一个文件中
- `evaluator.evals.name`: 评估规则名称，提供了三个专用的评估规则
  - `RuleMetadataMatchPaper`: 学术论文评估规则
  - `RuleMetadataMatchEbook`: 电子书评估规则
  - `RuleMetadataMatchTextbook`: 教科书评估规则
- `evaluator.evals.config.threshold`: 相似度阈值（0-1），默认 0.6

### 4. 运行脚本

```bash
python examples/sciencemetabench/paper.py
```

## 📊 评估规则说明

### 相似度计算规则

所有规则使用基于 `SequenceMatcher` 的字符串相似度算法：

1. **空值处理**: 一个为空另一个不为空 → 相似度为 0
2. **完全匹配**: 二者完全相同（包括全为空）→ 相似度为 1
3. **忽略大小写**: 转换为小写后进行比较
4. **序列匹配**: 使用最长公共子序列算法计算相似度（范围: 0-1）

**相似度分数解读**：
- `1.0`: 完全匹配
- `0.8-0.99`: 高度相似（可能有轻微格式差异）
- `0.5-0.79`: 部分匹配（提取了主要信息但不完整）
- `0.0-0.49`: 相似度低（提取结果与标准答案差异较大）

### 评估结果

每个样本的评估结果包含：

- `eval_status`: 评估状态
- `eval_details`: 详细评估信息
  - `metric`: 使用的评估规则名称
  - `status`: 是否有字段未达到阈值
  - `label`: 未达到阈值的字段列表（如果有）
  - `reason`: 包含所有字段的相似度分数

**示例输出**：
```json
{
  "sha256": "7d05cfd0101f9443...",
  "dingo_result": {
    "eval_status": true,
    "eval_details": {
      "default": [
        {
          "metric": "RuleMetadataMatchPaper",
          "status": true,
          "label": ["QUALITY_BAD_EFFECTIVENESS.RuleMetadataMatchPaper.abstract"],
          "reason": [
            {
              "similarity": {
                "doi": 1.0,
                "title": 0.941,
                "author": 1.0,
                "keyword": 0.977,
                "abstract": 0.488,
                "pub_time": 1.0
              }
            }
          ]
        }
      ]
    }
  }
}
```

## 📈 结果导出与分析

### 使用 write_similarity_to_excel 函数

评估完成后，可以使用内置函数将结果导出为 Excel 文件：

```python
from dingo.model.rule.rule_sciencemetabench import write_similarity_to_excel

# 导出学术论文评估结果
write_similarity_to_excel(
    type='paper',                    # 数据类型: 'paper', 'ebook', 'textbook'
    output_dir='outputs/xxx',        # 输出目录路径
    output_filename='custom.xlsx'    # 可选，自定义文件名
)
```

**参数说明**：
- `type`: 数据类型，必须是 `'paper'`、`'ebook'` 或 `'textbook'`
- `output_dir`: 包含评估结果 JSONL 文件的目录
- `output_filename`: （可选）自定义输出文件名，默认为 `similarity_{type}_{timestamp}.xlsx`

### Excel 输出格式

生成的 Excel 文件包含两个工作表（sheet）：

#### Sheet 1: 相似度分析

详细数据，包含以下列：

```
sha256 | benchmark_字段1 | product_字段1 | similarity_字段1 | benchmark_字段2 | product_字段2 | similarity_字段2 | ...
```

**说明**：
- 所有数据按 `sha256` 升序排序
- 每个字段包含三列：
  - `benchmark_{field}`: 标准答案
  - `product_{field}`: 提取结果
  - `similarity_{field}`: 相似度分数（字符串格式）
- 所有单元格内容均为字符串类型

#### Sheet 2: 汇总统计

汇总数据，包含：

| 字段 | 平均相似度 |
|------|-----------|
| doi | 0.6667 |
| title | 0.8730 |
| ... | ... |
| **总体准确率** | **0.6719** |

**指标说明**：
- **字段级准确率**：每个字段的平均相似度 = Σ(该字段所有样本的相似度) / 样本总数
- **总体准确率**：所有字段准确率的平均值 = Σ(各字段准确率) / 字段总数

汇总统计数据会自动计算并保存在第二个工作表中，无需手动计算。
