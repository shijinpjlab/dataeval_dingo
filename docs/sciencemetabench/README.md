# ScienceMetaBench Usage Guide

[English](README.md) | [中文](README_ZH.md)

## 📖 Introduction

ScienceMetaBench is a benchmark dataset for evaluating the accuracy of metadata extraction from scientific literature PDFs. This document introduces how to use the Dingo framework to assess the quality of extracted metadata.

**Dataset**: 🤗 [HuggingFace - ScienceMetaBench](https://huggingface.co/datasets/opendatalab/ScienceMetaBench)

### Supported Data Types

- **Academic Papers (Paper)**: Primarily from academic journals and conference papers
- **Textbooks (Textbook)**: Formally published textbooks
- **Ebooks (Ebook)**: Digitized historical documents and books

## 🚀 Quick Start

### 1. Install Dependencies

#### Install Dingo

**Option 1: Install Specific Version**

```bash
pip install dingo-python
```

**Option 2: Development Mode Installation (Recommended for Local Development)**

```bash
git clone https://github.com/MigoXLab/dingo.git
cd dingo
git checkout dev
pip install -e .
```


### 2. Prepare Data

Data format is JSONL, with each record containing the following fields:

#### Field Description

- **`sha256`**: Unique identifier for tracing source files

- **`benchmark`**: Ground truth (standard answer)
  - **Source**: Obtained from [ScienceMetaBench Dataset](https://huggingface.co/datasets/opendatalab/ScienceMetaBench)
  - **Included Fields**：
    - Academic Papers (Paper): `doi`, `title`, `author`, `keyword`, `abstract`, `pub_time`
    - Textbooks/Ebooks (Textbook/Ebook): `isbn`, `title`, `author`, `abstract`, `category`, `pub_time`, `publisher`

- **`product`**: Extraction results to be evaluated
  - **Source**: Extracted from PDF
  - **Included Fields**: Same fields as `benchmark` above

### 3. Run Evaluation

For complete example code, please refer to: `examples/sciencemetabench/paper.py`

**Configuration Parameters**:

- `input_path`: Input JSONL file path
- `dataset.source`: Data source type, use `"local"` for local files
- `dataset.format`: Data format, use `"jsonl"`
- `executor.result_save.merge`: Whether to save all results in one file
- `evaluator.evals.name`: Evaluation rule name, three dedicated evaluation rules are provided:
  - `RuleMetadataMatchPaper`: Academic paper evaluation rule
  - `RuleMetadataMatchEbook`: Ebook evaluation rule
  - `RuleMetadataMatchTextbook`: Textbook evaluation rule
- `evaluator.evals.config.threshold`: Similarity threshold (0-1), default 0.6

### 4. Run Script

```bash
python examples/sciencemetabench/paper.py
```

## 📊 Evaluation Rules

### Similarity Calculation Rules

All rules use a string similarity algorithm based on `SequenceMatcher`:

1. **Null Value Handling**: One is empty while the other is not → similarity is 0
2. **Exact Match**: Both are identical (including both empty) → similarity is 1
3. **Case Insensitive**: Converted to lowercase before comparison
4. **Sequence Matching**: Uses longest common subsequence algorithm to calculate similarity (range: 0-1)

**Similarity Score Interpretation**:
- `1.0`: Exact match
- `0.8-0.99`: Highly similar (may have slight formatting differences)
- `0.5-0.79`: Partial match (extracted main information but incomplete)
- `0.0-0.49`: Low similarity (significant difference between extraction result and ground truth)

### Evaluation Results

Each sample's evaluation result includes:

- `eval_status`: Evaluation status
- `eval_details`: Detailed evaluation information
  - `metric`: Name of the evaluation rule used
  - `status`: Whether any field failed to meet the threshold
  - `label`: List of fields that didn't meet the threshold (if any)
  - `reason`: Contains similarity scores for all fields

**Example Output**:
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

## 📈 Result Export and Analysis

### Using write_similarity_to_excel Function

After evaluation is complete, you can use the built-in function to export results to an Excel file:

```python
from dingo.model.rule.rule_sciencemetabench import write_similarity_to_excel

# Export academic paper evaluation results
write_similarity_to_excel(
    type='paper',                    # Data type: 'paper', 'ebook', 'textbook'
    output_dir='outputs/xxx',        # Output directory path
    output_filename='custom.xlsx'    # Optional, custom filename
)
```

**Parameters**:
- `type`: Data type, must be `'paper'`, `'ebook'`, or `'textbook'`
- `output_dir`: Directory containing evaluation result JSONL file
- `output_filename`: (Optional) Custom output filename, defaults to `similarity_{type}_{timestamp}.xlsx`

### Excel Output Format

The generated Excel file contains two worksheets (sheets):

#### Sheet 1: Similarity Analysis

Detailed data with the following columns:

```
sha256 | benchmark_field1 | product_field1 | similarity_field1 | benchmark_field2 | product_field2 | similarity_field2 | ...
```

**Notes**:
- All data sorted by `sha256` in ascending order
- Each field contains three columns:
  - `benchmark_{field}`: Ground truth
  - `product_{field}`: Extraction result
  - `similarity_{field}`: Similarity score (string format)
- All cell contents are string type

#### Sheet 2: Summary Statistics

Aggregated data containing:

| Field | Average Similarity |
|-------|-------------------|
| doi | 0.6667 |
| title | 0.8730 |
| ... | ... |
| **Overall Accuracy** | **0.6719** |

**Metric Explanation**:
- **Field-level Accuracy**: Average similarity for each field = Σ(similarity of all samples for that field) / total number of samples
- **Overall Accuracy**: Average of all field accuracies = Σ(accuracy of each field) / total number of fields

Summary statistics are automatically calculated and saved in the second worksheet, no manual calculation needed.
