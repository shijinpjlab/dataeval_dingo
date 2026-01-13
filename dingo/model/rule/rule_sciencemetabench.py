import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from dingo.config.input_args import EvaluatorRuleArgs
from dingo.io.input import Data, RequiredField
from dingo.io.output.eval_detail import EvalDetail, QualityLabel
from dingo.model.model import Model
from dingo.model.rule.base import BaseRule


def string_similarity(str1, str2) -> float:
    """
    计算两个字符串的相似度
    
    规则:
    1. 一个为空另一个不为空，相似度为0
    2. 二者完全相同（包括全为空），相似度为1
    3. 忽略大小写进行比较
    4. 使用SequenceMatcher计算相似度
    """
    # 规则1: 一个为空另一个不为空，相似度为0
    if (str1 is None or str1 == "") and (str2 is not None and str2 != ""):
        return 0.0
    if (str2 is None or str2 == "") and (str1 is not None and str1 != ""):
        return 0.0

    # 规则2: 二者完全相同（包括全为空），相似度为1
    if (str1 or "") == (str2 or ""):
        return 1.0

    # 规则3: 忽略大小写进行比较
    s1_lower = str1.lower()
    s2_lower = str2.lower()

    # 如果忽略大小写后相同，直接返回1
    if s1_lower == s2_lower:
        return 1.0

    # 使用SequenceMatcher计算相似度
    matcher = SequenceMatcher(None, s1_lower, s2_lower)
    similarity = matcher.ratio()

    return similarity


class RuleMetadataMatchBase(BaseRule):
    """
    元数据字段相似度匹配的基类
    
    比较 benchmark 和 product 字段中的各个子字段的相似度
    阈值为 0.6，只有所有字段的相似度都达到阈值才算通过
    
    子类需要定义:
    - _metric_info: 包含 metric_name 和 description
    - dynamic_config: 包含 key_list (要检查的字段列表)
    """
    
    _required_fields = [RequiredField.BENCHMARK, RequiredField.PRODUCT]
    
    @classmethod
    def eval(cls, input_data: Data) -> EvalDetail:
        res = EvalDetail(metric=cls.__name__)
        
        # 检查并获取 benchmark 和 product 字段
        if not hasattr(input_data, RequiredField.BENCHMARK.value):
            raise ValueError(f"input_data 中缺少必需字段: {RequiredField.BENCHMARK.value}")
        
        if not hasattr(input_data, RequiredField.PRODUCT.value):
            raise ValueError(f"input_data 中缺少必需字段: {RequiredField.PRODUCT.value}")
        
        benchmark = getattr(input_data, RequiredField.BENCHMARK.value)
        product = getattr(input_data, RequiredField.PRODUCT.value)
        
        # 检查所有字段的相似度
        failed_fields = []
        similarity_dict = {}
        
        for field in cls.dynamic_config.key_list:
            benchmark_value = benchmark.get(field, "")
            product_value = product.get(field, "")
            
            similarity = string_similarity(
                str(benchmark_value) if benchmark_value is not None else "",
                str(product_value) if product_value is not None else ""
            )
            
            similarity_dict[field] = round(similarity, 3)
            
            if similarity < cls.dynamic_config.threshold:
                failed_fields.append(field)
        
        # 如果有任何字段未通过，则标记为失败
        if failed_fields:
            res.status = True
            res.label = [f"{cls.metric_type}.{cls.__name__}.{field}" for field in failed_fields]
        else:
            res.label = [QualityLabel.QUALITY_GOOD]
        res.reason = [{"similarity": similarity_dict}]
        
        return res


@Model.rule_register(
    "QUALITY_BAD_EFFECTIVENESS",
    ["sciencemetabench"],
)
class RuleMetadataMatchPaper(RuleMetadataMatchBase):
    """
    检查学术论文(Paper)元数据字段的相似度匹配
    
    比较 benchmark 和 product 字段中的各个子字段，包括:
    doi, title, author, keyword, abstract, pub_time
    
    阈值为 0.6，只有所有字段的相似度都达到阈值才算通过
    """
    
    _metric_info = {
        "category": "Rule-Based Metadata Quality Metrics",
        "quality_dimension": "EFFECTIVENESS",
        "metric_name": "RuleMetadataMatchPaper",
        "description": "检查学术论文元数据字段与基准数据的相似度匹配，阈值为0.6",
    }
    
    dynamic_config = EvaluatorRuleArgs(
        key_list=['doi', 'title', 'author', 'keyword', 'abstract', 'pub_time'],
        threshold=0.6
    )


@Model.rule_register(
    "QUALITY_BAD_EFFECTIVENESS",
    ["sciencemetabench"],
)
class RuleMetadataMatchEbook(RuleMetadataMatchBase):
    """
    检查电子书(Ebook)元数据字段的相似度匹配
    
    比较 benchmark 和 product 字段中的各个子字段，包括:
    isbn, title, author, abstract, category, pub_time, publisher
    
    阈值为 0.6，只有所有字段的相似度都达到阈值才算通过
    """
    
    _metric_info = {
        "category": "Rule-Based Metadata Quality Metrics",
        "quality_dimension": "EFFECTIVENESS",
        "metric_name": "RuleMetadataMatchEbook",
        "description": "检查电子书元数据字段与基准数据的相似度匹配，阈值为0.6",
    }
    
    dynamic_config = EvaluatorRuleArgs(
        key_list=['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher'],
        threshold=0.6
    )


@Model.rule_register(
    "QUALITY_BAD_EFFECTIVENESS",
    ["sciencemetabench"],
)
class RuleMetadataMatchTextbook(RuleMetadataMatchBase):
    """
    检查教科书(Textbook)元数据字段的相似度匹配
    
    比较 benchmark 和 product 字段中的各个子字段，包括:
    isbn, title, author, abstract, category, pub_time, publisher
    
    阈值为 0.6，只有所有字段的相似度都达到阈值才算通过
    """
    
    _metric_info = {
        "category": "Rule-Based Metadata Quality Metrics",
        "quality_dimension": "EFFECTIVENESS",
        "metric_name": "RuleMetadataMatchTextbook",
        "description": "检查教科书元数据字段与基准数据的相似度匹配，阈值为0.6",
    }
    
    dynamic_config = EvaluatorRuleArgs(
        key_list=['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher'],
        threshold=0.6
    )


def write_similarity_to_excel(type: str, output_dir: str,  output_filename: str = None):
    """
    将相似度分析数据写入Excel文件
    
    Args:
        output_dir: 输出目录路径，如 outputs/20260113_102321_d4c76b9e
        type: 数据类型，可选值: 'paper'(学术论文), 'ebook'(电子书), 'textbook'(教科书)
        output_filename: 输出Excel文件名，默认为带时间戳的 similarity_{type}_{时间戳}.xlsx
    
    Returns:
        pd.DataFrame: 生成的数据框
    
    Raises:
        ValueError: 当输出目录不存在、未找到jsonl文件或type不合法时抛出
    """
    # 定义不同类型的字段列表
    KEY_LISTS = {
        'paper': ['doi', 'title', 'author', 'keyword', 'abstract', 'pub_time'],
        'ebook': ['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher'],
        'textbook': ['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher'],
    }
    
    # 验证type
    if type not in KEY_LISTS:
        raise ValueError(f"不支持的数据类型: {type}，可选值为: {list(KEY_LISTS.keys())}")
    
    key_list = KEY_LISTS[type]
    
    # 生成默认文件名
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"similarity_{type}_{timestamp}.xlsx"
    
    # 读取output_dir下所有的.jsonl文件
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"输出目录不存在: {output_dir}")
    
    # 收集所有jsonl文件
    jsonl_files = list(output_path.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"在目录 {output_dir} 中未找到任何.jsonl文件")
    
    # 读取所有数据
    all_data = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_data.append(data)
    
    if not all_data:
        raise ValueError("未读取到任何数据")
    
    # 准备Excel数据
    rows = []
    for data in all_data:
        sha256 = str(data.get('sha256', ''))
        benchmark = data.get('benchmark', {})
        product = data.get('product', {})
        
        # 从dingo_result中提取相似度数据
        dingo_result = data.get('dingo_result', {})
        eval_details = dingo_result.get('eval_details', {})
        default_details = eval_details.get('default', [])
        
        # 获取相似度字典
        similarity_dict = {}
        if default_details and len(default_details) > 0:
            reason_list = default_details[0].get('reason', [])
            if reason_list and len(reason_list) > 0:
                similarity_dict = reason_list[0].get('similarity', {})
        
        # 构建行数据，所有值转换为字符串
        row = {'sha256': sha256}
        
        for field in key_list:
            # benchmark字段值 - 转为字符串
            benchmark_value = benchmark.get(field, '')
            row[f'benchmark_{field}'] = str(benchmark_value) if benchmark_value is not None else ''
            
            # product字段值 - 转为字符串
            product_value = product.get(field, '')
            row[f'product_{field}'] = str(product_value) if product_value is not None else ''
            
            # similarity值 - 转为字符串
            similarity_value = similarity_dict.get(field, '')
            row[f'similarity_{field}'] = str(similarity_value) if similarity_value != '' else ''
        
        rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 定义列的顺序
    column_order = ['sha256']
    for field in key_list:
        column_order.extend([f'benchmark_{field}', f'product_{field}', f'similarity_{field}'])
    
    # 重新排列列顺序
    df = df[column_order]
    
    # 确保所有列都是字符串类型
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # 按sha256升序排序
    df = df.sort_values(by='sha256', ascending=True).reset_index(drop=True)
    
    # 写入Excel文件
    output_file_path = output_path / output_filename
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='相似度分析', index=False)
    
    print(f"数据已成功写入 {output_file_path}")
    print(f"总共处理了 {len(rows)} 条记录")
    
    return df

