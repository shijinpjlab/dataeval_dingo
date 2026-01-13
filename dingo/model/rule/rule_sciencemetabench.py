import json
from difflib import SequenceMatcher

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

