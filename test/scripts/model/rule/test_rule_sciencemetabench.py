"""
单元测试: dingo/model/rule/rule_sciencemetabench.py

测试 ScienceMetaBench 相关规则和功能函数
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from dingo.io.input import Data
from dingo.model.rule.rule_sciencemetabench import RuleMetadataMatchEbook, RuleMetadataMatchPaper, RuleMetadataMatchTextbook, string_similarity, write_similarity_to_excel


class TestStringSimilarity:
    """测试 string_similarity 函数"""

    def test_both_empty(self):
        """测试两个字符串都为空的情况"""
        assert string_similarity("", "") == 1.0
        assert string_similarity(None, None) == 1.0
        assert string_similarity("", None) == 1.0
        assert string_similarity(None, "") == 1.0

    def test_one_empty(self):
        """测试一个为空另一个不为空的情况"""
        assert string_similarity("", "hello") == 0.0
        assert string_similarity("hello", "") == 0.0
        assert string_similarity(None, "hello") == 0.0
        assert string_similarity("hello", None) == 0.0

    def test_exact_match(self):
        """测试完全匹配的情况"""
        assert string_similarity("hello", "hello") == 1.0
        assert string_similarity("测试", "测试") == 1.0

    def test_case_insensitive(self):
        """测试忽略大小写"""
        assert string_similarity("Hello", "hello") == 1.0
        assert string_similarity("WORLD", "world") == 1.0
        assert string_similarity("TeSt", "tEsT") == 1.0

    def test_similar_strings(self):
        """测试相似字符串"""
        # 相似度应该大于0但小于1
        sim = string_similarity("海底管道砂袋堆叠防护效果数值模拟", "海底管道砂袋堆叠防护效果数值模拟研究")
        assert 0.8 < sim < 1.0

        sim = string_similarity("Hello World", "Hello")
        assert 0.5 < sim < 1.0

    def test_completely_different(self):
        """测试完全不同的字符串"""
        sim = string_similarity("abc", "xyz")
        assert 0.0 <= sim < 0.5


class TestRuleMetadataMatchPaper:
    """测试 RuleMetadataMatchPaper 规则"""

    @pytest.fixture
    def test_data_file(self):
        """返回测试数据文件路径"""
        return Path(__file__).parent.parent.parent.parent / "data" / "sciencemetabench" / "paper.jsonl"

    def test_perfect_match(self):
        """测试完全匹配的情况"""
        data = Data()
        data.benchmark = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "author": "John Doe",
            "keyword": "test, paper",
            "abstract": "This is a test abstract",
            "pub_time": "2024"
        }
        data.product = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "author": "John Doe",
            "keyword": "test, paper",
            "abstract": "This is a test abstract",
            "pub_time": "2024"
        }

        result = RuleMetadataMatchPaper.eval(data)

        assert result.metric == "RuleMetadataMatchPaper"
        assert result.label == ["QUALITY_GOOD"]
        assert result.reason[0]["similarity"]["doi"] == 1.0
        assert result.reason[0]["similarity"]["title"] == 1.0

    def test_partial_match(self):
        """测试部分匹配的情况"""
        data = Data()
        data.benchmark = {
            "doi": "10.1234/test",
            "title": "Test Paper Title",
            "author": "John Doe",
            "keyword": "test, paper",
            "abstract": "This is a long test abstract with many details",
            "pub_time": "2024"
        }
        data.product = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "author": "John Doe",
            "keyword": "test, paper",
            "abstract": "This is a test abstract",
            "pub_time": "2024"
        }

        result = RuleMetadataMatchPaper.eval(data)

        assert result.metric == "RuleMetadataMatchPaper"
        # abstract 的相似度应该较低
        assert result.reason[0]["similarity"]["abstract"] < 1.0

    def test_below_threshold(self):
        """测试低于阈值的情况"""
        data = Data()
        data.benchmark = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "author": "John Doe",
            "keyword": "test, paper",
            "abstract": "This is a test abstract",
            "pub_time": "2024"
        }
        data.product = {
            "doi": "",
            "title": "Different Title",
            "author": "Jane Smith",
            "keyword": "different",
            "abstract": "Completely different",
            "pub_time": "2020"
        }

        result = RuleMetadataMatchPaper.eval(data)

        assert result.metric == "RuleMetadataMatchPaper"
        assert result.status is True
        assert len(result.label) > 0
        # 检查失败的字段是否被标记
        failed_fields = [label.split('.')[-1] for label in result.label if label != "QUALITY_GOOD"]
        assert len(failed_fields) > 0

    def test_with_real_data(self, test_data_file):
        """使用真实测试数据进行测试"""
        if not test_data_file.exists():
            pytest.skip(f"测试数据文件不存在: {test_data_file}")

        with open(test_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_data = json.loads(line)

                    data = Data()
                    data.sha256 = json_data["sha256"]
                    data.benchmark = json_data["benchmark"]
                    data.product = json_data["product"]

                    result = RuleMetadataMatchPaper.eval(data)

                    # 验证结果结构
                    assert result.metric == "RuleMetadataMatchPaper"
                    assert result.reason is not None
                    assert len(result.reason) > 0
                    assert "similarity" in result.reason[0]

                    # 验证所有字段都有相似度
                    similarity_dict = result.reason[0]["similarity"]
                    for field in ['doi', 'title', 'author', 'keyword', 'abstract', 'pub_time']:
                        assert field in similarity_dict
                        assert 0.0 <= similarity_dict[field] <= 1.0

    def test_missing_benchmark(self):
        """测试缺少 benchmark 字段的情况"""
        data = Data()
        data.product = {"title": "Test"}

        with pytest.raises(ValueError, match="缺少必需字段: benchmark"):
            RuleMetadataMatchPaper.eval(data)

    def test_missing_product(self):
        """测试缺少 product 字段的情况"""
        data = Data()
        data.benchmark = {"title": "Test"}

        with pytest.raises(ValueError, match="缺少必需字段: product"):
            RuleMetadataMatchPaper.eval(data)


class TestRuleMetadataMatchEbook:
    """测试 RuleMetadataMatchEbook 规则"""

    @pytest.fixture
    def test_data_file(self):
        """返回测试数据文件路径"""
        return Path(__file__).parent.parent.parent.parent / "data" / "sciencemetabench" / "ebook.jsonl"

    def test_perfect_match(self):
        """测试完全匹配的情况"""
        data = Data()
        data.benchmark = {
            "isbn": "9787030336613",
            "title": "Test Ebook",
            "author": "Author Name",
            "abstract": "Test abstract",
            "category": "Q949",
            "pub_time": "2024",
            "publisher": "Test Publisher"
        }
        data.product = {
            "isbn": "9787030336613",
            "title": "Test Ebook",
            "author": "Author Name",
            "abstract": "Test abstract",
            "category": "Q949",
            "pub_time": "2024",
            "publisher": "Test Publisher"
        }

        result = RuleMetadataMatchEbook.eval(data)

        assert result.metric == "RuleMetadataMatchEbook"
        assert result.label == ["QUALITY_GOOD"]
        assert result.reason[0]["similarity"]["isbn"] == 1.0
        assert result.reason[0]["similarity"]["title"] == 1.0

    def test_with_real_data(self, test_data_file):
        """使用真实测试数据进行测试"""
        if not test_data_file.exists():
            pytest.skip(f"测试数据文件不存在: {test_data_file}")

        with open(test_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_data = json.loads(line)

                    data = Data()
                    data.sha256 = json_data["sha256"]
                    data.benchmark = json_data["benchmark"]
                    data.product = json_data["product"]

                    result = RuleMetadataMatchEbook.eval(data)

                    # 验证结果结构
                    assert result.metric == "RuleMetadataMatchEbook"
                    assert result.reason is not None
                    assert len(result.reason) > 0
                    assert "similarity" in result.reason[0]

                    # 验证所有字段都有相似度
                    similarity_dict = result.reason[0]["similarity"]
                    for field in ['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher']:
                        assert field in similarity_dict
                        assert 0.0 <= similarity_dict[field] <= 1.0


class TestRuleMetadataMatchTextbook:
    """测试 RuleMetadataMatchTextbook 规则"""

    @pytest.fixture
    def test_data_file(self):
        """返回测试数据文件路径"""
        return Path(__file__).parent.parent.parent.parent / "data" / "sciencemetabench" / "textbook.jsonl"

    def test_perfect_match(self):
        """测试完全匹配的情况"""
        data = Data()
        data.benchmark = {
            "isbn": "9787030336613",
            "title": "Test Textbook",
            "author": "Author Name",
            "abstract": "Test abstract",
            "category": "Education",
            "pub_time": "2024",
            "publisher": "Test Publisher"
        }
        data.product = {
            "isbn": "9787030336613",
            "title": "Test Textbook",
            "author": "Author Name",
            "abstract": "Test abstract",
            "category": "Education",
            "pub_time": "2024",
            "publisher": "Test Publisher"
        }

        result = RuleMetadataMatchTextbook.eval(data)

        assert result.metric == "RuleMetadataMatchTextbook"
        assert result.label == ["QUALITY_GOOD"]
        assert result.reason[0]["similarity"]["isbn"] == 1.0
        assert result.reason[0]["similarity"]["title"] == 1.0

    def test_with_real_data(self, test_data_file):
        """使用真实测试数据进行测试"""
        if not test_data_file.exists():
            pytest.skip(f"测试数据文件不存在: {test_data_file}")

        with open(test_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_data = json.loads(line)

                    data = Data()
                    data.sha256 = json_data["sha256"]
                    data.benchmark = json_data["benchmark"]
                    data.product = json_data["product"]

                    result = RuleMetadataMatchTextbook.eval(data)

                    # 验证结果结构
                    assert result.metric == "RuleMetadataMatchTextbook"
                    assert result.reason is not None
                    assert len(result.reason) > 0
                    assert "similarity" in result.reason[0]

                    # 验证所有字段都有相似度
                    similarity_dict = result.reason[0]["similarity"]
                    for field in ['isbn', 'title', 'author', 'abstract', 'category', 'pub_time', 'publisher']:
                        assert field in similarity_dict
                        assert 0.0 <= similarity_dict[field] <= 1.0


class TestWriteSimilarityToExcel:
    """测试 write_similarity_to_excel 函数"""

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_paper_data(self, temp_output_dir):
        """创建示例 paper 数据"""
        data = [
            {
                "sha256": "test001",
                "benchmark": {
                    "doi": "10.1234/test001",
                    "title": "Test Paper 1",
                    "author": "Author 1",
                    "keyword": "keyword1",
                    "abstract": "Abstract 1",
                    "pub_time": "2024"
                },
                "product": {
                    "doi": "10.1234/test001",
                    "title": "Test Paper 1",
                    "author": "Author 1",
                    "keyword": "keyword1",
                    "abstract": "Abstract 1",
                    "pub_time": "2024"
                },
                "dingo_result": {
                    "eval_status": True,
                    "eval_details": {
                        "default": [
                            {
                                "metric": "RuleMetadataMatchPaper",
                                "status": True,
                                "label": ["QUALITY_GOOD"],
                                "reason": [
                                    {
                                        "similarity": {
                                            "doi": 1.0,
                                            "title": 1.0,
                                            "author": 1.0,
                                            "keyword": 1.0,
                                            "abstract": 1.0,
                                            "pub_time": 1.0
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            {
                "sha256": "test002",
                "benchmark": {
                    "doi": "10.1234/test002",
                    "title": "Test Paper 2",
                    "author": "Author 2",
                    "keyword": "keyword2",
                    "abstract": "Abstract 2",
                    "pub_time": "2024"
                },
                "product": {
                    "doi": "",
                    "title": "Different Title",
                    "author": "Author 2",
                    "keyword": "keyword2",
                    "abstract": "Different Abstract",
                    "pub_time": "2024"
                },
                "dingo_result": {
                    "eval_status": True,
                    "eval_details": {
                        "default": [
                            {
                                "metric": "RuleMetadataMatchPaper",
                                "status": True,
                                "label": ["QUALITY_BAD_EFFECTIVENESS.RuleMetadataMatchPaper.doi"],
                                "reason": [
                                    {
                                        "similarity": {
                                            "doi": 0.0,
                                            "title": 0.5,
                                            "author": 1.0,
                                            "keyword": 1.0,
                                            "abstract": 0.45,
                                            "pub_time": 1.0
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        ]

        # 写入 jsonl 文件
        jsonl_file = Path(temp_output_dir) / "test_result.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return temp_output_dir

    def test_write_paper_excel(self, sample_paper_data):
        """测试导出 paper 类型的 Excel"""
        output_filename = "test_paper.xlsx"

        df = write_similarity_to_excel(
            type='paper',
            output_dir=sample_paper_data,
            output_filename=output_filename
        )

        # 验证返回的 DataFrame
        assert df is not None
        assert len(df) == 2
        assert 'sha256' in df.columns

        # 验证所有 paper 字段都存在
        for field in ['doi', 'title', 'author', 'keyword', 'abstract', 'pub_time']:
            assert f'benchmark_{field}' in df.columns
            assert f'product_{field}' in df.columns
            assert f'similarity_{field}' in df.columns

        # 验证 Excel 文件是否创建
        excel_file = Path(sample_paper_data) / output_filename
        assert excel_file.exists()

        # 读取 Excel 验证内容
        df_from_excel = pd.read_excel(excel_file, sheet_name='相似度分析')
        assert len(df_from_excel) == 2

        # 验证汇总统计表
        df_summary = pd.read_excel(excel_file, sheet_name='汇总统计')
        assert len(df_summary) == 7  # 6个字段 + 1个总体准确率
        assert '字段' in df_summary.columns
        assert '平均相似度' in df_summary.columns
        assert df_summary.iloc[-1]['字段'] == '总体准确率'

    def test_invalid_type(self, temp_output_dir):
        """测试无效的数据类型"""
        with pytest.raises(ValueError, match="不支持的数据类型"):
            write_similarity_to_excel(
                type='invalid_type',
                output_dir=temp_output_dir
            )

    def test_nonexistent_directory(self):
        """测试不存在的目录"""
        with pytest.raises(ValueError, match="输出目录不存在"):
            write_similarity_to_excel(
                type='paper',
                output_dir='/nonexistent/directory'
            )

    def test_no_jsonl_files(self, temp_output_dir):
        """测试目录中没有 jsonl 文件"""
        with pytest.raises(ValueError, match="未找到任何.jsonl文件"):
            write_similarity_to_excel(
                type='paper',
                output_dir=temp_output_dir
            )

    def test_default_filename(self, sample_paper_data):
        """测试默认文件名生成"""
        write_similarity_to_excel(
            type='paper',
            output_dir=sample_paper_data
        )

        # 查找生成的文件
        output_path = Path(sample_paper_data)
        excel_files = list(output_path.glob("similarity_paper_*.xlsx"))
        assert len(excel_files) > 0

    def test_data_sorting(self, sample_paper_data):
        """测试数据按 sha256 排序"""
        df = write_similarity_to_excel(
            type='paper',
            output_dir=sample_paper_data,
            output_filename="test_sorted.xlsx"
        )

        # 验证排序
        sha256_list = df['sha256'].tolist()
        assert sha256_list == sorted(sha256_list)

    def test_all_string_type(self, sample_paper_data):
        """测试所有列都是字符串类型"""
        df = write_similarity_to_excel(
            type='paper',
            output_dir=sample_paper_data,
            output_filename="test_types.xlsx"
        )

        # 验证所有列都是字符串类型
        for col in df.columns:
            assert df[col].dtype == 'object'  # pandas 中字符串类型显示为 object


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
