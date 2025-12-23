import pytest

from dingo.config import InputArgs
from dingo.data.datasource.huggingface import HuggingFaceSource


class TestHfDataset:
    def test_hf_datasource_get_data(self):
        path = "chupei/format-text"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "plaintext"},
            evaluator=[{"fields": {"content": "text"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        data_iter = source.load()
        for i in data_iter:
            print(i)

    def test_hf_datasource_get_data_2(self):
        path = "chupei/format-json"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "json"},
            evaluator=[{"fields": {"content": "prediction", "prompt": "origin_prompt"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        data_iter = source.load()
        for i in data_iter:
            print(i)

    def test_hf_datasource_get_data_3(self):
        path = "chupei/format-jsonl"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "jsonl"},
            evaluator=[{"fields": {"content": "content"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        data_iter = source.load()
        for i in data_iter:
            print(i)

    def test_hf_datasource_get_data_4(self):
        path = "chupei/format-listjson"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "listjson"},
            evaluator=[{"fields": {"content": "output", "prompt": "instruction"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        data_iter = source.load()
        for i in data_iter:
            print(i)

    def test_hf_datasource_get_data_5(self):
        path = "lmms-lab/LLaVA-OneVision-Data"
        ri = InputArgs(
            input_path=path,
            output_path='./test/outputs/',
            dataset={"source": "hugging_face", "format": "hf-image"},
            evaluator=[{"fields": {"image": ["image"], "content": "conversations"}, "evals": [{"name": "RuleAspectRatio"}, {"name": "RuleImageSize"}]}]
        )
        source = HuggingFaceSource(input_args=ri, config_name='CLEVR-Math(MathV360K)')
        data_iter = source.load()
        print(data_iter[0])


if __name__ == "__main__":
    pytest.main(["-s", "-q"])
