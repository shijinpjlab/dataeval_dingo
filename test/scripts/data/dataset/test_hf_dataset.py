import pytest

from dingo.config import InputArgs
from dingo.data.dataset.huggingface import HuggingFaceDataset
from dingo.data.datasource.huggingface import HuggingFaceSource


class TestHfDataset:
    def test_hf_dataset_get_data(self):
        path = "chupei/format-text"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "plaintext"},
            evaluator=[{"fields": {"content": "text"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="chupei_text")
        data_iter = dataset.get_data()
        for i in data_iter:
            print(i)

    def test_hf_dataset_get_data_1(self):
        path = "chupei/format-json"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "json"},
            evaluator=[{"fields": {"content": "prediction", "prompt": "origin_prompt"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="chupei_json")
        data_iter = dataset.get_data()
        for i in data_iter:
            print(i)

    def test_hf_dataset_get_data_2(self):
        path = "chupei/format-jsonl"
        ri = InputArgs(
            input_path=path,
            output_path='data/outputs/',
            dataset={"source": "hugging_face", "format": "jsonl"},
            evaluator=[{"fields": {"content": "content"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="chupei_jsonl")
        data_iter = dataset.get_data()
        for i in data_iter:
            print(i)

    def test_hf_dataset_get_data_3(self):
        path = "chupei/format-listjson"
        ri = InputArgs(
            input_path=path,
            output_path='./test/outputs/',
            dataset={"source": "hugging_face", "format": "listjson"},
            evaluator=[{"fields": {"content": "output", "prompt": "instruction"}, "evals": [{"name": "RuleColonEnd"}, {"name": "RuleContentNull"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="chupei_listjson")
        data_iter = dataset.get_data()
        for i in data_iter:
            print(i)
            break

    @pytest.mark.skip(reason="Large dataset download required, run manually with: pytest -k test_hf_dataset_get_data_4 --run-slow")
    def test_hf_dataset_get_data_4(self):
        path = "lmms-lab/LLaVA-OneVision-Data"
        ri = InputArgs(
            input_path=path,
            output_path='./test/outputs/',
            dataset={"source": "hugging_face", "format": "hf-image", "hf_config": {"huggingface_config_name": "CLEVR-Math(MathV360K)"}},
            evaluator=[{"fields": {"image": ["image"]}, "evals": [{"name": "RuleAspectRatio"}, {"name": "RuleImageSize"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="LLaVA-OneVision-Data")
        data_iter = dataset.get_data()
        first_ele = next(data_iter)
        print(first_ele)

    @pytest.mark.skip(reason="Large dataset download required, run manually with: pytest -k test_hf_dataset_get_data_5 --run-slow")
    def test_hf_dataset_get_data_5(self):
        path = "HuggingFaceM4/Docmatix"
        ri = InputArgs(
            input_path=path,
            output_path='./test/outputs/',
            dataset={"source": "hugging_face", "format": "hf-image", "hf_config": {"huggingface_split": "test", "huggingface_config_name": "zero-shot-exp"}},
            evaluator=[{"fields": {"image": ["images"]}, "evals": [{"name": "RuleAspectRatio"}, {"name": "RuleImageSize"}]}]
        )
        source = HuggingFaceSource(input_args=ri)
        dataset: HuggingFaceDataset = HuggingFaceDataset(source=source, name="Docmatix")
        data_iter = dataset.get_data()
        first_ele = next(data_iter)
        print(first_ele)


if __name__ == "__main__":
    pytest.main(["-s", "-q"])
