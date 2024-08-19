import os

from datasets import load_dataset, DatasetDict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "/Users/shixiufeng/Code/github/llm-fine-tuning/hf"

language_abbr = "zh-CN"
dataset_name = "mozilla-foundation/common_voice_11_0"

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)
