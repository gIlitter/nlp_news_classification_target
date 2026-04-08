# setup_models.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


def download_pretrained_models():
    """
    自动下载必要的预训练模型到本地 01_data/bert-base-chinese 目录
    """
    cache_dir = "./01_data/bert-base-chinese"
    model_name = "bert-base-chinese"

    print(f"Checking for {model_name} in {cache_dir}...")

    # 检查是否已存在 config.json (判断是否下载过)
    if not os.path.exists(os.path.join(cache_dir, "config.json")):
        print("Downloading pretrained model weights...")
        # 使用 transformers 库自动下载并保存到指定路径
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True  # 如果用于分类任务可能需要调整
        )
        print("Pretrained model downloaded successfully.")
    else:
        print("Pretrained model already exists locally.")


if __name__ == "__main__":
    download_pretrained_models()
    print("Setup complete! You can now run the training scripts.")