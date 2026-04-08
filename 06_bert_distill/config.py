"""
    BERT知识蒸馏模型配置文件，统一管理全局配置信息。
"""
import os   # 文件操作模块
from pathlib import Path  # 路径操作模块
# 不显示警告信息
import warnings
import torch

warnings.filterwarnings("ignore")

# 1.定义配置类，集中管理全局配置信息
class Config():
    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径 - 指向01_data目录
        self.root_path = os.path.join(Path(__file__).parent.parent, "01_data")
        # 2.初始化数据文件路径（训练和评估需要）
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 3.初始化类别文件路径（预测时需要）
        self.class_path = os.path.join(self.root_path, "class.txt")
        # 4.预训练的BERT模型路径
        self.bert_path = os.path.join(self.root_path, "bert-base-chinese")
        # 5.模型保存路径
        os.makedirs(os.path.join(self.root_path, "save_model"), exist_ok=True)
        # BERT微调模型文件
        self.bert_model_path = os.path.join(self.root_path, "save_model", "bert_model.pt")
        # BERT量化模型文件
        self.bert_quantization_path = os.path.join(self.root_path, "save_model", "bert_quantization.pt")
        # 知识蒸馏的学生模型的保存路径
        self.student_model_path = os.path.join(self.root_path, "save_model", "student_model.pt")
        # 6.API配置（部署需要）
        self.api_host = "127.0.0.1"
        self.api_port = 5000
        # 7.类别索引-名称的映射字典（预测时需要）
        with open(self.class_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        self.id2class = {i: class_name for i, class_name in enumerate(class_list)}
        # 8.全局配置参数
        # 设备, cuda > mps > cpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # 超参数：训练前设置的参数
        self.epochs = 5  # 训练轮数, BERT微调通常3~5轮
        self.batch_size = 64  # 每批次样本数量
        self.lr = 2e-5  # 学习率, BERT微调通常1e-5~5e-5
        self.max_len = 32   # 最大序列长度
        self.weight_decay = 1e-2    # 权重衰减, AdamW优化器参数
        self.alpha = 0.7    # 模型蒸馏的权重
        self.T = 4.0    # 模型蒸馏的温度

# 2.测试
if __name__ == "__main__":
    # 1.创建配置对象
    config = Config()
    # 2.打印配置信息，验证是否正确加载
    print("根路径:", config.root_path)
    print("训练数据路径:", config.train_path)
    print("验证数据路径:", config.dev_path)
    print("测试数据路径:", config.test_path)
    print("停用词文件路径:", config.stopwords_path)
    print("类别文件路径:", config.class_path)
    print(f"处理后的训练数据路径: {config.process_train_path}")
    print(f"服务器地址: api_host: {config.api_host} | "
          f"端口号:{config.api_port}")
    print(f"类别索引-名称的映射字典: {config.id2class}")
    print(f"预训练的BERT模型路径: {config.bert_path}")
    print(f"BERT微调模型保存路径: {config.bert_model_path}")
    print(f"全局配置参数: {config.weight_decay}")
