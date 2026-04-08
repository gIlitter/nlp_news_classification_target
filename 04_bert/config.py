"""
    配置文件，统一管理全局配置信息，包括数据路径、类别信息等。方便项目的管理和后期维护。
"""
import os   # 文件操作模块
from pathlib import Path  # 路径操作模块
# 不显示警告信息
import warnings
import torch

warnings.filterwarnings("ignore")

# 1.定义配置类，集中管理全局配置信息，比如数据路径、类别信息
class Config():
    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径
        # self.root_path = os.path.join(Path(__file__).parent.parent, "01-data")
        self.root_path = r"D:\PycharmProjects\news_classification\01_data"
        # 2.初始化数据文件路径
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")
        # 4.初始化 处理后的数据文件路径
        self.process_train_path = os.path.join(self.root_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.root_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.root_path, "process_test.csv")
        # 添加按字符分词的数据文件路径
        self.char_train_path = os.path.join(self.root_path, "char_train.txt")
        self.char_dev_path = os.path.join(self.root_path, "char_dev.txt")
        self.char_test_path = os.path.join(self.root_path, "char_test.txt")
        # 添加按词分词的数据文件路径
        self.word_train_path = os.path.join(self.root_path, "word_train.txt")
        self.word_dev_path = os.path.join(self.root_path, "word_dev.txt")
        self.word_test_path = os.path.join(self.root_path, "word_test.txt")
        # 5.模型保存路径
        os.makedirs(os.path.join(self.root_path, "save_model"), exist_ok=True)
        # pkl格式，用于保存sklearn机器学习模型
        self.rf_model_path = os.path.join(self.root_path, "save_model", "rf_model.pkl")
        self.tfidf_model_path = os.path.join(self.root_path, "save_model", "tfidf_model.pkl")
        self.ft_model_path = os.path.join(self.root_path, "save_model", "ft_model.bin")
        # 6.模型预测结果路径，txt
        os.makedirs(os.path.join(self.root_path, "result"), exist_ok=True)
        self.model_predict_result = os.path.join(self.root_path, "result", "model_predict_result.txt")
        # 7.API配置
        self.api_host = "127.0.0.1" # 本地地址
        self.api_port = 5000 # 端口号
        # 8.类别索引-名称的映射字典
        with open(self.class_path, 'r', encoding='utf-8') as f:
            # 1.使用for循环遍历行，去掉空格和空白, 内存占用小
            class_list = [line.strip() for line in f if line.strip()]
            # 2.直接使用f.read().split()来一次性获取所有行，内存占用大
            # class_list = f.read().split()
        self.id2class = {i: class_name for i, class_name in enumerate(class_list)}
        # 9.预训练的BERT模型路径
        self.bert_path = os.path.join(self.root_path, "bert-base-chinese")
        # 10.BERT微调模型保存路径
        # pytorch格式，.pt/.pth
        self.bert_model_path = os.path.join(self.root_path, "save_model", "bert_model.pt")
        # 11.全局配置参数
        # 设备, cuda > mps > cpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # 超参数：训练前设置的参数，在训练过程中不会学习，不会由反向传播进行更新
        self.epochs = 5 # 训练轮数, BERT微调通常3~5轮
        self.batch_size = 64 # 每批次样本数量
        self.lr = 2e-5 # 学习率, BERT微调通常1e-5~5e-5
        self.max_len = 32   # 最大序列长度
        self.weight_decay = 1e-2    # 权重衰减, AdamW优化器参数

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
