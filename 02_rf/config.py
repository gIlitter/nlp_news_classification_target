"""
    随机森林模型配置文件，统一管理全局配置信息。
"""
import os  # 文件操作模块
from pathlib import Path  # 路径操作模块
# 不显示警告信息
import warnings

warnings.filterwarnings("ignore")


# 1.定义配置类，集中管理全局配置信息
class Config(object):
    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径
        # self.root_path = Path(__file__).parent
        self.root_path = r"D:\PycharmProjects\news_classification\01_data"
        # 2.初始化数据文件路径
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")
        # 3.初始化处理后的数据文件路径（训练和评估需要）
        self.process_train_path = os.path.join(self.root_path, "process_train.csv")
        self.process_dev_path = os.path.join(self.root_path, "process_dev.csv")
        self.process_test_path = os.path.join(self.root_path, "process_test.csv")
        # 4.停用词文件路径（训练时需要）
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 5.模型保存路径
        os.makedirs(os.path.join(self.root_path, "save_model"), exist_ok=True)
        self.rf_model_path = os.path.join(self.root_path, "save_model", "rf_model.pkl")
        self.tfidf_model_path = os.path.join(self.root_path, "save_model", "tfidf_model.pkl")
        # 6.API配置（部署需要）
        self.api_host = "127.0.0.1"
        self.api_port = 5000


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

    ...
