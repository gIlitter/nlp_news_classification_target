"""
    配置文件，统一管理全局配置信息，包括数据路径、类别信息等。方便项目的管理和后期维护。
"""
import os   # 文件操作模块
from pathlib import Path # 路径操作模块
# 1.定义配置类，集中管理全局配置信息，比如数据路径、类别信息
class Config():
    # 1.定义__init__方法，初始化配置项
    def __init__(self):
        # 1.初始化根路径
        self.root_path = Path(__file__).parent
        # 2.初始化数据文件路径
        self.train_path = os.path.join(self.root_path, "train.txt")
        self.dev_path = os.path.join(self.root_path, "dev.txt")
        self.test_path = os.path.join(self.root_path, "test.txt")
        # 停用词文件路径
        self.stopwords_path = os.path.join(self.root_path, "stopwords.txt")
        # 3.初始化类别文件路径
        self.class_path = os.path.join(self.root_path, "class.txt")

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
