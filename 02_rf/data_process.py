"""
    数据预处理, 进行jieba分词, 并保存为CSV文件
"""

# 导包
import pandas as pd # 处理表格数据
import jieba # 中文分词
from config import Config # 配置文件
import matplotlib.pyplot as plt # 数据可视化
import seaborn as sns

# 1.创建配置对象
config = Config()
# 2.定义函数，处理数据，进行jieba分词，并保存为CSV文件
def process_data(data_path, save_path):
    """
    数据处理函数，对原始数据集text进行jieba分词并保存为CSV文件
    :param data_path: 原始数据文件路径，txt格式，每行格式：label \t text
    :param save_path: 处理后数据文件路径，CSV格式，包含两列：label和text
    :return: 无
    """
    # 1.读取原始文件，返回pandas数据框
    data = pd.read_csv(data_path, sep="\t", names=["text","label"])
    print(data.head())
    # 2.进行jieba分词
    def cut_words(sentence):
        return " ".join(jieba.lcut(sentence)[:20])
    data['words'] = data["text"].apply(cut_words)
    # 3.进行jieba分词，获取序列长度seq_len
    def get_seq_len(sentence):
        return len(jieba.lcut(sentence)[:20])
    data['seq_len'] = data["text"].apply(get_seq_len)
    print(data.head())
    # 4.可视化序列长度分布
    data['seq_len'].hist()
    plt.show()
    print(data['seq_len'].describe())
    # 5.保存处理后的数据为CSV文件
    data.to_csv(save_path, index=False)
    print(f"处理后的数据保存在：{save_path}")

# 3.处理数据
# 训练集
process_data(data_path=config.train_path, save_path=config.process_train_path)
# 验证集
process_data(data_path=config.dev_path, save_path=config.process_dev_path)
# 测试集
process_data(data_path=config.test_path, save_path=config.process_test_path)