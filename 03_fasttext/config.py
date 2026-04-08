"""
    FastText模型配置文件，统一管理全局配置信息。
"""
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Config():
    def __init__(self):
        # 1.初始化根路径 - 指向01_data目录
        self.root_path = os.path.join(Path(__file__).parent.parent, '01_data')
        
        # 2.初始化类别文件路径（预测时需要）
        self.class_path = os.path.join(self.root_path, 'class.txt')
        
        # 3.初始化按字符分词的数据文件路径（训练和评估需要）
        self.char_train_path = os.path.join(self.root_path, 'char_train.txt')
        self.char_dev_path = os.path.join(self.root_path, 'char_dev.txt')
        self.char_test_path = os.path.join(self.root_path, 'char_test.txt')
        
        # 4.初始化按词分词的数据文件路径（训练和评估需要）
        self.word_train_path = os.path.join(self.root_path, 'word_train.txt')
        self.word_dev_path = os.path.join(self.root_path, 'word_dev.txt')
        self.word_test_path = os.path.join(self.root_path, 'word_test.txt')
        
        # 5.模型保存路径
        os.makedirs(os.path.join(self.root_path, 'save_model'), exist_ok=True)
        self.ft_model_path = os.path.join(self.root_path, 'save_model', 'ft_model.bin')
        
        # 6.API配置（部署需要）
        self.api_host = '127.0.0.1'
        self.api_port = 5000
        
        # 7.类别索引-名称的映射字典（预测时需要）
        with open(self.class_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        self.id2class = {i: classname for i, classname in enumerate(class_list)}

if __name__ == '__main__':
    config = Config()
    print('根路径:',config.root_path)
    print('训练数据路径:',config.train_path)
    print('验证数据路径:',config.dev_path)
    print('测试数据路径:',config.test_path)
    print('停用词文件路径:',config.stopwords_path)
    print('类别文件路径:',config.class_path)
    print(f'处理后的训练数据路径:{config.process_train_path}')
    print(f'服务器地址:api_host:{config.api_host} | '
          f'端口号:{config.api_port}')
    print(f'类别索引-名称的映射字典:{config.id2class}')



