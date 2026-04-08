"""
进行数据预处理,将原始文本 转为 fasttext需要的输入格式
"""

# 导包
import jieba
from config import Config

# 1.创建配置对象
config = Config()
# 2.定义函数，将 原始文本 转为 fasttext需要的输入格式
def process_data(data_path, save_path, is_char=True):
    """
    数据处理函数，加载原始文本，生成处理后的数据并保存
    :param data_path: 原始数据文件路径
    :param save_path: 处理后的数据文件保存路径
    :param is_char: 是否进行字符级分词，默认为True，如果为False则进行jieba词级分词
    :return: None
    """
    # 1.打开原始文件
    with open(data_path, 'r', encoding='utf-8') as f:
        # 2.写入处理后的数据到save_path
        with open(save_path, 'w', encoding='utf-8') as fw:
            # 3.循环读取每行
            for line in f:
                # print(line,end="")
                # 4.去掉首尾空格和空白
                line = line.strip()
                # 5.分割文本和标签，使用'\t'分割
                text, label = line.split('\t')
                # print(text)
                # print(label)
                # 6.将标签label转换为对应的字符串
                label_name = config.id2class[int(label)]
                # print(label_name)
                # 7.根据is_char参数选择分词方式
                if is_char:
                    # 7.1 进行字符级分词
                    split_text = " ".join(list(text))
                else:
                    # 7.2 进行jieba词级分词
                    split_text = " ".join(jieba.lcut(text))
                # 8.构建fasttext需要的输入格式，标签前加"__label__"前缀
                fasttext_line = f"__label__{label_name} {split_text}\n"
                # 9.写入处理后的数据到文件
                fw.write(fasttext_line)

# 测试
if __name__ == '__main__':
    # 处理训练集
    process_data(config.train_path, config.char_train_path, is_char=True)
    process_data(config.train_path, config.word_train_path, is_char=False)
    # 处理验证集
    process_data(config.dev_path, config.char_dev_path, is_char=True)
    process_data(config.dev_path, config.word_dev_path, is_char=False)
    # 处理测试集
    process_data(config.test_path, config.char_test_path, is_char=True)
    process_data(config.test_path, config.word_test_path, is_char=False)

