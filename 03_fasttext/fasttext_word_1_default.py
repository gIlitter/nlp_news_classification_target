"""
演示
    fasttext文本分类训练, 采用手动调参，进行词级分词
涉及到的API:
    fasttext.train_supervised()
"""
# 导包
import fasttext
from config import Config
import os

# 1.创建配置对象
config = Config()
# 2.模型训练
model = fasttext.train_supervised(
    input = config.word_train_path, # 训练数据路径
    dim=10, # 词向量维度
    wordNgrams=1, # n-gram特征，默认为1，即不使用n-gram特征，设置为2表示使用2-gram特征
    epoch=5,
    lr=0.1
)
# 3.模型保存
# 模型保存路径设为ft_model_word_1.bin,
# ft_model.bin -> ft_model_word_1.bin
model.save_model(config.ft_model_path.replace(".bin", "_word_1.bin"))
print(f"模型已保存到: {config.ft_model_path.replace('.bin', '_word_1.bin')}")
# 4.模型评估
result = model.test(config.word_test_path)
print(f"评估结果(样本数, 精确率P, 召回率R): {result}")
f1_score = 2 * result[1] * result[2] / (result[1] + result[2]) if (result[1] + result[2]) > 0 else 0
print(f"F1-score(micro): {f1_score:.4f}")
# 要计算F1-score(micro)，需要自己手动计算,先预测结果model.predict()，然后使用sklearn.metrics中的f1_score函数计算F1-score(micro)
# 5.打印模型关键信息，比如词向量，类别标签
print(f"词向量: {model.get_word_vector('海')}")
print(f"类别标签: {model.get_labels()}")
# print(f"类别标签: {model.labels}")
# 词表里的词
# print(f"词: {model.get_words(include_freq=True)}")
# 词表大小
print(f"词表大小: {len(model.get_words())}")