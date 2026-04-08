"""
演示
    fasttext文本分类训练, 采用自动调参，进行词级分词
涉及到的API:
    fasttext.train_supervised(
        input, # 训练数据路径
        autotuneValidationFile, # 自动调参验证集路径，设置后会自动进行调参
    )
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
    autotuneValidationFile = config.word_dev_path,  # 自动调参验证集路径
    verbose = 3, # 输出训练过程的详细信息，默认为2，设置为3表示输出更详细的信息
    seed = 4, # 随机数种子，默认为0
    autotuneDuration = 60*5,
)
# 3.模型保存
# 模型保存路径设为ft_model_word_2.bin,
# ft_model.bin -> ft_model_word_2.bin
model.save_model(config.ft_model_path.replace(".bin", "_word_2.bin"))
print(f"模型已保存到: {config.ft_model_path.replace('.bin', '_word_2.bin')}")
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