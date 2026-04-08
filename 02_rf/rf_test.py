"""
演示
    随机森林模型的 模型评估
"""

# 导包
import pandas as pd
import pickle
from config import Config
# 分类模型评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
# 取消警告显示
import warnings
warnings.filterwarnings("ignore")
# 设置pandas显示
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)  # 确保所有列可见

# 1.创建配置对象
config = Config()
# 2.加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
# 3.加载dev数据
dev_data = pd.read_csv(config.process_dev_path)
print(dev_data.shape)   # (10000,4)
# 构造输入words
words = dev_data['words']
# 构造输出labels
labels = dev_data['label']
# 对输入words进行TF-IDF向量化
print(f"TF-IDF向量化器的词表大小为：{len(tfidf_vectorizer.vocabulary_)}")
features = tfidf_vectorizer.transform(words)    # (batch_size, vocab_size)
print(f"features的维度为：{features.shape}")
# 4.模型预测
y_pred = rf_model.predict(features)
# print(len(y_pred))
# 5.模型评估
# acc = 预测正确的样本数量 / 总样本数量
print(f"准确率为：{accuracy_score(labels, y_pred)}")
# 分类指标的计算方式: macro / micro / weighted
# macro: 宏平均，计算各个类别指标，再平均，每个类别权重一样
# weighted: 加权平均，计算各个类别指标，再按样本数量加权平均，大类的权重更大
# micro: 微平均，汇总所有类别的TP/FP/FN，再计算指标，大类的权重会更大
# presion = TP / (TP + FP)
print(f"精确率macro为：{precision_score(labels, y_pred, average='macro')}")
# print(f"精确率micro为：{precision_score(labels, y_pred, average='micro')}")
# print(f"精确率weighted为：{precision_score(labels, y_pred, average='weighted')}")
# recall = TP / (TP + FN)
print(f"召回率macro为：{recall_score(labels, y_pred, average='macro')}")
# print(f"召回率micro为：{recall_score(labels, y_pred, average='micro')}")
# print(f"召回率weighted为：{recall_score(labels, y_pred, average='weighted')}")
# F1-score = 2 * precision * recall / (precision + recall)
print(f"F1-score macro为：{f1_score(labels, y_pred, average='macro')}")
# print(f"F1-score micro为：{f1_score(labels, y_pred, average='micro')}")
# print(f"F1-score weighted为：{f1_score(labels, y_pred, average='weighted')}")
# 分类评估报告
print(f"分类评估报告为：\n{classification_report(labels, y_pred)}")
# 混淆矩阵
print(f"混淆矩阵为：\n{confusion_matrix(labels, y_pred)}")