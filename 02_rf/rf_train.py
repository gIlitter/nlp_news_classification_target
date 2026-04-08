"""
演示
    训练 随机森林模型 和 TF-IDF向量化器
工作流：
    1.加载数据集
    2.文本数值化:使用TF-IDF提取文本特征
    3.训练随机森林模型
    4.模型评估
    5.保存模型

"""
# 导包
import pandas as pd
import pickle   # 用于保存模型,sklearn模型可以使用pickle进行序列化和反序列化
from sklearn.feature_extraction.text import TfidfVectorizer # 用于文本特征提取，TF-IDF向量化器
from sklearn.model_selection import train_test_split    # 用于划分训练集和验证集
# 模型评估指标：准确率、精确率、召回率、F1-score、分类评估报告、混淆矩阵
# F1-score = 2P*R/(P+R), P是精确率，R是召回率
# confusion_matrix: 混淆矩阵，矩阵的行表示真实类别，列表示预测类别
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from config import Config
# 集成学习模型：随机森林、AdaBoost、GBDT
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from tqdm import tqdm
# 0.pandas的基础设置
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)  # 确保所有列可见
# 创建配置对象
config = Config()

# 1.加载数据集，训练集的前20000条数据
train_data = pd.read_csv(config.process_train_path)[:20000]
print(f"训练集样本量为：{len(train_data)}")
# print(train_data.head())
# 2.文本数值化:使用TF-IDF提取文本特征
# 2.1 提取特征列words
words = train_data['words']
# print(words)
# 2.2 提取标签列
labels = train_data['label']
# 2.3 读取停用词文件，返回停用词列表
with open(config.stopwords_path, 'r', encoding='utf-8') as f:
    # 1.使用for循环遍历行，去掉空格和空白, 内存占用小
    stopwords = [line.strip() for line in f if line.strip()]
    # 2.直接使用f.read().split()来一次性获取所有行，内存占用大
    # stopwords = f.read().split()
print(f"停用词数量为：{len(stopwords)}")
print(f"停用词为：{stopwords[:20]}")
# 2.4 创建TF-IDF向量化器对象，设置停用词参数
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
# 2.5 使用TF-IDF向量化器对象，对words进行向量化
# fit_transform: fit训练，transform转换文本为数值向量
features = tfidf_vectorizer.fit_transform(words)    # (batch_size, vocab_size)
print(type(features))
print(f"特征矩阵的维度为：{features.shape}")
print(f"features: {features[:10]}")
# 查看词表
print(f"词表大小为：{len(tfidf_vectorizer.vocabulary_)}")

# 3.训练随机森林模型
# 3.1 划分训练集和测试集，使用train_test_split函数，test_size=0.2
x_train, x_test, y_train, y_test = train_test_split(
    features,   # 输入特征
    labels,    # 输出标签
    test_size=0.2,  # 测试集比例
    random_state=4
)
print(f"训练集样本量为：{x_train.shape}")   # (16000, vocab_size)
print(f"测试集样本量为：{x_test.shape}")    # (4000, vocab_size)
# 3.2 创建随机森林模型对象
rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(x_train, y_train)
# 4.模型评估
y_pred = rf_model.predict(x_test)
# print(f"预测结果为：{y_pred[:20]}")
# acc = 预测正确的样本数量 / 总样本数量
print(f"准确率为：{accuracy_score(y_test, y_pred)}")
# 分类指标的计算方式: macro / micro / weighted
# macro: 宏平均，计算各个类别指标，再平均，每个类别权重一样
# weighted: 加权平均，计算各个类别指标，再按样本数量加权平均，大类的权重更大
# micro: 微平均，汇总所有类别的TP/FP/FN，再计算指标，大类的权重会更大
# presion = TP / (TP + FP)
print(f"精确率macro为：{precision_score(y_test, y_pred, average='macro')}")
print(f"精确率micro为：{precision_score(y_test, y_pred, average='micro')}")
print(f"精确率weighted为：{precision_score(y_test, y_pred, average='weighted')}")
# recall = TP / (TP + FN)
print(f"召回率macro为：{recall_score(y_test, y_pred, average='macro')}")
print(f"召回率micro为：{recall_score(y_test, y_pred, average='micro')}")
print(f"召回率weighted为：{recall_score(y_test, y_pred, average='weighted')}")
# F1-score = 2 * precision * recall / (precision + recall)
print(f"F1-score macro为：{f1_score(y_test, y_pred, average='macro')}")
print(f"F1-score micro为：{f1_score(y_test, y_pred, average='micro')}")
print(f"F1-score weighted为：{f1_score(y_test, y_pred, average='weighted')}")
# 分类评估报告
print(f"分类评估报告为：\n{classification_report(y_test, y_pred)}")
# 混淆矩阵
print(f"混淆矩阵为：\n{confusion_matrix(y_test, y_pred)}")
# 5.保存模型
with open(config.rf_model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"随机森林模型保存路径为：{config.rf_model_path}")
with open(config.tfidf_model_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF向量化器保存路径为：{config.tfidf_model_path}")