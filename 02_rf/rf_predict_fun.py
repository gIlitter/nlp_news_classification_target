"""
演示
    随机森林模型的 模型预测
"""

# 导包
import jieba    # 中文分词工具
from config import Config
import pickle
import time
# 取消警告显示
import warnings
warnings.filterwarnings("ignore")

# 1.加载配置文件
config = Config()

# 2.加载模型和向量化器
with open(config.rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)
with open(config.tfidf_model_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 3.定义函数，实现模型预测
def predict_fun(data):
    """
    模型预测函数
    :param data: 字典，包含文本数据，格式为{"text": "文本内容"}
    :return: 字典，包含预测结果，格式为{"text": "文本内容","pred_class": "预测标签名称"}
    """
    # 1.对输入的文本进行分词
    words = " ".join(jieba.lcut(data["text"])[:30])
    # print(words)
    # 2.TF-IDF向量化
    features = tfidf_vectorizer.transform([words])
    # 3.模型预测
    y_pred = rf_model.predict(features)[0] # (1)
    print(y_pred)
    # 4.根据预测标签索引获取预测标签名称
    # 构造标签索引和名称的映射字典
    with open(config.class_path, 'r', encoding='utf-8') as f:
        # 1.使用for循环遍历行，去掉空格和空白, 内存占用小
        class_list = [line.strip() for line in f if line.strip()]
        # 2.直接使用f.read().split()来一次性获取所有行，内存占用大
        # class_list = f.read().split()
    id2class = {i: class_name for i, class_name in enumerate(class_list)}
    print(id2class)
    pred_class = id2class[y_pred]
    # 5.返回结果
    data["pred_class"] = pred_class
    return data

# 测试
if __name__ == '__main__':
    data = {"text": "从前有一段真挚的爱情放到晋伟的面前，可惜他当时正在黑马学习AI大模型"}
    result = predict_fun(data)
    print(result)
    # 计算推理延迟：单条样本的平均推理时长
    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    end_time = time.time()
    print(f"推理延迟为：{(end_time - start_time)/100*1000:.2f}ms")
