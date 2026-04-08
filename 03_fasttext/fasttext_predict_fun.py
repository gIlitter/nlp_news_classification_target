"""
演示
    模型预测,这里采用ft_model_char_2.bin(按字分词+自动调参)
涉及到的API:
    fasttext.load_model
    model.predict
"""

# 导包
import fasttext
import time
import jieba
from config import Config

# 取消警告显示
import warnings
warnings.filterwarnings("ignore")

# 1.创建配置对象
config = Config()
# 2.加载模型
model = fasttext.load_model(config.ft_model_path.replace(".bin", "_char_2.bin"))
# 3.定义函数，实现模型预测
def predict_fun(data):
    """
    预测函数，把传入的data字典，调用模型预测得到预测结果，返回带有预测结果的字典
    :param data: 传入的字典，格式{text: 文本内容}
    :return: 带有预测结果的字典，格式{text: 文本内容, pred_class: 预测标签名称}
    """
    # 1.分词，按字分词
    text = data["text"]
    split_text = " ".join(list(text))
    # 2.模型预测
    y_pred = model.predict(split_text)
    # print(y_pred)   # (标签,概率)
    # 3.获取预测标签名称
    y_pred = y_pred[0][0].replace("__label__", "")
    data["pred_class"] = y_pred
    return data

# 4.测试
if __name__ == '__main__':
    data = {"text": "恒指下午跌幅收窄 可留意牛64357熊65110"}
    result = predict_fun(data)
    print(result)
    # 计算推理延迟：单条样本的平均推理时长
    start_time = time.time()
    for i in range(100):
        result = predict_fun(data)
    end_time = time.time()
    print(f"推理延迟为：{(end_time - start_time) / 100 * 1000:.2f}ms")