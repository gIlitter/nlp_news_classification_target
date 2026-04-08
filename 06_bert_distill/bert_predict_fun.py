"""
演示
    基于预训练BERT的下游分类模型的模型预测

"""

# 导包
import time
import torch
from config import Config
from bert_train import MyBertClassifier
from student_train import MyStudentClassifier
from transformers import BertTokenizer

# 1.创建Config对象
config = Config()
# 初始化BERT分词器
bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)
# 2.加载模型，并迁移到设备
# 2.1 加载教师模型
teacher_model = MyBertClassifier().to(config.device)
# weights_only=True,表示只加载权重数据，不加载任何可执行对象，更安全，更适合模型部署
teacher_model.load_state_dict(torch.load(
    config.bert_model_path,
    weights_only=True,
    map_location=config.device
)
)
# 2.2 加载学生模型
student_model = MyStudentClassifier().to(config.device)
student_model.load_state_dict(torch.load(
    config.student_model_path,
    weights_only=True,
    map_location=config.device
)
)
# 3.模型预测，输入文本，返回预测结果
# 3.1 教师模型预测
@torch.no_grad()    # 关闭梯度计算
def predict_fun1(data_dict):
    """
    输入文本，使用BERT模型预测类别
    :param data_dict: 输入文本字典，{text: 文本}
    :return: 预测结果，{text: 文本, pred_class: 预测类别}
    """
    # 1.设为评估模式
    teacher_model.eval()
    # 2.获取输入文本
    text = data_dict['text']
    # 3.BERT分词器编码
    output = bert_tokenizer(
        [text], # 输入文本列表
        padding=True,   # 填充, 按照batch内最大长度进行填充
        truncation=True,    # 截断, 截断至最大长度
        max_length=config.max_len,  # 最大长度
        return_tensors='pt',  # 返回张量
    )
    # 获取张量，input_ids, attention_mask
    input_ids = output['input_ids'].to(config.device)
    attention_mask = output['attention_mask'].to(config.device)
    # 4.前向传播
    logits = teacher_model(input_ids, attention_mask)
    # 5.获取预测类别名称
    y_pred_class = config.id2class[logits.argmax(dim=-1).item()]
    data_dict['pred_class'] = y_pred_class
    # 6.返回结果
    return data_dict

# 3.2 学生模型预测
@torch.no_grad()    # 关闭梯度计算
def predict_fun2(data_dict):
    """
    输入文本，使用BERT模型预测类别
    :param data_dict: 输入文本字典，{text: 文本}
    :return: 预测结果，{text: 文本, pred_class: 预测类别}
    """
    # 1.设为评估模式
    student_model.eval()
    # 2.获取输入文本
    text = data_dict['text']
    # 3.BERT分词器编码
    output = bert_tokenizer(
        [text], # 输入文本列表
        padding=True,   # 填充, 按照batch内最大长度进行填充
        truncation=True,    # 截断, 截断至最大长度
        max_length=config.max_len,  # 最大长度
        return_tensors='pt',  # 返回张量
    )
    # 获取张量，input_ids, attention_mask
    input_ids = output['input_ids'].to(config.device)
    attention_mask = output['attention_mask'].to(config.device)
    # 4.前向传播
    logits = student_model(input_ids, attention_mask)
    # 5.获取预测类别名称
    y_pred_class = config.id2class[logits.argmax(dim=-1).item()]
    data_dict['pred_class'] = y_pred_class
    # 6.返回结果
    return data_dict

# 测试
if __name__ == '__main__':
    # 1.创建输入文本
    data_dict = {"text": "马上就要放假啦，我十分开心"}
    # 2.调用预测函数
    result = predict_fun1(data_dict)
    print(result)
    # 计算推理延迟
    start = time.time()
    for i in range(100):
        predict_fun1(data_dict)
    end = time.time()
    print(f"推理延迟：{(end - start) / 100 *1000:.5f}ms")
    result = predict_fun2(data_dict)
    print(result)
    # 计算推理延迟
    start = time.time()
    for i in range(100):
        predict_fun2(data_dict)
    end = time.time()
    print(f"推理延迟：{(end - start) / 100*1000:.5f}ms")