"""
演示
    基于预训练BERT模型的下游分类微调模型的模型预测

"""

# 导包
import time
import torch
from config import Config
from bert_train import MyBertClassifier
from transformers import BertTokenizer

# 1.创建配置对象
config = Config()
# 2.加载模型
model = MyBertClassifier().to(config.device)
# 加载模型文件
model.load_state_dict(torch.load(
    config.bert_model_path, # 本地保存的预训练BERT的下游分类模型文件
    weights_only=True, # 只加载模型权重参数，不加载其他对象，确保安全
    map_location=config.device, # 将模型加载到指定设备上，自动适配CPU或GPU，防止训练和推理阶段设备不一致
))
# 3.初始化分词器
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
# 4.模型预测函数
@torch.no_grad()
def predict_fun(data):
    """
    预测函数，输入文本字典，返回带有预测结果的字典
    :param data: 文本字典{text: "文本内容"}
    :return: 带有预测结果的字典{text: "文本内容", pred_class: "预测类别名称"}
    """
    # 1.设为评估模式
    model.eval()
    # 2.获取输入文本
    text = data["text"]
    # 3.分词器编码，获取input_ids, attention_mask, token_type_ids(这里不用它)
    output = BERT_TOKENIZER(
        [text],   # 输入文本列表
        padding=True,  # 填充到当前批次的最大长度
        truncation=True,  # 截断到最大长度max_length
        max_length=config.max_len*4,  # 最大长度
    )
    # input_ids, attention_mask, token_type_ids
    # 4.bert下游分类模型的预测
    logits = model(
        input_ids=torch.tensor(output.input_ids).to(config.device),
        attention_mask=torch.tensor(output.attention_mask).to(config.device),
    )   # (B,num_classes)
    # 5.获取预测类别
    y_pred_class = torch.argmax(logits, dim=-1).item() # (1,)
    y_pred_class = config.id2class[y_pred_class]
    # 6.返回结果
    data["pred_class"] = y_pred_class
    return data

# 测试
if __name__ == '__main__':
    # 1.创建输入文本
    data = {
        "text": "不知道为什么，今天香香不爱说话, 感觉有点不对劲，可能是生病了吧？"
    }
    # 2.进行预测
    result = predict_fun(data)
    print(result)
    # 计算推理延迟
    start = time.time()
    for i in range(100):
        predict_fun(data)
    end = time.time()
    print(f"推理延迟：{(end - start) / 100*1000:.5f}ms")
    ...