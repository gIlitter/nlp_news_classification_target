"""
演示
    bert模型的 动态量化DQ
需要掌握的API:
    torch.quantization.quantize_dynamic()
    只能在CPU上使用，不能在GPU上使用
注意：
    新建一个虚拟环境nlp_cpu, conda create -n nlp_cpu python=3.10
    安装cpu版本的pytorch, pip3 install torch torchvision
    其余包自己搞定
"""
# 导包
import torch
from torch import nn
from config import Config
from bert_train import MyBertClassifier, evaluate, build_dataloader

# 0.查看量化引擎
# 常见量化引擎：onednn(AMD/intel),fbgemm(intel)
print("当前量化引擎:", torch.backends.quantized.engine)

# 1.创建配置对象
config = Config()
# 2.创建数据加载器
train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
# 3.加载模型，权重参数精度float32
model = MyBertClassifier()
model.load_state_dict(
    torch.load(
        config.bert_model_path,
        weights_only=True,  # 安全加载模型权重参数
        map_location=torch.device("cpu")  # 将模型加载到CPU上
    )
)
# 4.评估原始模型性能
test_loss, test_acc, test_f1 = evaluate(
    model, test_dataloader,loss_fn=nn.CrossEntropyLoss()
)
print("原始模型性能:", test_loss, test_acc, test_f1)
# 5.进行动态量化torch.quantization.quantize_dynamic()
# 设为评估模式，关闭dropout等训练特有的层
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model,  # 需要量化的模型
    {nn.Linear},  # 需要量化的层
    dtype=torch.qint8,  # 量化的数据类型,常用int8
)
# 6.评估量化后模型性能
test_loss, test_acc, test_f1 = evaluate(
    quantized_model, test_dataloader,loss_fn=nn.CrossEntropyLoss()
)
print("量化后模型性能:", test_loss, test_acc, test_f1)
# 7.保存量化模型参数
torch.save(quantized_model.state_dict(), config.bert_quantization_path)
print("保存量化模型到：", config.bert_quantization_path)
