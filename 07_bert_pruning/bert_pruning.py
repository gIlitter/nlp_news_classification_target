import os

import torch
from torch.nn.utils import prune
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from bert_train import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

config = Config()
def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    num_encoder_layers = len(model.bert.encoder.layer)
    for i in range(num_encoder_layers):
        query_weight = model.bert.encoder.layer[i].attention.self.query.weight
        # print(query_weight)
        total_params += query_weight.numel()
        zero_params += (query_weight == 0).sum().item()
    sparsity = zero_params/total_params
    return sparsity

def prune_model(model):
    num_encoder_layers = len(model.bert.encoder.layer)
    prune_params = [
        (model.bert.encoder.layer[i].attention.self.query,'weight') for i in range(num_encoder_layers)
    ]
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=0.3,
    )
    # prune.global_unstructured(
    #     prune_params,
    #     pruning_method=prune.RandomUnstructured,  # 剪枝方式: 随机剪枝
    #     amount=0.8,  # 剪枝比例
    # )
    for module,params in prune_params:
        prune.remove(module,params)
    torch.save(model.state_dict(),config.bert_pruning_path)
    return model

if __name__ == '__main__':
    model = MyBertClassifier().to(config.device)
    model.load_state_dict(torch.load(config.bert_model_path,weights_only=True,map_location=config.device))
    train_dataloader,valid_dataloader,test_dataloader = build_dataloader()
    test_loss,test_acc,test_f1 = evaluate(model,test_dataloader,loss_fn=nn.CrossEntropyLoss())
    print(f'剪枝前模型F1分数:{test_f1}')
    result = calculate_sparsity(model)
    print(f'剪枝前模型稀疏度:{result}')
    model = prune_model(model)
    # 5.测试剪枝后的模型
    test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
    print(f"剪枝后模型F1分数：{test_f1}")
    # 6.打印剪枝后模型的稀疏度
    result = calculate_sparsity(model)
    print(f"剪枝后模型稀疏度：{result}")