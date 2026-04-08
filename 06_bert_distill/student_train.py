"""
演示
    知识蒸馏中软标签蒸馏的学生模型的训练过程
NLP中神经网络训练的工作流：
    1.构建词表/加载分词器
        由bert_tokenizer自带的
    2.准备数据集
        拆分为训练集、验证集、测试集
        构造样本的输入输出, MyDataset类
        DataSet,collate_fn,DataLoader关系:
            Dataset 提供单个样本，collate_fn 将多个样本打包成标准等长批次，Dataloader自动按批次取数据。
            Dataset: 孙悟空拿到一个桃子；collate_fn: 把孙悟空拿到的多个桃子打包成一个篮子，并且每个篮子有64个桃子；
            DataLoader: 负责自动按批次从collate_fn打包好的篮子里取桃子
        张量 -> 数据集对象 -> 数据加载器
    3.搭建神经网络模型
        预训练BERT，输出last_hidden_state(表示token语义),pooler_output(池化输出,整个句子的语义表示)
        pooler_output句子表示
        线性层nn.Linear
    4.模型训练 - 训练集 + 验证集
        1.前向传播
        2.计算损失
        3.梯度清零
        4.反向传播
        5.更新参数
    5.模型测试
        1.前向传播
        2.计算损失
        # 3.梯度清零
        # 4.反向传播
        # 5.更新参数

模型调优思路:
    1.调整超参数：学习率、批次大小、训练轮数、权重衰减等
    2.调整模型结构：增加线性层、增加dropout层、使用不同的预训练模型等
    3.调节训练的冻结策略：刚开始冻结BERT参数，只训练线性层，训练后期再解冻部分BERT参数进行微调
    4.使用last_hidden_state进行分类: last_hidden_state(batch_size, seq_len, d_model)，Conv1d/Attention+linear

"""

# 导包
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tqdm import tqdm   # 可视化训练过程
import torch    # pytorch框架
import torch.nn as nn   # 神经网络模块
import torch.optim as optim # 优化器模块，提供各种优化器，比如SGD,Adam,AdamW
import torch.nn.functional as F   # 提供各种函数，比如softmax,log_softmax等
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig  # huggingface的transformers框架的具体模型方法
from config import Config   # 自定义的全局配置类
import time

# 计算 accuracy准确率 和 F1(精确率和召回率的调和平均)
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt # 绘图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

# 0.全局配置
# 创建配置对象
config = Config()
# 创建BERT分词器对象，BERT配置对象
BERT_TOKENIZER = BertTokenizer.from_pretrained(config.bert_path)
BERT_CONFIG = BertConfig.from_pretrained(config.bert_path)
# print(BERT_CONFIG)
# 1.构建词表/加载分词器
# 由bert_tokenizer自带的
# print(f"BERT词表: {BERT_TOKENIZER.vocab}")
# 2.准备数据集
# 拆分为训练集、验证集、测试集
# 构造样本的输入输出, MyDataset类
# 张量 -> 数据集对象 -> 数据加载器
# 2.1 定义函数，加载数据，返回 (文本,标签索引) 的列表
def load_raw_data(file_path):
    """
    加载数据文件，返回 (文本,标签索引) 的列表
    :param file_path: 原始文件路径
    :return: 元组列表 [(文本,标签索引), ...]
    """
    # 1.初始化结果列表
    results = []
    # 2.加载原始文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 1.遍历行，获取每行
        for line in f:
            # 1.去除首尾空白
            line = line.strip()
            # 2.跳过空行
            if not line:
                continue
            # 3.分割文本和标签，分隔符为'\t'
            text, label = line.split('\t')
            # 4.添加元组(文本，标签)到结果列表中
            results.append((text, int(label)))
    # 3.返回结果列表
    return results

# 2.2 定义MyDataset类，构造样本的输入和输出。继承torch.utils.data.Dataset
class MyDataset(Dataset):
    """
    自定义数据集类，构造某个样本的输入和输出，继承torch.utils.data.Dataset
     - __init__: 初始化方法，接收原始数据列表
     - __len__: 返回数据集大小
     - __getitem__: 根据索引获取样本
    """
    # 1.定义__init__方法，接收原始数据列表
    def __init__(self, data_list):
        super().__init__()
        # 1.初始化原始数据列表
        self.data_list = data_list
    # 2.定义__len__方法，返回数据集大小
    def __len__(self):
        return len(self.data_list)
    # 3.定义__getitem__方法，根据索引获取样本
    def __getitem__(self, index):
        text, label = self.data_list[index]
        return text, label

# DataSet,collate_fn,DataLoader关系:
# Dataset 提供单个样本，collate_fn 将多个样本打包成标准等长批次，Dataloader自动按批次取数据。
# 2.3 定义collate_fn函数，负责将一个批次的样本整理为相同长度的样本
# 需要进行: 文本 -> 分词器分词 ——> 填充到固定长度 -> 转换为张量
def collate_fn(batch):
    """
    collate_fn函数，负责将一个批次的样本整理为相同长度的样本
    需要进行: 文本 -> 分词器分词 ——> 填充到固定长度 -> 转换为张量
    实际流程: 文本 -> BERT分词器编码: input_ids, attention_mask, token_type_ids -> 转换为张量
    :param batch: 一个批次的样本, 内容是：[(文本,标签索引), ...]
    :return: 元组，代表输入张量和标签张量
            input_ids: token id张量，形状为(batch_size, seq_len)
            attention_mask: 注意力掩码张量，形状为(batch_size, seq_len)
            token_type_ids: 句子id张量，形状为(batch_size, seq_len)
            labels: 标签张量，形状为(batch_size,)
    """
    # 1.获取当前批次的文本和标签的元组
    texts, labels = zip(*batch)
    # 2.BERT分词器进行编码
    output = BERT_TOKENIZER(
        list(texts), # 文本列表
        padding=True, # 填充到当前批次的最大长度
        truncation=True, # 截断到设置的最大长度max_len
        max_length=config.max_len, # 设置的最大长度
        return_tensors='pt' # 返回张量,pytorch格式
    )
    # 3.获取张量input_ids，attention_mask，token_type_ids
    input_ids = output['input_ids'] # (batch_size, seq_len)
    attention_mask = output['attention_mask']
    # token_type_ids = output['token_type_ids']
    # 4.获取标签张量labels
    labels = torch.tensor(labels, dtype=torch.int64)
    # 5.返回输入张量和标签张量
    return input_ids, attention_mask, labels

# 2.4 构建数据加载器
def build_dataloader():
    # 1.加载数据集
    train_data = load_raw_data(config.train_path)
    dev_data = load_raw_data(config.dev_path)
    test_data = load_raw_data(config.test_path)
    # 2.创建数据集对象
    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)
    test_dataset = MyDataset(test_data)
    # 3.构建数据加载器对象
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn, # 批处理函数，负责将一个批次的样本整理为相同长度的样本
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,  # 批处理函数，负责将一个批次的样本整理为相同长度的样本
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,  # 批处理函数，负责将一个批次的样本整理为相同长度的样本
    )
    return train_dataloader, dev_dataloader, test_dataloader

# 3.搭建神经网络模型
# 预训练BERT，输出hidden_state(表示token语义),pooler_output(池化输出,整个句子的语义表示)
# pooler_output句子表示
# 线性层nn.Linear
class MyBertClassifier(nn.Module):
    """
    基于预训练BERT模型的下游分类模型
    输入:
        input_ids: 输入的token id张量，(batch_size, seq_len),seq_len是当前批次的最大长度
        attention_mask: 注意力掩码张量，(batch_size, seq_len)
    返回:
        logits: 预测分数，(batch_size, num_classes)
    """
    # 1.定义__init__方法，初始化BERT模型和线性层
    def __init__(self):
        super().__init__()
        # 1.初始化预训练BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 2.初始化线性层，输出层
        self.linear1 = nn.Linear(BERT_CONFIG.hidden_size, len(config.id2class))
        # 3.可选:冻结BERT模型的参数，只训练线性层
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    # 2.定义forward方法，前向传播
    def forward(self, input_ids, attention_mask):
        # input_ids: 输入的token id张量，(batch_size, seq_len),seq_len是当前批次的最大长度
        # attention_mask: 注意力掩码张量，(batch_size, seq_len)
        # 1.输入BERT模型，获取输出
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # output:last_hidden_state(batch_size,seq_len,d_model),pooler_output(batch_size,d_model)
        # 2.获取池化输出pooler_output，作为句子表示
        pooler_output = output['pooler_output'] # (batch_size, d_model)
        # 3.输入线性层，获取预测分数logits
        logits = self.linear1(pooler_output)    # (batch_size, num_classes)
        return logits
# 定义学生模型，也采用BERT架构
class MyStudentClassifier(nn.Module):
    """
    自定义BERT+linear实现分类模型
    输入:
        input_ids: 输入的token id张量，(batch_size, seq_len),seq_len是当前批次的最大长度
        attention_mask: 注意力掩码张量，(batch_size, seq_len)
    返回:
        logits: 预测分数，(batch_size, num_classes)
    """
    # 1.定义__init__方法，初始化BERT模型和线性层
    def __init__(self):
        super().__init__()
        # 1.初始化自定义BERT模型的参数
        self.student_config = BertConfig(
            vocab_size=BERT_CONFIG.vocab_size,
            hidden_size=256, # 隐藏层维度，学生模型更小
            num_hidden_layers=2, # 隐藏层数量，学生模型更少
            num_attention_heads=8, # 注意力头数量，学生模型一样
            intermediate_size=256*4, # 前馈网络维度，学生模型更小
            max_position_embeddings=config.max_len,
        )
        self.bert = BertModel(self.student_config)
        # 2.初始化线性层，输出层
        self.linear1 = nn.Linear(self.student_config.hidden_size, len(config.id2class))
        # 3.可选:冻结BERT模型的参数，只训练线性层
        # for param in self.bert.parameters():
        #     param.requires_grad = False
    # 2.定义forward方法，前向传播
    def forward(self, input_ids, attention_mask):
        # input_ids: 输入的token id张量，(batch_size, seq_len),seq_len是当前批次的最大长度
        # attention_mask: 注意力掩码张量，(batch_size, seq_len)
        # 1.输入BERT模型，获取输出
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # output:last_hidden_state(batch_size,seq_len,d_model),pooler_output(batch_size,d_model)
        # 2.获取池化输出pooler_output，作为句子表示
        pooler_output = output['pooler_output'] # (batch_size, d_model)
        # 3.输入线性层，获取预测分数logits
        logits = self.linear1(pooler_output)    # (batch_size, num_classes)
        return logits

# 4.模型训练 - 训练集 + 验证集
# 4.1 训练一个轮次的函数-重点
def train_one_epoch(
            teacher_model,
            student_model,
            train_dataloader,
            ce_loss_fn,
            kl_loss_fn,
            optimizer
        ):
    """
    学生模型训练一个轮次，并返回训练损失、准确率、f1分数
    :param teacher_model: 教师模型对象
    :param student_model: 学生模型对象
    :param train_dataloader: 训练集数据加载器对象
    :param ce_loss_fn: 交叉熵损失函数对象，计算学生模型预测分数和真实标签之间的损失
    :param kl_loss_fn: KL散度损失函数对象，计算学生模型预测分数和教师模型预测分数之间的损失
    :param optimizer: 优化器对象，更新学生模型的参数
    :return: 训练损失、准确率、f1分数
    """
    # 1.设置为训练模式
    student_model.train()
    # 2.初始化 总损失、总样本数、预测结果列表、真实标签列表
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []
    # 3.遍历数据加载器，分批训练
    for batch in tqdm(train_dataloader, desc="Training"):
        # batch: (input_ids, attention_mask, labels)
        # input_ids: (batch_size, seq_len), seq_len是当前批次的最大长度
        # attention_mask: (batch_size, seq_len)
        # labels: (batch_size,)
        # 0.迁移数据到设备
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
        # 1.前向传播
        # 1.1 教师模型前向传播，获取教师模型的预测分数
        with torch.no_grad(): # 关闭教师模型的梯度计算，节省显存
            teacher_logits = teacher_model(input_ids, attention_mask)   # (batch_size, num_classes)
        # 1.2 学生模型前向传播，获取学生模型的预测分
        student_logits = student_model(input_ids, attention_mask)   # (batch_size, num_classes)

        # 2.计算损失
        # 2.1 计算ce_loss
        ce_loss = ce_loss_fn(student_logits, labels)
        # 2.2 计算kl_loss
        kl_loss = kl_loss_fn(F.log_softmax(student_logits/config.T, dim=-1), F.softmax(teacher_logits/config.T, dim=-1))
        # 2.3 计算总损失
        loss = config.alpha * config.T**2 * kl_loss + (1-config.alpha) * ce_loss
        # 3.梯度清零
        optimizer.zero_grad()
        # 4.反向传播
        loss.backward()
        # 5.更新参数
        optimizer.step()
        # 6.记录结果
        total_loss += loss.item()*input_ids.size(0)
        total_samples += input_ids.size(0)
        y_pred_class = student_logits.argmax(dim=-1).to(device=torch.device('cpu')).tolist()    # (batch_size,)
        total_preds.extend(y_pred_class)
        total_labels.extend(labels.to(device=torch.device('cpu')).tolist())
    # 4.计算 平均损失、准确率、f1分数
    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    # macro-f1 score: 宏平均F1分数
    f1_avg = f1_score(total_labels, total_preds, average='macro')
    return avg_loss, acc, f1_avg

# 4.2 模型评估
@torch.no_grad()    # 关闭梯度计算,节省20%显存
def evaluate_student(
            teacher_model,
            student_model,
            dev_dataloader,
            ce_loss_fn,
            kl_loss_fn,
        ):
    # 1.设置为评估模式
    student_model.eval()
    # 2.初始化 总损失、总样本数、预测结果列表、真实标签列表
    total_loss = 0.0
    total_samples = 0
    total_preds = []
    total_labels = []
    # 3.遍历数据加载器，分批验证
    for batch in tqdm(dev_dataloader, desc="Evaluating"):
        # batch: (input_ids, attention_mask, labels)
        # input_ids: (batch_size, seq_len), seq_len是当前批次的最大长度
        # attention_mask: (batch_size, seq_len)
        # labels: (batch_size,)
        # 0.迁移数据到设备
        input_ids, attention_mask, labels = [x.to(config.device) for x in batch]
        # 1.前向传播
        # 1.1 教师模型前向传播，获取教师模型的预测分数
        with torch.no_grad():  # 关闭教师模型的梯度计算，节省显存
            teacher_logits = teacher_model(input_ids, attention_mask)  # (batch_size, num_classes)
        # 1.2 学生模型前向传播，获取学生模型的预测分
        student_logits = student_model(input_ids, attention_mask)  # (batch_size, num_classes)

        # 2.计算损失
        # 2.1 计算ce_loss
        ce_loss = ce_loss_fn(student_logits, labels)
        # 2.2 计算kl_loss
        kl_loss = kl_loss_fn(F.log_softmax(student_logits / config.T, dim=-1),
                             F.softmax(teacher_logits / config.T, dim=-1))
        # 2.3 计算总损失
        loss = config.alpha * config.T ** 2 * kl_loss + (1 - config.alpha) * ce_loss
        # # 3.梯度清零
        # optimizer.zero_grad()
        # # 4.反向传播
        # loss.backward()
        # # 5.更新参数
        # optimizer.step()
        # 6.记录结果
        total_loss += loss.item()*input_ids.size(0)
        total_samples += input_ids.size(0)
        y_pred_class = student_logits.argmax(dim=-1).to(device=torch.device('cpu')).tolist()    # (batch_size,)
        total_preds.extend(y_pred_class)
        total_labels.extend(labels.to(device=torch.device('cpu')).tolist())
    # 4.计算 平均损失、准确率、f1分数
    avg_loss = total_loss / total_samples
    acc = accuracy_score(total_labels, total_preds)
    # macro-f1 score: 宏平均F1分数
    f1_avg = f1_score(total_labels, total_preds, average='macro')
    return avg_loss, acc, f1_avg

# 4.3 训练主函数-重点
def train_student():
    # 1.创建数据加载器对象
    train_dataloader, dev_dataloader, _ = build_dataloader()
    # 2.创建模型对象
    # 2.1 加载教师模型对象
    teacher_model = MyBertClassifier().to(config.device)
    teacher_model.load_state_dict(
        torch.load(
            config.bert_model_path,
            weights_only=True, # 只加载模型权重
            map_location=config.device
        ))
    teacher_model.eval()
    # 2.2 创建学生模型对象
    student_model = MyStudentClassifier().to(config.device)
    # 3.定义损失函数和优化器
    # label_smoothing: 软标签比例，代表其他类别所占的比例。
    # 对比原始one-hot编码，label_smoothing会更加柔软。比如label_smoothing=0.1，代表正确类别为0.9+0.1/10，其他类别为0.1/10
    ce_loss_fn = nn.CrossEntropyLoss()
    # kl散度
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    # 优化器
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay)
    # 4.初始化训练过程指标
    # 初始化最优F1分数
    best_f1 = 0.0
    # 可视化指标：训练损失、准确率、f1分数列表，验证损失、准确率、f1分数列表
    train_losses = []
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []
    # 5.开始训练，遍历轮数
    for epoch in range(config.epochs):
        # 1.训练一个轮次
        train_loss, train_acc, train_f1 = train_one_epoch(
            teacher_model,
            student_model,
            train_dataloader,
            ce_loss_fn,
            kl_loss_fn,
            optimizer
        )
        # 2.评估当前模型
        val_loss, val_acc, val_f1 = evaluate_student(
            teacher_model,
            student_model,
            dev_dataloader,
            ce_loss_fn,
            kl_loss_fn,
        )
        # 3.根据验证f1分数来保存模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            # 保存最优模型到模型文件
            torch.save(student_model.state_dict(), config.student_model_path)
            print(f"保存最优模型到文件: {config.student_model_path}, 验证F1分数: {best_f1:.4f}")
        # 4.打印训练指标，损失、准确率、f1分数
        print(f"epoch: {epoch+1}/{config.epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"train_f1: {train_f1:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | "
              f"val_f1: {val_f1:.4f}")
        # 5.记录训练指标到列表中
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
    # 6.返回训练结果
    results = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "train_f1s": train_f1s,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s
    }
    return results

# 4.4 绘制训练曲线
def plot_history(history):
    # 1.设置x轴刻度，epoch轮次
    epochs = range(1, len(history['train_losses'])+1)
    # 2.设置画布大小
    plt.figure(figsize=(15, 5))
    # 3.绘制loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='train_loss')
    plt.plot(epochs, history['val_losses'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    # 4.绘制acc曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accs'], label='train_acc')
    plt.plot(epochs, history['val_accs'], label='val_acc')
    plt.title('Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend()
    # 5.绘制f1曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1s'], label='train_f1')
    plt.plot(epochs, history['val_f1s'], label='val_f1')
    plt.title('F1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)
    plt.legend()

    # 优化布局并显示
    plt.tight_layout()
    plt.show()

# 测试
if __name__ == '__main__':
    # # 1.加载数据集
    # train_data = load_raw_data(config.train_path)
    # dev_data = load_raw_data(config.dev_path)
    # test_data = load_raw_data(config.test_path)
    # print(f"训练集样本数量: {len(train_data)}")
    # print(f"验证集样本数量: {len(dev_data)}")
    # print(f"测试集样本数量: {len(test_data)}")
    # # print(train_data[1])
    # # 2.创建数据集对象
    # train_dataset = MyDataset(train_data)
    # dev_dataset = MyDataset(dev_data)
    # test_dataset = MyDataset(test_data)
    # print(f"训练集对象大小: {len(train_dataset)}")
    # print(f"验证集对象大小: {len(dev_dataset)}")
    # print(f"测试集对象大小: {len(test_dataset)}")
    # 3.构建数据加载器对象
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # print(f"训练集加载器大小: {len(train_dataloader)}") # 180000/64
    # print(f"验证集加载器大小: {len(dev_dataloader)}")
    # print(f"测试集加载器大小: {len(test_dataloader)}")
    # 4.创建模型对象
    # 4.1 创建教师模型对象
    teacher_model = MyBertClassifier().to(config.device)
    print(teacher_model)
    # 4.2 创建学生模型对象
    student_model = MyStudentClassifier().to(config.device)
    print(student_model)
    # 5.模型训练
    results = train_student()
    plot_history(results)
    # 6.模型测试
    # 加载模型文件
    student_model.load_state_dict(torch.load(
        config.student_model_path, # 本地保存的预训练BERT的下游分类模型文件
        weights_only=True, # 只加载模型权重参数，不加载其他对象，确保安全
        map_location=config.device, # 将模型加载到指定设备上，自动适配CPU或GPU，防止训练和推理阶段设备不一致
    ))
    test_loss, test_acc, test_f1 = evaluate_student(
        teacher_model,
        student_model,
        dev_dataloader,
        ce_loss_fn=nn.CrossEntropyLoss(),
        kl_loss_fn=nn.KLDivLoss(reduction='batchmean'),
    )
    print(f"测试集Loss: {test_loss:.4f} | "
          f"测试集Acc: {test_acc:.4f} | "
          f"测试集macro-F1: {test_f1:.4f}")