这是一份README **补充说明文档**，解释该项目结构01_data文件夹下bert-base-chinese和save_model为什么是空文件夹的原因和细节说明。

**⚠️ 重要提示：关于大文件上传**
由于 Git 版本控制系统对单个文件大小有限制（通常建议 < 50MB，最大 < 100MB），而本项目中的模型文件体积较大（单文件最高约 400MB），**这些文件未直接提交至 Git 仓库**。

#### 1. 🧠 预训练模型 (`bert-base-chinese`)

此目录存放 Hugging Face 官方的 `bert-base-chinese` 预训练权重及配置。它是下游任务的基础底座。

**状态**：❌ 未上传

- 主要文件解析
  - `model.safetensors` (~400MB): 模型权重文件 (Safetensors 格式，推荐用于安全加载)。
  - `config.json`: 模型超参数配置 (隐藏层维度、注意力头数等)。
  - `tokenizer.json` / `vocab.txt`: 分词器配置与词表。
- **用途**：用于提取文本特征，作为分类任务的输入层。

#### 2. 🛠️ 微调后模型 (`save_model`)

此目录存放经过新闻分类任务微调（Fine-tuning）后的各种变体模型。根据实验需求选择加载不同的文件。

**状态**：❌ 未上传

**文件列表与用途**：

| 文件名                 | 类型    | 大小估算     | 用途说明                                                     |
| ---------------------- | ------- | ------------ | ------------------------------------------------------------ |
| `bert_model.pt`        | PyTorch | ~390 MB      | 标准微调模型：基础 BERT 在新闻数据集上的完整权重。           |
| `bert_pruning.pt`      | PyTorch | ~390 MB      | 剪枝模型：经过结构剪枝优化的模型，旨在减少计算量。           |
| `bert_quantization.pt` | PyTorch | ~140 MB      | 量化模型：FP16/INT8 量化版本，显著减小体积并加速推理。       |
| `ft_model_char_*.bin`  | Bin     | 78MB - 355MB | 字符级微调模型：基于字符粒度的变体模型 (Char-based)。        |
| `rf_model.pkl`         | Pickle  | ~347 MB      | 随机森林模型：非神经网络方案，基于传统机器学习算法的基线对比。 |
| `student_model.pt`     | PyTorch | ~27 MB       | 蒸馏学生模型：知识蒸馏后的轻量级模型，用于端侧部署。         |
| `tfidf_model.pkl`      | Pickle  | < 1 MB       | TF-IDF 基线：最基础的 NLP 基线模型。                         |

### 🔄 环境初始化指南 (Setup Instructions)

为了让项目可复现，在项目根目录提供一个初始化脚本 `setup_models.py`。