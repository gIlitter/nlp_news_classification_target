"""
ONNX + TensorRT + INT8 推理优化完整流程

核心流程：
1. PyTorch模型 → 导出ONNX
2. ONNX → TensorRT Engine（高性能推理）
3. TensorRT INT8量化（进一步加速）
"""

from bert_train import MyBertClassifier, build_dataloader
from config import Config
import torch
import torch.onnx
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

config = Config()

# =========================
# 1. 导出 ONNX 模型
# =========================
def export_onnx(model):
    """
    将 PyTorch 模型导出为 ONNX 格式
    """
    model.eval()

    # 构造 dummy 输入（用于推断计算图）
    # ⚠️ 必须指定 dtype=torch.long（BERT要求）
    dummy_input_ids = torch.randint(
        0, 1000,
        (config.batch_size, config.max_len),
        dtype=torch.long
    ).to(config.device)

    dummy_attention_mask = torch.ones(
        (config.batch_size, config.max_len),
        dtype=torch.long
    ).to(config.device)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        "bert_classifier.onnx",
        input_names=["input_ids", "attention_mask"],   # 输入名
        output_names=["logits"],                       # 输出名
        dynamic_axes={                                # 支持动态batch和seq_len
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"}
        },
        opset_version=18,          # ✅ BERT必须使用18（支持LayerNorm）
        do_constant_folding=True,  # 常量折叠优化
    )

    print("✅ ONNX模型导出成功")


# =========================
# 2. ONNX Runtime 推理（基线）
# =========================
import onnxruntime as ort

def onnx_inference(test_dataloader):
    """
    使用 ONNX Runtime 推理（作为基线对比）
    """
    session = ort.InferenceSession(
        "bert_classifier.onnx",
        providers=["CPUExecutionProvider"]  # 可改为CUDAExecutionProvider
    )

    total_preds = []
    total_labels = []

    for batch in tqdm(test_dataloader, desc="ONNX Testing"):
        input_ids, attention_mask, labels = batch

        # 转 numpy + 类型对齐
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64)
        }

        outputs = session.run(None, ort_inputs)
        logits = outputs[0]

        preds = np.argmax(logits, axis=-1)

        total_preds.extend(preds.tolist())
        total_labels.extend(labels.tolist())

    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='macro')

    print(f"[ONNX Runtime] Acc: {acc:.4f}, F1: {f1:.4f}")


# =========================
# 3. TensorRT 推理（核心优化）
# =========================
import tensorrt as trt      # TensorRT 核心库
import pycuda.driver as cuda  # PyCUDA：GPU内存管理（分配/释放/拷贝）
import pycuda.autoinit        # PyCUDA：自动初始化CUDA上下文（必须导入）

# TensorRT 日志器：用于输出构建和推理过程中的警告/错误信息
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ONNX 和 TensorRT Engine 文件路径
ONNX_PATH = "bert_classifier.onnx"
ENGINE_PATH = "bert_classifier.engine"


def build_engine(onnx_path, engine_path, use_fp16=True):
    """
    从 ONNX 模型构建 TensorRT Engine（高性能推理引擎）

    核心流程：
        1. 创建 Builder（构建器）：负责生成优化后的推理引擎
        2. 创建 Network（网络定义）：用于解析ONNX模型的计算图
        3. 使用 OnnxParser 解析ONNX文件，填充Network
        4. 配置优化选项（FP16精度、动态形状等）
        5. 序列化引擎并保存到磁盘

    参数:
        onnx_path: ONNX模型文件路径
        engine_path: 输出的TensorRT Engine文件路径
        use_fp16: 是否启用FP16半精度优化（默认True, 速度提升约2倍）
    """
    # ---- 步骤1: 创建Builder和Network ----
    # EXPLICIT_BATCH: 显式指定batch维度（TensorRT 7+必须使用）
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # ---- 步骤2: 使用OnnxParser解析ONNX文件 ----
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        # 解析ONNX模型，如果失败则打印错误信息
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"❌ ONNX解析错误: {parser.get_error(i)}")
            raise RuntimeError("ONNX解析失败")

    # ---- 步骤3: 配置构建选项 ----
    build_config = builder.create_builder_config()
    # 设置最大工作空间内存: 1GB（TensorRT用于中间计算的临时显存）
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # 启用FP16半精度推理（将FP32运算降为FP16, 精度损失极小, 速度提升明显）
    if use_fp16 and builder.platform_has_fast_fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16模式已启用")

    # ---- 步骤4: 配置动态形状（Dynamic Shape） ----
    # 因为ONNX导出时使用了dynamic_axes, TensorRT需要知道输入的形状范围
    # 设置最小/最优/最大形状，TensorRT会针对最优形状生成最高效的kernel
    profile = builder.create_optimization_profile()
    # input_ids: (batch_size, seq_len), 范围: batch 1~64, seq_len 1~32
    profile.set_shape("input_ids",
                      min=(1, 1),                                   # 最小形状
                      opt=(config.batch_size, config.max_len),      # 最优形状（优化目标）
                      max=(config.batch_size, config.max_len))      # 最大形状
    # attention_mask: 形状与input_ids保持一致
    profile.set_shape("attention_mask",
                      min=(1, 1),
                      opt=(config.batch_size, config.max_len),
                      max=(config.batch_size, config.max_len))
    build_config.add_optimization_profile(profile)

    # ---- 步骤5: 构建并序列化引擎 ----
    # 此步骤耗时较长（数分钟），TensorRT会尝试各种kernel组合找到最优方案
    print("⏳ 正在构建TensorRT Engine（可能需要几分钟）...")
    serialized_engine = builder.build_serialized_network(network, build_config)
    if serialized_engine is None:
        raise RuntimeError("❌ TensorRT Engine构建失败")

    # 将序列化的Engine保存到磁盘（下次可直接加载，无需重新构建）
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ TensorRT Engine已保存到: {engine_path}")

    return serialized_engine


def load_engine(engine_path):
    """
    从磁盘加载已序列化的TensorRT Engine

    参数:
        engine_path: Engine文件路径
    返回:
        反序列化后的 ICudaEngine 对象
    """
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def tensorrt_inference(test_dataloader, engine_path=ENGINE_PATH):
    """
    使用 TensorRT Engine 进行推理

    核心流程：
        1. 加载TensorRT Engine
        2. 创建ExecutionContext（推理上下文）
        3. 对每个batch:
           a. 将输入数据从CPU拷贝到GPU（Host → Device）
           b. 执行推理
           c. 将输出数据从GPU拷贝回CPU（Device → Host）
        4. 计算准确率和F1分数

    参数:
        test_dataloader: 测试集数据加载器
        engine_path: TensorRT Engine文件路径
    """
    # ---- 步骤1: 加载Engine并创建推理上下文 ----
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    total_preds = []
    total_labels = []

    # ---- 步骤2: 遍历测试集进行推理 ----
    for batch in tqdm(test_dataloader, desc="TensorRT Testing"):
        input_ids, attention_mask, labels = batch

        # 转为numpy, 类型对齐为int64（与ONNX导出时一致）
        input_ids_np = input_ids.cpu().numpy().astype(np.int64)
        attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)

        # 获取当前batch的实际形状（最后一个batch可能不足batch_size）
        batch_size_actual, seq_len_actual = input_ids_np.shape

        # ---- 步骤3: 设置动态形状（每个batch可能不同） ----
        # 告诉TensorRT当前batch的实际输入形状
        context.set_input_shape("input_ids", (batch_size_actual, seq_len_actual))
        context.set_input_shape("attention_mask", (batch_size_actual, seq_len_actual))

        # ---- 步骤4: 分配GPU显存 ----
        # Host（CPU）内存: 用于存储输入和输出数据
        # Device（GPU）显存: 用于TensorRT在GPU上执行计算

        # 输入: 在GPU上分配显存，并将CPU数据拷贝过去
        d_input_ids = cuda.mem_alloc(input_ids_np.nbytes)
        d_attention_mask = cuda.mem_alloc(attention_mask_np.nbytes)
        cuda.memcpy_htod(d_input_ids, input_ids_np)            # Host → Device
        cuda.memcpy_htod(d_attention_mask, attention_mask_np)   # Host → Device

        # 输出: 在CPU和GPU上分别分配内存
        # logits形状: (batch_size, num_classes)
        num_classes = len(config.id2class)
        output_shape = (batch_size_actual, num_classes)
        h_output = np.empty(output_shape, dtype=np.float32)     # CPU上的输出缓冲区
        d_output = cuda.mem_alloc(h_output.nbytes)              # GPU上的输出显存

        # ---- 步骤5: 绑定输入输出的GPU地址并执行推理 ----
        # 通过tensor名称绑定GPU内存地址（TensorRT 10+ API）
        context.set_tensor_address("input_ids", int(d_input_ids))
        context.set_tensor_address("attention_mask", int(d_attention_mask))
        context.set_tensor_address("logits", int(d_output))

        # 执行推理（同步模式）
        context.execute_async_v3(stream_handle=0)

        # ---- 步骤6: 将结果从GPU拷贝回CPU ----
        cuda.memcpy_dtoh(h_output, d_output)    # Device → Host

        # 取argmax得到预测类别
        preds = np.argmax(h_output, axis=-1)
        total_preds.extend(preds.tolist())
        total_labels.extend(labels.tolist())

        # ---- 步骤7: 释放GPU显存 ----
        d_input_ids.free()
        d_attention_mask.free()
        d_output.free()

    # ---- 步骤8: 计算评估指标 ----
    acc = accuracy_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='macro')
    print(f"[TensorRT] Acc: {acc:.4f}, F1: {f1:.4f}")


# =========================
# 4. TensorRT INT8 量化（进一步加速）
# =========================
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8量化校准器

    作用：
        TensorRT INT8量化需要校准数据来确定每一层的量化范围（scale和zero_point）。
        校准器会用少量真实数据跑一遍前向传播，统计每层激活值的分布。

    继承 IInt8EntropyCalibrator2：
        使用熵最小化法计算最优量化阈值（效果通常优于MinMax方法）

    参数:
        dataloader: 校准数据加载器（通常用训练集或验证集的子集）
        cache_file: 校准缓存文件路径（避免重复校准）
    """
    def __init__(self, dataloader, cache_file="int8_calibration.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.batch_iter = iter(dataloader)

        # 预分配GPU显存（用于校准数据的输入）
        # 形状: (batch_size, max_len), 类型: int64 (8字节)
        input_nbytes = config.batch_size * config.max_len * 8  # int64 = 8 bytes
        self.d_input_ids = cuda.mem_alloc(input_nbytes)
        self.d_attention_mask = cuda.mem_alloc(input_nbytes)

    def get_batch_size(self):
        """返回校准的batch大小"""
        return config.batch_size

    def get_batch(self, names):
        """
        获取一个校准batch的数据

        TensorRT每次调用此方法获取一批数据进行校准统计。
        返回每个输入tensor在GPU上的指针列表。

        参数:
            names: 输入tensor的名称列表（如["input_ids", "attention_mask"]）
        返回:
            GPU指针列表，或None表示校准数据已遍历完
        """
        try:
            batch = next(self.batch_iter)
            input_ids, attention_mask, _ = batch

            # 转numpy并拷贝到GPU
            input_ids_np = input_ids.cpu().numpy().astype(np.int64)
            attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)
            cuda.memcpy_htod(self.d_input_ids, input_ids_np)
            cuda.memcpy_htod(self.d_attention_mask, attention_mask_np)

            # 返回GPU指针列表（顺序与输入名称对应）
            return [int(self.d_input_ids), int(self.d_attention_mask)]
        except StopIteration:
            # 所有校准数据已遍历完毕
            return None

    def read_calibration_cache(self):
        """读取校准缓存（如果存在则跳过重新校准）"""
        import os
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """保存校准缓存到磁盘"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# =========================
# 5. 主函数
# =========================
if __name__ == "__main__":

    # 1.加载PyTorch模型
    model = MyBertClassifier().to(config.device)
    model.load_state_dict(torch.load(
        config.bert_model_path,
        weights_only=True,
        map_location=config.device,
    ))

    # 2.导出ONNX
    export_onnx(model)

    # 3.加载测试数据
    _, _, test_dataloader = build_dataloader()

    # 4.ONNX Runtime推理（基线对比）
    onnx_inference(test_dataloader)

    # 5.TensorRT FP16 构建 + 推理
    # build_engine(ONNX_PATH, ENGINE_PATH, use_fp16=True)
    # tensorrt_inference(test_dataloader, ENGINE_PATH)
