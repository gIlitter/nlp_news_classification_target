"""
EDA(Exploratory Data Analysis):
    探索性数据分析,帮助发现数据规律或异常，一般集合图标等可视化形式
    就是分析数据，来查看标签数量分布、句子长度分布
"""
# 导包
import pandas as pd # 处理表格数据
from collections import Counter # 统计标签的频次
import matplotlib.pyplot as plt # 可视化工具
import seaborn as sns # 可视化工具
from config import Config # 导入配置类

# 1.初始化配置类
config = Config()
# 2.读取数据
train_data = pd.read_csv(config.train_path, sep="\t", names=["text", "label"])
dev_data = pd.read_csv(config.dev_path, sep="\t", names=["text", "label"])
test_data = pd.read_csv(config.test_path, sep="\t", names=["text", "label"])
print(train_data.shape) # (180k, 2)
print(dev_data.shape)   # (10k, 2)
print(test_data.shape)  # (10k, 2)
print("-"*40)
# 3.统计标签分布
label_counts = Counter(train_data["label"])
print("标签分布:", label_counts)
for label, count in label_counts.items():
    print(f"标签: {label}, 频次: {count}")

# 4.统计句子长度，默认一个字符是一个token,没有进行分词
# 使用apply方式来获取句子长度
# train_data["text_length"] = train_data["text"].apply(len)
# 使用pandas向量化操作str.len()来获取句子长度
train_data["text_length"] = train_data["text"].fillna("").str.len()
print(train_data.head())
print(train_data["text_length"].describe())
# 5.绘制标签分布图
# train_data['label'].hist()
sns.countplot(x="label",data=train_data,hue="label") #对label的不同列使用不同的颜色表示
plt.show()

# 6.绘制句子长度分布图
train_data['text_length'].hist()
# sns.histplot(train_data['text_length'],kde=True)
plt.show()