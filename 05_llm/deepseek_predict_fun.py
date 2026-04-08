"""
演示
    使用大模型的提示词工程来完成文本分类任务 的 预测函数

注意:
    1.系统提示词也是大模型生成的，我们只需要写提示词的提示词；
    2.也就是我们用大模型生成系统提示词
    3.注意：尽可能把任务描述清楚，最好能给出数据示例
"""
import os
from openai import OpenAI   # 安装openai: pip install openai
# 0.加载环境变量，使用dotenv库: pip install dotenv
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
你是专业文本分类器。请将输入的文本严格分类为以下10个类别之一：
finance, realty, stocks, education, science, society, politics, sports, game, entertainment

【示例】
输入：鲁比尼：美国与英国经济已陷入衰退 → 输出：finance
输入：美国商学院学生语录：康奈尔大学等 → 输出：education
输入：东5环海棠公社230-290平2居准现房98折优惠 → 输出：realty
输入：卡佩罗：告诉你德国脚生猛的原因 → 输出：sports

【强制规则】
1. 仅输出1个类别名称，**不允许编号、解释、分析、标点、空格、换行**
2. 输出必须完全匹配上述列表中的某一个单词
3. 禁止任何额外内容
"""
# 定义函数，调用DeepSeek API获取回复
def call_deepseek_api(user_prompt, system_prompt="You are a helpful assistant"):
    # 1.创建OpenAI客户端，使用环境变量中的API Key
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    # 2.调用聊天接口，发送消息并获取回复
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

# 定义函数，实现预测过程：输入文本字典，返回带有预测结果的字典
def predict_fun(data):
    """
    预测函数，输入文本字典，返回带有预测结果的字典
    :param data: 文本字典，格式为{"text": "待分类的文本内容"}
    :return: 带有预测结果的字典，格式为{"text": "待分类的文本内容", "pred_class": "预测类别名称"}
    """
    # 1.获取文本
    text = data["text"]
    # 2.调用call_deepseek_api函数，获取预测结果
    pred_class = call_deepseek_api(text, system_prompt=SYSTEM_PROMPT)
    # 3.将预测结果添加到字典中，返回字典
    data["pred_class"] = pred_class
    return data
# 测试
if __name__ == "__main__":
    data = {"text": "我叫胡欢，是一个AI大模型算法工程师，喜欢跳舞"}
    result = predict_fun(data)
    print(result)
