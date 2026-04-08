# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI   # 安装openai: pip install openai
# 0.加载环境变量，使用dotenv库: pip install dotenv
from dotenv import load_dotenv
load_dotenv()

# 1.创建OpenAI客户端，使用环境变量中的API Key
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# 2.调用聊天接口，发送消息并获取回复
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "我是杰哥，是一名AI讲师，帮我写一个赞美我的诗歌，注意我又帅又年轻"},
    ],
    stream=False
)

print(response.choices[0].message.content)