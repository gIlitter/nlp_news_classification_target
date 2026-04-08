"""
实现一个streamlit框架的web前端
需要安装streamlit库: pip install streamlit
"""

# 导包
import streamlit as st
import requests
import time
from config import Config

# 0.创建配置对象
config = Config()

# 1.streamlit页面
# 1.1 设置title
st.title("投满分项目(文本分类系统)")
# 1.2 输入文本
text = st.text_input("请输入文本：")

# 2.向后端发送请求
# 2.1 构造url
url = f"http://{config.api_host}:{config.api_port}/predict"
# 2.2 设置 预测 按钮
if st.button("fasttext预测"):
    try:
        # 0.记录开始时间
        start_time = time.time()
        # 1.使用requests来发送请求体
        data = {"text": text}
        r = requests.post(url, json=data)
        # 2.打印结果
        print(f"预测结果：{r.json()}")
        total_time = (time.time() - start_time)*1000
        # 3.在前端界面显示预测结果
        st.success(f"预测结果：{r.json()['pred_class']} | "
                   f"推理时长：{total_time:.2f}ms")

    except Exception as e:
        st.error(f"出问题了,请联系管理员!")
        print(f"出问题了,请联系管理员, {e}!")
