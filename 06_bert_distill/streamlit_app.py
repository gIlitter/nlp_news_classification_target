"""
实现一个streamlit框架的web前端
安装streamlit： pip install streamlit
"""
# 导包
import streamlit as st
import requests
import time
from config import Config
config = Config()
# 1.streamlit 创建画面
# 1.1 设置标题
st.title("投满分项目")
# 1.2 前端输入文本
text = st.text_input("请输入文本：")

# 2.后台发送请求
# 使用columns实现左右布局
col1, col2 = st.columns(2)

# 2.1 准备url1
url1 = f"http://{config.api_host}:{config.api_port}/predict1"
with col1:
    if st.button("bert预测-教师模型"):
        # 2.2 获取当前时刻
        start = time.time()
        try:
            # 2.3 发送请求，获取数据
            r = requests.post(url1, json={"text": text})
            print(f"r: {r.json()}")
            # 2.4 计算耗时
            total_time = (time.time() - start)*1000
            # 2.5 显示预测结果
            st.success(f"预测结果：{r.json()['pred_class']}，耗时：{total_time:.2f}ms")

        except Exception as e:
            st.error(f"出问题了,请联系管理员!")

# 2.2 准备url2
url2 = f"http://{config.api_host}:{config.api_port}/predict2"
with col2:
    if st.button("bert蒸馏预测-学生模型"):
        # 2.2 获取当前时刻
        start = time.time()
        try:
            # 2.3 发送请求，获取数据
            r = requests.post(url2, json={"text": text})
            print(f"r: {r.json()}")
            # 2.4 计算耗时
            total_time = (time.time() - start)*1000
            # 2.5 显示预测结果
            st.success(f"预测结果：{r.json()['pred_class']}，耗时：{total_time:.2f}ms")

        except Exception as e:
            st.error(f"出问题了,请联系管理员!")
