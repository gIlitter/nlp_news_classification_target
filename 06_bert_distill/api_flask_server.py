"""
通过Flask组件，构建 路由 + 预测函数 的应用
"""

# 导包
from flask import Flask, request, jsonify
from bert_predict_fun import predict_fun1, predict_fun2
from config import Config

# 1.创建App应用
app = Flask(__name__)
config = Config()
# 2.创建预测接口(路由 + 预测函数)
# 指定该路由只接受POST请求方法，用于接收客户端发送的数据进行预测处理
# 常见的HTTP请求方法有：GET、POST、PUT、DELETE、PATCH、OPTIONS、HEAD、TRACE
@app.route('/predict1', methods=['POST'])
def predict1():
    # 获取用户请求中的数据
    data = request.get_json()
    print(f"data: {data}, {type(data)}")
    # 调用预测函数
    result = predict_fun1(data)
    return jsonify(result)

@app.route('/predict2', methods=['POST'])
def predict2():
    # 获取用户请求中的数据
    data = request.get_json()
    print(f"data: {data}, {type(data)}")
    # 调用预测函数
    result = predict_fun2(data)
    return jsonify(result)

# 3.启动App应用
if __name__ == '__main__':
    # 支持局域网访问
    app.run(host=config.api_host, port=config.api_port, debug=True)