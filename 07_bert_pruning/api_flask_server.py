"""
演示
    通过Flask组件，构建 路由 + 函数 的应用
"""
# 导包
from flask import Flask, request, jsonify
from bert_predict_fun import predict_fun
from config import Config

# 1.创建APP应用
app = Flask(__name__)
config = Config()
# 2.创建预测接口：路由 + 视图函数
@app.route('/predict', methods=['POST'])
def predict():
    # 1.获取用户请求的数据
    data = request.get_json()
    print(f"data:{data}, type(data):{type(data)}")
    # 2.调用预测函数predict_fun，进行模型预测
    result = predict_fun(data)
    # 3.返回预测结果，使用jsonify()函数来将python对象转为json格式
    return jsonify(result)

# 3.启动APP应用
if __name__ == '__main__':
    app.run(host=config.api_host, port=config.api_port)