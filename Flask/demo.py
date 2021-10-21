"""
服务器端
"""
import sys, os
sys.path.insert(0, "./PyRetri")
import base64
from flask import Flask
from flask import request
from flask import abort
from flask import jsonify
import numpy as np
import requests
import cv2

from utils.load_feature import load_feature
from utils.extract_feature import extract_fea
from faiss_utils.faiss_search import faiss_search

from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用cpu提供服务

class Users(Resource):
    def get(self):
        # get请求时将USERS信息返回
        return "欢迎使用特征检索，请使用post方式调用检索api"

    def post(self):
        # 获取验证后的数据
        args = parser.parse_args()
        # 获取传入的数据类型 获取数据内容
        data_type, data = args["type"], args["data"]
        # 调用request 加载原始图片
        with requests.Session() as session:
            if data_type == "url":
                image_buff = session.get(data).content
                np_arr = np.asarray(bytearray(image_buff), np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            elif data_type == "base64":
                # base64 图片
                image_buff = base64.b64decode(data)
                img_array = np.fromstring(image_buff, np.uint8)
                # 转换成opencv可用格式
                image = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)
        # 图片转化为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image, dtype=np.float32)
        # 图片统一尺寸
        image = cv2.resize(image, (224, 224))
        # 提取图片特征 图片为numpy
        query_imgs = extract_fea(query_imgs=image)
        # 特征库路径 特征检索
        feature_file = r"./PyRetri/data/features/cls/gallery/part_0.json"
        dict_keys = search_fea(feature_file, query_imgs)
        print(f"[INFO] {dict_keys}")
        return "POST"


app = Flask(__name__)
api = Api(app, default_mediatype="application/json")

# 1. 创建RequestParser实例
parser = reqparse.RequestParser()

# 2. 添加验证参数
# 第一个参数： 传递的参数的名称
# 第二个参数（location）： 传递参数的方式
# 第三个参数（type）： 验证参数的函数(可以自定义验证函数)
parser.add_argument('url')  # 关键词验证
parser.add_argument('base64')

api.add_resource(Users, '/search')  # 方法来添加路由
api.add_resource(Users, '/insert')  # 方法来添加路由

# 方法的第一个参数是一个类名，该类继承Resource基类，其成员方法定义了不同的HTTP请求方法的逻辑；
# 第二个参数定义了URL路径。在Users类中，我们分别实现了get、post、delete方法，分别对应HTTP的GET、POST、DELETE请求。

app.run(host='0.0.0.0', port=5001, use_reloader=True)
if __name__ == '__main__':
    app.run(port=5001, debug=True)

