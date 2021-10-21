"""
    功能说明：
        服务器端部署
        Flask提供RESTful接口，api传入图片base64或者url,api快速检索并返回最佳匹配结果
    本共项目由PyRetri+Faiss+Flask构建，预计能实现百万级规模检索快速响应，后续逻辑需要进一步优化
"""
import sys, os
sys.path.insert(0, "./PyRetri")
import socket
import time
import argparse
import cv2

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

from utils.extract_feature import extract_fea
from utils.load_image import load_image
from utils.search_fea import search_fea
from utils.insert_feature import insert_fea
from apscheduler.schedulers.blocking import BlockingScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用cpu提供服务
opt_types = ["insert", "search"]


def get_ip():
    """获取本机的ip地址"""
    return socket.gethostbyname(socket.gethostname())


class ReSearchApi(Resource):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get():
        return "Welcome to Image Feature Fetrieval. Call the retrieval API using POST"

    def post(self, opt_type):
        print(f"[INFO] 请求类型：{opt_type}")
        if opt_type not in opt_types:
            return "Welcome to Image Feature Fetrieval. Call the retrieval API using POST with 'search' or 'insert'"
        # 获取验证后的数据
        args = parser.parse_args()
        # 获取传入的数据类型 获取数据内容
        data_type, data = args["type"], args["data"]
        print(f"[INFO] data_type:{data_type}")
        # 加载图片 返回原始图片
        image = load_image(data_type, data)
        if opt_type == "search":
            try:
                # 图片转化为特征向量
                query_imgs = extract_fea(query_imgs=image)
                # 特征检索
                best_img_path, probability, used_time = search_fea(query_imgs)
                print(f"[INFO] 最匹配的图片：{best_img_path}")
                print(f"[INFO] 匹配率：{probability}")
                print(f"[INFO] 耗时：{used_time}")
                return best_img_path
            except Exception as err:
                return err
        elif opt_type == "insert":
            # 直接保存原始图片到缓存区
            os.makedirs("./.cache/", exist_ok=True)
            name_indx = len(os.listdir("./cache/"))
            cv2.imwrite(f"./.cache/{int(time.time())}_{name_indx}.jpg", image)
            return "done"
        else:
            return "ERROR!"


app = Flask(__name__)
api = Api(app, default_mediatype="application/json")

parser = reqparse.RequestParser()
parser.add_argument('type')
parser.add_argument('data')
# flask增加不同url请求响应功能
api.add_resource(ReSearchApi, '/<string:opt_type>')  # 方法来添加路由
app.run(host="0.0.0.0", port=5001, use_reloader=True)


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--temp', '-t', default=None, type=str, help='待用')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sched = BlockingScheduler()
    sched.add_job(insert_fea, 'cron', day_of_week='1-5', hour=2, minute=0)
    sched.start()
    app.run(port=5001, debug=True)


