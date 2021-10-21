from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse


class Users(Resource):
    def get(self):
        # get请求时将USERS信息返回
        return "Hello"

    def post(self):
        # 获取验证后的数据
        args = parser.parse_args()
        print(args.keys())
        return args['url']

    def delete(self):
        USERS = []
        return jsonify(USERS)


app = Flask(__name__)
api = Api(app, default_mediatype="application/json")

# 1. 创建RequestParser实例
parser = reqparse.RequestParser()

# 2. 添加验证参数
# 第一个参数： 传递的参数的名称
# 第二个参数（location）： 传递参数的方式
# 第三个参数（type）： 验证参数的函数(可以自定义验证函数)
parser.add_argument('base64')
parser.add_argument('url')

# 3. 验证数据
# args是一个字典

api.add_resource(Users, '/test')  # 方法来添加路由

# 方法的第一个参数是一个类名，该类继承Resource基类，其成员方法定义了不同的HTTP请求方法的逻辑；
# 第二个参数定义了URL路径。在Users类中，我们分别实现了get、post、delete方法，分别对应HTTP的GET、POST、DELETE请求。

app.run(host='127.0.0.1', port=5001, use_reloader=True)
if __name__ == '__main__':
    app.run(port=5001, debug=True)
