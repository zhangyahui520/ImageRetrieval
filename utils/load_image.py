"""
根据url或者base64加载对应的图片
"""
import cv2
import base64
import requests
import numpy as np


def load_image(data_type: str, data: str) -> np.array:
    """
    调用request 加载原始图片
    return: RGB_image(原始尺寸)
    """
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
    return image