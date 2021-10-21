"""
    对传入的特征调度faiss检索匹配
"""

from utils.load_feature import load_feature
from faiss_utils.faiss_search import faiss_search

from PyRetri.pyretri.index import feature_loader

import numpy as np
import time


def softmax(x: np.array, axis: int = 1) -> np.array:
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    probability = x_exp / x_sum
    return probability


def search_fea(
    query_imgs: list,
    feature_file: str = "/data/data2/wufuming/models/PyRetri950541/PyRetri/data/features/cls/gallery/part_0.json"
) -> tuple:
    """
    query_imgs: 图片图片特征
    feature_file: 特征库，缺损值路径
    return: 返回图片的检索结果
    """
    start_time = time.time()

    print(f"[INFO] 单张图片转化为特征向量耗时=%4.02fs" % (time.time() - start_time))
    # 提取特征
    start_time = time.time()
    feature_database, img_path_list = load_feature(feature_file)
    print(f"[INFO] 从pickle数据文件中加载所有特征耗时=%4.02fs" % (time.time() - start_time))
    # 调用特征检索功能 传入特征库
    start_time = time.time()
    top_idx, distance, used_time = faiss_search(query=query_imgs, feature_database=feature_database)
    print(f"[INFO] 特征检索耗时=%4.02fs" % (time.time() - start_time))
    # 将top_idx转化为图片名称
    print(distance)
    best_img_path = img_path_list[top_idx]
    # 使用softmax对distance归一化
    probability = 1 - float(softmax(distance))
    print(f"[INFO] 相似度概率probability=%0.2f%%, 耗时=%4.02fs" % (probability, time.time()-start_time))
    # todo 这个位置增加概率阈值筛选 threshold
    if probability <= 1:  # 小于阈值threshold的时候就返回结果
        return best_img_path, probability, used_time
    else:
        return None, None, used_time