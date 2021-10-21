"""
    提供faiss快速检索服务
"""
import time
import numpy as np
from faiss_utils.create_index_model import create_index_model


def faiss_search(query: list, feature_database: list, top: int = 1) -> tuple:
    feature_database = np.asarray(feature_database, dtype=np.float32)  # 转化为张量
    query = np.asarray(query, np.float32)
    # 增加维度
    if len(query.shape) == 1:
        query = np.expand_dims(query, axis=0)
    assert len(query.shape) == 2, f"[ERROR] 检索特征维度应该为2维矩阵，实际为：{query.shape}"
    # we want to see 4 nearest neighbors
    dimension = feature_database.shape[1]
    # 创建模型
    index_model = create_index_model(dimension, feature_database)
    start_time = time.time()
    distance, top_idx = index_model.search(query, top)  # actual search
    print(f"[INFO] 时间检索耗时=%4.02fs" % (time.time() - start_time))
    return int(top_idx), distance, (time.time() - start_time)  # 相似用户的ID,相似向量的距离,查询耗时
