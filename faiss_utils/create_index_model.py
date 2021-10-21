"""
    返回检索模型
"""
import time
import numpy as np
import faiss  # make faiss available


def create_index_model(dimension: int, feature_database: np.array, top: int = 1, nlist:int = 5) -> object:
    """
    生成到排序模型，加载特征库数据+文件训练
    """
    print(f"\n[INFO] 使用倒排文件查询")
    start_time = time.time()
    # todo 聚类中心个数不能少于样本数，需要优化
    index_model = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist, faiss.METRIC_L2)
    assert not index_model.is_trained, f"[INFO] 初始化失败"
    # todo 每一次调用检索时候都需要训练一次模型，考虑将训练模型保存保存为pickle
    # todo 查询结果的时候就加载model，插入特征的时候选择训练+更新权重
    index_model.train(feature_database)  # todo 这个部分是临时使用的功能
    assert index_model.is_trained, f"[ERROR] 创建检索模型异常"
    index_model.add(feature_database)
    print(f"[INFO] 模型到排序耗时=%4.02fs" % (time.time() - start_time))
    return index_model