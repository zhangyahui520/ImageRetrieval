"""
    将新的特征向量插入到已有的特征库文件中
    pickle不支持新内容的追加
        1 读取特征库文件
        2 转化为dict
        3 追加结果
"""
import os
import cv2
import pickle
import shutil
import numpy as np
from utils.load_image import load_image
from utils.extract_feature import extract_fea
from utils.load_feature import load_feature
from faiss_utils.create_index_model import create_index_model


def insert_fea(
    cache_path: str = "./.cache",
    feature_file: str = "./part_0.json",
    save_path: str = "/data/data2/wufuming/models/PyRetri950541/PyRetri/data/myDatasets/cls/database"
) -> object:
    """
    @feature_file：固定路径
    @save_path：缺省值路径不用修改

    1 加载缓存区所有图片到变量
    2 变量图片转化为特征向量
    3 加载全部特征库文件到字典
    4 特征向量插入字典
    5 特征写入文件
    """
    # if feature_file is None or cache_path is None:
    #     assert False, f"[ERROR] 需要指定feature_file与cache_path变量内容"
    # 1 加载图片
    total_images = []  # 存放所有图片 numpy.darray
    total_images_new_path_list = []  # 与total_images一一对应存放图片转移后的新路径
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
        return  # 缓存文件夹不存在就结束任务
    for single_img in os.listdir(cache_path):
        dsy = os.path.join(save_path, single_img)  # 图片的保存路径
        single_img_path = os.path.join(cache_path, single_img)
        total_images.append(load_image(single_img_path))  # 提取图片特征
        total_images_new_path_list.append(dsy)
        shutil.move(single_img_path, dsy)  # old->new 从缓存文件夹中将图片转移到归纳文件夹中
    # 2 变量图片转化为特征向量
    total_images_fea = []
    for single_img in total_images:
        # 图片统一尺寸224*224 图片转化为特征向量 ResNet提取
        total_images_fea.append(extract_fea(query_imgs=cv2.resize(single_img, (224, 224))))
    # 3 加载全部特征库文件到字典
    with open(feature_file, 'rb') as temp_file:
        database_fea_dict = pickle.load(temp_file)
    # 4 特征向量插入字典
    # temp_dict["info_dicts"] 是一个list，元素是字典
    # 字典关键词dict_keys(['nr_class', 'path_type', 'info_dicts'])
    # dict_keys(['path', 'label', 'label_idx', 'feature', 'idx'])
    for path, fea in zip(total_images_new_path_list, total_images_fea):
        database_fea_dict["info_dicts"].append(
            {
                "path": path,
                "feature": {
                    "pool5_GeM": fea  # PyRetri格式规范
                }
            }
        )
    # 重新写入文件 备份特征信息
    with open(feature_file, 'wb') as temp_file:
        pickle.dump(database_fea_dict, temp_file)
    # 5 生成新的、训练好的index_model模型
    feature_database, img_path_list = load_feature(feature_file)
    feature_database = np.asarray(feature_database, dtype=np.float32)  # 转化为张量
    index_model = create_index_model(feature_database.shape[1], feature_database)
    return index_model


if __name__ == '__main__':
    with open("../part_0.json", 'rb') as temp_file:
        temp_dict = pickle.load(temp_file)
    print(temp_dict["info_dicts"][0].keys())