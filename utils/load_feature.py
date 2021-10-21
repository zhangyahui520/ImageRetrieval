"""
    提取特征数据库的 图片特征暂时使用pickle文件存储
    后期考虑使用lmdb 数据库存储等
"""
import pickle


def load_feature(
    feature_file: str = None
) -> (list, list):
    """
    feature_file：特征文件路径
    model.keys()：
        nr_class
        path_type
        info_dicts
    model["info_dicts"]中特征存储方式：
        dict_keys(['path', 'label', 'label_idx', 'feature', 'idx'])
    预处理特征文件，将特征文件重新打包为一个字典
    feature_dict={feature:img_path}
    """
    assert isinstance(feature_file, str), f"[ERROR] {feature_file}类型不是string"
    with open(feature_file, 'rb') as file:
        feature_json_dict = pickle.load(file)
    feature_database = list()  # 存储特征
    img_path_list = list()  # 存储特征对应的图片路径
    # 解析这个特征文件
    for sub_dict in feature_json_dict["info_dicts"]:
        img_path_list.append(sub_dict["path"])
        feature_database.append(sub_dict["feature"]['pool5_GeM'])
    return feature_database, img_path_list
