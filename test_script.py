import sys, os
sys.path.insert(0, "./PyRetri")
from PIL import Image
from tqdm import tqdm
import cv2
from utils.extract_feature import extract_fea
from utils.search_fea import search_fea
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用cpu提供服务


if __name__ == '__main__':
    img_path = "/data/data2/wufuming/models/PyRetri950541/PyRetri/data/myDatasets/cls/database"
    for r, ds, fs in tqdm(os.walk(img_path)):
        for _name in fs:
            print("----------------------------------------------------------------------------")
            name = os.path.join(r, _name)
            image = cv2.imread(name)
            image = Image.fromarray(image)
            query_imgs = extract_fea(query_imgs=image)
            # 传入图片，自动检索
            best_img_path, probability, used_time = search_fea(query_imgs)
            print(f"[INFO] 预测结果：{best_img_path.split('/')[-1]}   检测图片：{_name.split('/')[-1]}")


