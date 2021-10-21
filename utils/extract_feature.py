# -*- coding: utf-8 -*-

import argparse
import os
from PIL import Image
import numpy as np
import cv2
from PyRetri.pyretri.config import get_defaults_cfg, setup_cfg
from PyRetri.pyretri.datasets import build_transformers
from PyRetri.pyretri.models import build_model
from PyRetri.pyretri.extract import build_extract_helper


def extract_fea(
    query_imgs: np.array = None,
    config_file: str = "./config/cfg.yaml"
) -> list:
    assert query_imgs is not None, f"[ERROR] 检索对象必须是图片"
    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    # todo 这个模型可以后台一直运行，不用每次都启动，推荐使用trt推理
    # build transformers
    transformers = build_transformers(cfg.datasets.transformers)

    # build model
    model = build_model(cfg.model)

    # build helper and extract feature for single image
    extract_helper = build_extract_helper(model, cfg.extract)

    # 图片转化为张量
    img_tensor = transformers(query_imgs)
    img_fea_info = extract_helper.do_single_extract(img_tensor)
    return img_fea_info[0]["pool5_GeM"]