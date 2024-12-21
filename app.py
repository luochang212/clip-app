# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

import utils

from pathlib import Path
from PIL import Image


IMG_PATH = 'img'
BATCH_SIZE = 16


ge = utils.GenEmbedding()  # 嵌入服务
fs = utils.FaissServer()  # 向量搜索服务


def save_to_faiss(image_paths, max_workers=2):
    chunked_list = utils.chunk_list(image_paths)
    image_responses = utils.process_data(func=ge.get_remote_image_features,
                                         lst=chunked_list,
                                         max_workers=max_workers)
    image_features = [ee for e in image_responses for ee in e.get('image_features')]

    # 将图片向量灌入 Faiss
    for img_fp, img_embd in zip(image_paths, image_features):
        fs.add_embedding(img_fp, img_embd)

    return fs.test_info()


def get_top_k_images(text: str, top_k: int = 3):
    texts = [text]

    # 获取文本嵌入
    text_response = ge.get_remote_text_features(texts=texts)
    text_features = text_response.get('text_features')

    # 用 Faiss 取回 top k 向量
    search_response = fs.search_embedding(vector=text_features[0], top_k=top_k)
    results = search_response.get('results')
    search_paths = [e.get('id') for e in results]

    return search_paths


def show_pictures(search_paths):
    num_images = len(search_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axes = [axes]

    for i, image_path in enumerate(search_paths):
        fn = Path(image_path)
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].set_title(fn.name)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    directory_path = utils.gen_abspath('./', IMG_PATH)
    image_paths = utils.get_image_paths(directory_path)

    save_to_faiss(image_paths, max_workers=2)
    search_paths = get_top_k_images(text='两眼一白', top_k=3)
    show_pictures(search_paths)
