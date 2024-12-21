# -*- coding: utf-8 -*-

import os
import base64
import requests
import concurrent.futures
import numpy as np
import pandas as pd


BATCH_SIZE = 16


def gen_abspath(
        directory: str,
        rel_path: str
) -> str:
    """
    Generate the absolute path by combining the given directory with a relative path.

    :param directory: The specified directory, which can be either an absolute or a relative path.
    :param rel_path: The relative path with respect to the 'dir'.
    :return: The resulting absolute path formed by concatenating the absolute directory
             and the relative path.
    """
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


def read_csv(
    file_path: str,
    sep: str = ',',
    header: int = 0,
    on_bad_lines: str = 'warn',
    encoding: str = 'utf-8',
    dtype: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file from the specified path.
    """
    return pd.read_csv(file_path,
                       header=header,
                       sep=sep,
                       on_bad_lines=on_bad_lines,
                       encoding=encoding,
                       dtype=dtype,
                       **kwargs)


def get_image_paths(directory, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    """获取指定目录下所有图片文件的路径"""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def chunk_list(lst, sz=BATCH_SIZE):
    return [lst[i:i + sz] for i in range(0, len(lst), sz)]


def process_data(func, lst, max_workers):
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(func, lst)
        for result in results:
            res.append(result)
    return res


class GenEmbedding:

    def __init__(self, base_url="http://127.0.0.1:8787"):
        self.base_url = base_url

    @staticmethod
    def to_base64(img_paths):
        """图片转为 base64"""
        images_base64 = [base64.b64encode(open(pth, "rb").read()).decode("utf-8")
                         for pth in img_paths]
        return images_base64

    def test_home(self):
        """测试首页"""
        response = requests.get(f"{self.base_url}/")
        if response.status_code == 200:
            print("Index Response:", response.json())
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")

    def get_remote_text_features(self, texts):
        payload = {"texts": texts}
        response = requests.post(f"{self.base_url}/text_embedding", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

    def get_remote_image_features(self, img_paths):
        images_base64 = self.to_base64(img_paths)
        payload = {"names": img_paths, "images_base64": images_base64}
        response = requests.post(f"{self.base_url}/image_embedding", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

    def get_similarity(self, a_features, b_features):
        payload = {"a_features": a_features,
                   "b_features": b_features}
        response = requests.post(f"{self.base_url}/similarity", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}


class FaissServer:

    def __init__(self, base_url="http://127.0.0.1:8383"):
        self.base_url = base_url

    def test_home(self):
        """测试首页"""
        response = requests.get(f"{self.base_url}/")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

    def test_info(self):
        """测试获取索引信息"""
        response = requests.get(f"{self.base_url}/info")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

    def add_embedding(self, vid, vector):
        """测试添加向量"""
        payload = {"id": vid, "vector": vector}
        response = requests.post(f"{self.base_url}/add", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

    def search_embedding(self, vector, top_k=5):
        """测试检索向量"""
        payload = {"vector": vector, "top_k": top_k}
        response = requests.post(f"{self.base_url}/search", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")
        return {}

