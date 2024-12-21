# -*- coding:utf-8 -*-

"""Embedding 推理服务
    Author: github@luochang212
    Date: 2024-12-01
    Usage: uvicorn server:app --reload
"""

import os
import base64
import numpy as np
import clip
import torch

from io import BytesIO
from PIL import Image
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel


# 初始化
MODEL_PATH = '../model'
IMG_PATH = '../img'
CN_CLIP_PATH = 'Taiyi-CLIP-Roberta-102M-Chinese'
VIT_CLIP_PATH = 'clip-vit-base-patch32'


def to_image(images_base64):
    """base64 转为图片"""
    images = [Image.open(BytesIO(base64.b64decode(base64_string)))
              for base64_string in images_base64]
    return images


def get_url_raw(url):
    """获取图像 url 内容"""
    return requests.get(url, stream=True).raw


def gen_abspath(directory, rel_path):
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def init_model(device):
    # 加载 IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
    clip_model_path = gen_abspath(MODEL_PATH, CN_CLIP_PATH)
    text_tokenizer = BertTokenizer.from_pretrained(clip_model_path)
    text_encoder = BertForSequenceClassification.from_pretrained(clip_model_path).to(device).eval()

    # 加载 openai/clip-vit-base-patch32
    vit_model_path = gen_abspath(MODEL_PATH, VIT_CLIP_PATH)
    clip_model = CLIPModel.from_pretrained(vit_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(vit_model_path)

    return text_tokenizer, text_encoder, clip_model, clip_processor


def get_text_embedding(texts, text_tokenizer, text_encoder, device):
    """计算文本 Embedding"""
    inputs = text_tokenizer(texts, return_tensors='pt', padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        text_features = text_encoder(**inputs).logits

    # 归一化
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


def gen_get_text_embedding(text_tokenizer, text_encoder, device):
    def func(texts,
             text_tokenizer=text_tokenizer,
             text_encoder=text_encoder,
             device=device):
        res = get_text_embedding(texts, text_tokenizer, text_encoder, device)
        return res
    return func


def get_image_embedding(images, clip_model, clip_processor, device):
    """计算图像 Embedding"""
    inputs = clip_processor(images=images, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features


def gen_get_image_embedding(clip_model, clip_processor, device):
    def func(images,
             clip_model=clip_model,
             clip_processor=clip_processor,
             device=device):
        res = get_image_embedding(images, clip_model, clip_processor, device)
        return res
    return func


def compute_similarity(feature, features, clip_model, device):
    """计算图文嵌入余弦相似度"""
    feature = torch.tensor(feature, device=device)
    features = torch.tensor(features, device=device)

    with torch.no_grad():
        logit_scale = clip_model.logit_scale.exp()  # logit_scale 是尺度系数
        logits_per_image = logit_scale * feature @ features.t()
        logits_per_text = logits_per_image.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return np.around(probs, 3)


def gen_compute_similarity(clip_model, device):
    def func(feature, features, clip_model=clip_model, device=device):
        res = compute_similarity(feature, features, clip_model, device)
        return res
    return func
