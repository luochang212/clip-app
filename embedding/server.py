# -*- coding:utf-8 -*-

"""Embedding 推理服务
    Author: github@luochang212
    Date: 2024-12-01
    Usage: uvicorn server:app --reload
"""

import base64
import model

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image


# 参数初始化
device = model.get_device()  # 设备
text_tokenizer, text_encoder, clip_model, clip_processor = model.init_model(device)

# 初始化嵌入计算函数
get_text_embedding = model.gen_get_text_embedding(text_tokenizer, text_encoder, device)
get_image_embedding = model.gen_get_image_embedding(clip_model, clip_processor, device)
compute_similarity = model.gen_compute_similarity(clip_model, device)

app = FastAPI(debug=True)


class TextRequest(BaseModel):
    texts: list[str]


class ImageRequest(BaseModel):
    names: list[str]
    images_base64: list[str]


class SimilarityRequest(BaseModel):
    a_features: list[list[float]]
    b_features: list[list[float]]


class TextImageSimRequest(BaseModel):
    texts: list[str]


@app.get('/')
def index():
    return {'app_name': 'embedding-server'}


@app.post('/text_embedding')
def text_embedding(request: TextRequest):
    text_features = get_text_embedding(request.texts).cpu().numpy()
    return {'text_features': text_features.tolist()}


@app.post('/image_embedding')
def image_embedding(request: ImageRequest):
    if len(request.names) != len(request.images_base64):
        raise HTTPException(status_code=400, detail="length no equal")

    images = model.to_image(request.images_base64)
    image_features = get_image_embedding(images)

    return {'image_features': image_features.tolist()}


@app.post('/similarity')
def similarity(request: SimilarityRequest):
    probs = compute_similarity(request.a_features,
                               request.b_features)
    return {'probs': probs.tolist()}
