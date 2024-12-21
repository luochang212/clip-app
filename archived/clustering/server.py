# -*- coding:utf-8 -*-

"""聚类服务
    Author: github@luochang212
    Date: 2024-12-01
    Usage: uvicorn server:app --reload --port 9000
"""

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from river import cluster, stream


# 初始化
N_SAMPLES_INIT = 10  # 最少训练样本数
ndict = dict()  # 存储历史簇号

# 初始化 DenStream 聚类器
denstream = cluster.DenStream(decaying_factor=0.01,
                              beta=0.5,
                              mu=2.5,
                              epsilon=0.5,
                              n_samples_init=N_SAMPLES_INIT)


app = FastAPI(debug=True)


class Query(BaseModel):
    id: str
    vector: list[float]


@app.get('/')
def index():
    return {'app_name': 'clustering-server'}


@app.post("/train")
def train_cluster(query: Query):
    """
    增量学习一个新的数据点，更新 DenStream 聚类器
    """
    global denstream

    # 学习新数据点
    x = {i: e for i, e in enumerate(query.vector)}
    denstream.learn_one(x)
    prediction = denstream.predict_one(x)

    ndict[query.id] = prediction

    return {"id": query.id, "cluster": prediction}


@app.post('/search')
def search_cluster(query: Query):
    global ndict

    # 获取历史数据点所属的簇
    if query.id not in ndict:
        raise HTTPException(status_code=400, detail="Cannot find the vector.")

    return {
        "id": query.id,
        "cluster": ndict.get(query.id)
    }
