# -*- coding:utf-8 -*-

"""向量检索服务
    Author: github@luochang212
    Date: 2024-12-01
    Usage: uvicorn server:app --reload
"""

import faiss
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


DIMENSION = 512


app = FastAPI()
index = faiss.IndexFlatL2(DIMENSION)
embedding_data = dict()


class AddEmbeddingRequest(BaseModel):
    id: str
    vector: list[float]


class SearchEmbeddingRequest(BaseModel):
    vector: list[float]
    top_k: int


@app.get('/')
def home():
    return {'app_name': 'faiss-server'}


@app.on_event("startup")
def load_index():
    """
    在服务启动时加载索引（如果有需要持久化索引）
    """
    global index
    try:
        # 从文件加载索引 (可选，需事先保存索引)
        # index = faiss.read_index("faiss_index.index")
        pass
    except Exception as e:
        print(f"Failed to load index: {e}")


@app.get("/info")
def get_index_info():
    """
    返回索引的信息，例如已存储向量数量
    """
    global index
    return {"total_vectors": index.ntotal}


@app.post("/add")
def add_embedding(request: AddEmbeddingRequest):
    global index, embedding_data

    if len(request.vector) != DIMENSION:
        raise HTTPException(status_code=400, detail="Vector dimension mismatch!")
    if request.id in embedding_data:
        raise HTTPException(status_code=400, detail="ID already exists!")
    
    vector_np = np.array(request.vector).astype('float32').reshape(1, -1)
    index.add(vector_np)
    embedding_data[request.id] = request.vector

    return {"id": request.id}


@app.post("/search")
def search_embedding(request: SearchEmbeddingRequest):
    global index, embedding_data

    if len(request.vector) != DIMENSION:
        raise HTTPException(status_code=400, detail="Vector dimension mismatch!")

    query_np = np.array(request.vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_np, request.top_k)

    # 提取所有键
    keys = list(embedding_data.keys())

    # 构造结果
    results = [
        {
            "id": keys[idx],
            "vector": list(map(float, embedding_data[keys[idx]])),
            "distance": float(distances[0][i])
        }
        for i, idx in enumerate(indices[0])
        if idx < len(keys)
    ]

    return {"results": results}
