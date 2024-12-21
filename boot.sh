# 启动嵌入生成服务
cd embedding;uvicorn server:app --port 8787

# 启动向量检索服务
cd faiss;uvicorn server:app --port 8383
