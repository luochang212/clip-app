# clip-app

> 在手机相册搜索框中输入“笔记本电脑”时，即使照片本身并未包含“笔记本电脑”这几个字，相关图片依然能够被精准地检索出来。这说明现代相册 APP 不仅使用图片 OCR 文本来召回搜索结果。它综合使用了多种技术，包括多模态技术。

本文使用多模态模型 CLIP 搭建一个简单的『以文搜图』应用，实现与相册 APP 类似的搜索效果。

本文涉及的内容包括：

- 用 FastAPI 搭建图文向量生成服务
- 用 FastAPI 搭建 Faiss 向量搜索服务
- 集成以上两个服务，实现『以文搜图』应用
- 使用 `@torch.inference_mode()` 装饰器，优化推理性能

✨ 如果你对多模态模型的效果有更高的要求，可以尝试 <a href="https://huggingface.co/docs/transformers/en/model_doc/blip-2" target="_blank">BLIP-2</a>。

## 一、加载 CLIP 模型

加载 CLIP 模型，生成图文 Embedding。

1. 获取文本 Embedding
2. 获取图像 Embedding
3. 多条文本对一张图片
4. 一条文本对多张图片
5. 多条文本对多张图片


## 二、Faiss 向量搜索

用 FastAPI + Faiss 写一个向量检索服务 [faiss/server.py](/faiss/server.py)，该服务提供两个接口：

|路由名|描述|
| -- | -- |
|`add`|用于向索引中添加新的 Embedding|
|`search`|用于从索引中取回给定 Embedding 的最近邻 Embedding|


## 三、CLIP 向量生成

用 FastAPI 写一个简单的 CLIP Embedding 生成服务 [embedding/server.py](/embedding/server.py)。接口输入是图片、文本，输出是图文 Embedding。该服务提供 3 个 API 接口：

|路由名|描述|
| -- | -- |
|`text_embedding`|用于获取文本的 Embedding|
|`image_embedding`|用于获取图片的 Embedding，输入是图片的 base64 编码|
|`similarity`|计算 Embedding 的相似度，输入是两个 Embedding 列表，输出是概率矩阵|

1. 获取文本嵌入
2. 获取图片嵌入
3. 计算图文相似度


## 四、以文搜图

本节我们跑通以文搜图功能。

1. 将图片向量存入 Faiss
2. 获取文本嵌入
3. 取回 top k 相似图片
4. 用 `get_similarity` 验证 Faiss 取回结果是否正确


## 五、功能集成

经过前几节的探索，现在我们有足够的经验搭建一个『以文搜图』应用了。

搭建应用的步骤归纳如下：

1. 计算图片 Embedding，并将 Embedding 存入 Faiss 索引
2. 将 Query 转成 Query Embedinng，用 Faiss 索引取回 Query Embedding 的 TOP K 最近邻图片
3. 将上一步获取的图片展示给用户

遵循以上步骤，编写以下功能集成函数，并存放到 [app.py](/app.py)

|函数|说明|
| -- | -- |
|`save_to_faiss`|将图片向量化后存入 Faiss|
|`get_top_k_images`|从 Faiss 取回 Query 向量的 TOP K 最近邻图片向量，同时取回对应的图片路径|
|`show_pictures`|将获取的图片展示给用户|

使用应用前，需要先启动 CLIP 向量生成服务 和 Faiss 向量搜索服务。

```
# 启动嵌入生成服务
cd embedding; uvicorn server:app --port 8787

# 启动 Faiss 向量搜索服务
cd faiss; uvicorn server:app --port 8383
```
