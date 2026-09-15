# 第二部分实施设计: 多模态模型支持与多模态密集向量后端 (v2)

> 状态: 设计定稿, 待开发 | 日期: 2026-09-15
> 基线: `MULTIMODAL_MILVUS_MIGRATION.md` §4 (v1 需求) — 本文细化并部分**收窄/覆盖** v1; 冲突处以本文为准。
> 调研来源: [rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr) (MIT, 现更名 dots.mocr) 源码级分析 (parser.py / model/inference.py / utils/prompts.py)。

## 0. 硬约束 (用户决策, 不可违反)

1. **零改动边界**: 现有文本 RAG 密集向量链路 (`tools/vectorizer.py`、`tools/retrieval_backends/dense_milvus.py`、`rag_nodes` collection、`/agent/api/ask` 管线) **一行不改**。
2. **新建多模态密集向量实现**: 独立文件实现 `DenseBackend` 协议, 操作新 collection `rag_multimodal`。
3. **env 切换**: `DENSE_BACKEND=milvus_multimodal` 切换到多模态密集向量; 默认 `milvus` 保持现状。
4. 依赖边界不变 (v1 §3.4): 不引入 LangChain 全家; Milvus 读写仅 `pymilvus`。

## 1. dots.ocr 代码级借鉴清单

官方仓库结构 `dots_ocr/{parser.py, model/inference.py, utils/{prompts,image_utils,doc_utils,layout_utils,format_transformer,output_cleaner}.py}`。逐项映射:

| dots.ocr 实现 (已读源码) | 本项目落点 | 借鉴方式 |
|---|---|---|
| `inference_with_vllm`: OpenAI 兼容客户端, `base_url=f"{protocol}://{ip}:{port}/v1"`, 消息= `image_url`(PIL→base64 data URI) + text block; 文本需带 `<\|img\|><\|imgpad\|><\|endofimg\|>` 前缀(vLLM v1 防换行 hack) | 新 `tools/dots_ocr_client.py` | **复用项目现有 OpenAI 兼容客户端**, 不引 `dots_ocr` 包; base64 data URI + imgpad 前缀照抄 |
| `DotsOCRParser` 参数: `temperature=0.1, top_p=1.0, max_completion_tokens=16384, dpi=200, ThreadPool(num_thread) 逐页并行, 结果按 page_no 排序` | 同上 | 参数与并行模型照抄; 线程数= `DOT_OCR_MAX_THREADS`(默认 16) |
| `prompt_layout_all_en` 输出协议: 单 JSON, 元素 `{bbox:[x1,y1,x2,y2], category, text}`; 11 类 category; Picture 无 text; Formula→LaTeX; Table→HTML; 其余 Markdown; 阅读序排序 | 同上 (常量内嵌) | 协议照抄; 解析失败降级路径 (存原始响应, `filtered=True`) 照抄 |
| `layoutjson2md` (layout JSON → 顺序 md) + `_nohf` 去页眉页脚变体 | 同上 (自实现 ~80 行) | 效果复刻: Table 的 HTML 直接内嵌, Picture 占位 `![](image_{n}.jpg)` 供插图抽取 |
| `load_images_from_pdf(dpi=200)` fitz 渲染 | 同上 | 照抄 dpi 语义 |
| `docker/docker-compose.yml` vLLM 部署 | `docs/` 部署节摘录命令 | 仅文档化, 不并入项目 compose (多模态服务可独立启停) |
| HF 本地推理分支 (`use_hf`, flash_attention_2) | **不借鉴** | vLLM 服务化是唯一路径; 单机单卡跑解析+检索不现实 |

**不引 `dots_ocr` 包的理由**: 该包核心价值 = vLLM 调用(3 行 OpenAI 客户端) + prompt 常量 + md 转换, 均为百行内可自实现; 引入将拖进 `qwen_vl_utils`/`torch` 依赖树。需求: 解析层只依赖 `openai`(已有) + `fitz`(已有) + `Pillow`(已有)。

## 2. 多模态模型支持矩阵

"多模态 model 支持" = 三类模型均可经 env 更换, 各自接口协议:

| 角色 | 默认 | env | 协议 | 可换范围 |
|---|---|---|---|---|
| **解析 VLM** | vLLM 服务上的 dots.ocr/dots.mocr | `DOT_OCR_BASE_URL` + `DOT_OCR_MODEL` | OpenAI 兼容 `/v1/chat/completions`, image_url block, 输出=layout JSON 协议(§1) | 任何遵循该 prompt 协议的 vLLM 模型(dots 系/微调版); 换协议模型需改 `dots_ocr_client.py` |
| **多模态 embedding** | DashScope `multimodal-embedding-one-peace-v1` | `MULTIMODAL_EMBEDDING_MODEL` / `_DIM`(0=首响应自动探测) | DashScope SDK: text 块 `{text}`, image 块 `{image: base64, text}` | 参考项目实测 `tongyi-embedding-vision-flash-*` 亦可用; 换模型必须核维度 |
| **图片描述 VLM** | `qwen-vl-plus` | `MULTIMODAL_VLM_MODEL` | 项目现有 OpenAI 兼容客户端 + image_url block | 任意 OpenAI 兼容 VL 模型 |

推理参数: 解析 VLM `temperature=0.1, max_completion_tokens=16384`(env 可覆盖); embedding 限流 `MULTIMODAL_EMBED_RPM=120` + 429 指数退避(5 次, base 2.0s, 复刻参考)。

## 3. 架构总览

```
入库(scripts/ingest_multimodal_pdf.py, 离线 CLI):
  PDF → fitz dpi=200 页图 ─┬─ DOTS·OCR(vLLM) → layout JSON → md → 分块 → [text块|image块]
                           └─ (服务不可达 & 降级开) fitz 纯文本 + 整页图, 仅 text 块
  image块 → VLM ≤300字描述 → multimodal embedding({image, text})
  text块  → multimodal embedding({text})
  → Milvus rag_multimodal (PK=块UUID) + rag_documents 登记 + 页图/插图落盘

查询(/agent/api/library, 新 router, 不动 ask_api):
  collection=multimodal: 问题 → 多模态 embedding(纯文本) → rag_multimodal dense top-k
                        → generate_answer ? DEFAULT_MODEL(上下文=文字) : 直接返回
  collection=text:      现有文本向量 → rag_nodes dense+BM25 全库 → 应用层 RRF → (可选 rerank)
```

图片三阶段角色(v1 §4.2.1 不变): 入库=图+描述联合向量化; 检索=与文本块同空间平等候选; 生成=LLM 只看 VLM 文字描述; 前端=原图缩略图。LLM 直接看图生成 = §7 P2 非目标。

## 4. 密集向量后端切换设计 (本文核心增量)

### 4.1 新后端

新文件 `tools/retrieval_backends/dense_milvus_multimodal.py`:

```python
class MilvusMultimodalDenseBackend:  # implements DenseBackend 协议 (types.py, 不改协议)
    # upsert_document_nodes(document_id, nodes)  → rag_multimodal
    #   text 块: 多模态 embedding({text}); image 块: ({image_url/base64, text=描述})
    #   注意: 与 dense_milvus 不同 — 不依赖 PG rag_nodes 行, 向量在写入时生成(离线管线)
    # search(query_vector, top_k, document_ids=None) → dense top-k, 返回 kind/page_no/image_ref 等字段
    # replace_document_nodes(document_id, []) → 按 document_id 表达式删点(幂等重跑/删除工具)
    # collection 自动创建: schema 见 §6, dim= MULTIMODAL_EMBEDDING_DIM(0=首向量探测后建)
```

向量生成职责在 backend 内(新 `tools/multimodal_vectorizer.py`, DashScope 客户端+限流+退避), **不复用/不修改 `tools/vectorizer.py`** — 两者模型、输入结构、限流参数完全不同, 强行抽象反而耦合。

### 4.2 factory 分支 (唯一改动点, 纯增量)

`retrieval_backends/factory.py` 新增分支(**不删不改现有 milvus 分支**):

```python
if backend == "milvus_multimodal":
    return MilvusMultimodalDenseBackend()
```

校验矩阵(fail-fast, 沿用现有报错风格):

| DENSE_BACKEND | SPARSE_BACKEND | 结果 |
|---|---|---|
| `milvus`(默认) | `milvus`\|`postgres` | 现状, 零改动 |
| `milvus_multimodal` | `none`(新合法值) | ✅ 多模态模式 |
| `milvus_multimodal` | `milvus`\|`postgres` | ValueError: 多模态 dense 无共享 sparse 行, 需 `SPARSE_BACKEND=none` |

`SPARSE_BACKEND=none` = 新合法值: `get_sparse_backend()` 返回 `NoneSparseBackend`(空实现, search 返回 [])。现有 milvus/postgres 分支不动。

### 4.3 作用域守卫

`milvus_multimodal` 模式下 `/agent/api/ask` 管线语义不成立(依赖 rag_nodes 的 PG 行/section tree/sibling)。守卫方式 = `rag_service.answer_question()` 入口 3 行前置检查:

```python
if config.dense_backend == "milvus_multimodal":
    raise ValueError("DENSE_BACKEND=milvus_multimodal 仅服务 /agent/api/library; /ask 请用 DENSE_BACKEND=milvus")
```

这是新增防御分支, 不触碰任何密集向量逻辑; 启动期即可在 server 日志 warn 一次。`/ask` 契约(含 document_ids 必填)零变化 — 默认模式下该分支永不触发。

### 4.4 library 路的实例化

`tools/library/library_service.py` **显式实例化** `MilvusMultimodalDenseBackend()`(collection=multimodal 时), 不经全局 `get_dense_backend()` — 全局 env 切换是"实验/演示整站多模态"用; `/library` 自身按请求参数选库, 与全局 DENSE_BACKEND 解耦。两条使用路径互不干扰。

## 5. 配置清单 (env 全表, 增量)

```bash
# --- 解析 VLM (dots.ocr via vLLM) ---
DOT_OCR_BASE_URL=            # 空=直接 fitz 降级; 如 http://127.0.0.1:6006/v1
DOT_OCR_MODEL=rednote-hilab/dots.mocr
DOT_OCR_API_KEY=0            # vLLM 服务无鉴权时占位
DOT_OCR_DPI=200
DOT_OCR_MAX_THREADS=16
DOT_OCR_TEMPERATURE=0.1
DOT_OCR_MAX_COMPLETION_TOKENS=16384
DOT_OCR_FALLBACK_FITZ=true

# --- 多模态 embedding ---
MULTIMODAL_EMBEDDING_MODEL=multimodal-embedding-one-peace-v1
MULTIMODAL_EMBEDDING_DIM=0   # 0=首响应自动探测并建 collection; 显式值=强校验
MULTIMODAL_EMBED_RPM=120
MULTIMODAL_EMBED_MAX_RETRIES=5
MULTIMODAL_EMBED_BACKOFF_BASE=2.0
DASHSCOPE_API_KEY=           # 密钥仅经 env, 严禁硬编码(参考仓库的反面教材)

# --- 图片描述 VLM ---
MULTIMODAL_VLM_MODEL=qwen-vl-plus

# --- 存储 (图片资产两期策略见 §6) ---
MULTIMODAL_ASSET_STORE=local             # local(一期默认) | minio(二期)
MULTIMODAL_COLLECTION=rag_multimodal
MULTIMODAL_PAGES_DIR=tools/data/multimodal_pages   # local 模式根目录; minio 模式忽略
LIBRARY_ASK_DEFAULT_TOP_K=8

# --- 切换 (§4) ---
DENSE_BACKEND=milvus                  # milvus(默认,零改动) | milvus_multimodal(新)
SPARSE_BACKEND=milvus                 # milvus | postgres | none(新, 仅配套 multimodal dense)
```

新增依赖: `dashscope`(仅多模态 embedding SDK)。**不新增**: `dots_ocr`、`torch`、`qwen_vl_utils`、LangChain 系。

## 6. 数据模型

### Milvus `rag_multimodal` schema (混合, 参考建法复刻, PK 非 auto_id)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(64) PK | 块 UUID |
| `document_id` | INT64 | 对应 rag_documents |
| `filename` | VARCHAR | 源文件名 |
| `title` | VARCHAR | 标题层级补全串 (`H1 --> H2 --> H3`) |
| `kind` | VARCHAR | `text` \| `image` |
| `page_no` | INT64 | 页码(0-based) |
| `text` | VARCHAR | BM25 输入(正文/image 描述); jieba analyzer |
| `category` | VARCHAR | layout category(降级模式为 `fitz`) |
| `image_ref` | VARCHAR | **asset key**(`{document_id}/{name}.jpg`), 与存储后端无关, 仅 image 块 |
| `sparse` | SPARSE_FLOAT_VECTOR | Milvus BM25 Function 自动生成, **v1 查询不用**(预留) |
| `dense` | FLOAT_VECTOR(dim=探测) | 多模态向量, COSINE + AUTOINDEX |

v1 检索仅 dense; sparse 字段为 §7 P2 预留(与 rag_nodes 的 text 加权策略不同: image 块的 text=VLM 描述, 权重设计延后)。

### PG `rag_documents` (复用, 不改表)

新行 `metadata`: `{source: "multimodal_pdf", pages: N, chunks: {text: X, image: Y}, dpi: 200, embedding_model: ...}`。catalog/删除工具复用。

### 图片资产存储 (两期策略, 接口一期定型)

**切分图片(插图+整页图)的存取统一走 asset store 抽象**, 新 `tools/multimodal_asset_store.py`:

```python
class MultimodalAssetStore(Protocol):
    def save(self, document_id: int, name: str, data: bytes) -> str: ...      # 返回 asset key
    def open(self, document_id: int, name: str) -> bytes: ...                  # KeyError=缺失
    def delete_document(self, document_id: int) -> None: ...                   # 删除工具复用
```

- **一期 local**(默认): `LocalFileAssetStore` → `MULTIMODAL_PAGES_DIR/{document_id}/{name}.jpg`; 路径穿越校验内聚于此。
- **二期 minio**: `MinioAssetStore` → bucket `rag-multimodal`, object key 与一期完全同构(`{document_id}/{name}.jpg`)。

**关键不变量**: Milvus `image_ref` 与 API 契约中的 `image_url` 参数只出现 **asset key**, 永不出现绝对路径/presigned URL。因此 一期→二期 迁移 = 换 env + 数据搬迁脚本, **Milvus 点数据零迁移、API 契约零改动、前端零改动**。

二期(`/library/page-image` 行为不变, 后端从 MinIO 流式代理; presigned URL 直连为后续优化):

```
MULTIMODAL_ASSET_STORE=minio
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY= / MINIO_SECRET_KEY=
MINIO_BUCKET=rag-multimodal
MINIO_SECURE=false
```

二期新增依赖 `minio` SDK + 独立 compose service `rag-minio`(不复用 langfuse 的 minio 或 milvus 内嵌 minio — 生命周期不同); 一期不引入。迁移工具: `scripts/multimodal_assets_migrate.py`(遍历本地目录上传 + 抽样校验字节数)。

一期落盘布局(local 模式):

```
MULTIMODAL_PAGES_DIR/
  {document_id}/
    page_{n}.jpg        # 整页图(检索证据缩略图 + fitz 降级唯一产物)
    image_{md5}.jpg     # 插图(base64 解码, md5 命名)
```

## 7. 入库管线 (`scripts/ingest_multimodal_pdf.py`, 六步)

`--data-dir --glob "*.pdf" --document-id-start N`:

1. **解析**(`dots_ocr_client.py`): 探测 `DOT_OCR_BASE_URL` 可达 → 逐页 layout JSON + md(线程池, 页序排序, 失败页记清单继续); 不可达且降级开 → fitz 文本(`## 第 N 页` 前缀)+整页 jpg, 无插图。
2. **分块**(自实现, 不引 LangChain, v1 §4.2 已定): 每页 md → H1-H3 标题边界分块(~60 行) → 正则抽插图占位单独成 image 块、正文去图 → >1000 字语义分块(切句→相邻余弦→percentile 断点, 复用现有**文本** embedding 仅作分块判据, 不入库)→ 标题层级补全拼 title(~40 行)。
3. **图片描述**: image 块带前后文调 `MULTIMODAL_VLM_MODEL` 生成 ≤300 字(复刻参考 prompt: 结合前文/后文)。
4. **向量化**: `multimodal_vectorizer.py` — text 块 `{text: title+"："+正文}`; image 块 `{image: base64, text: 描述}`; 固定窗口限流+429 退避。
5. **存储**: `MilvusMultimodalDenseBackend.upsert_document_nodes()`(含 collection 首建/维度校验) + `rag_documents` 登记 + 落盘。幂等: 先 `document_id==X` 删点再全量写。
6. **可观测**: `log_rag` 各 stage; 失败页/失败块清单 + 截断标记(`text_truncated: true`)。

## 8. 查询 API (`/agent/api/library`)

v1 §4.3 契约全部沿用(ask/collections/page-image 三端点、请求响应模型、错误契约 422/404/502、路径穿越防护), 唯一修正:

- **multimodal 检索改走 `MilvusMultimodalDenseBackend.search()`**(§4.4 显式实例化), 不再是"library_service 内联调用 embedding+Milvus" — 后端化统一了删除工具/factory/维度校验的复用面。
- 其余(text 路 RRF、rerank 复用、CONTEXT_CHAR_BUDGET、generate_answer 语义、证据 `[文件名:p页]`)不变。

## 9. 非功能需求 (v1 §4.4 全部沿用)

维度 fail-fast / 密钥仅 env / 长文本截断标记 / Windows 路径 / dpi=200 磁盘占用注明。补充:
- **零改动回归门禁**: 部署后默认 env 下跑现有 70 单测 + `/ask` 冒烟一例, 证明文本链路无感。
- **vLLM 服务不可达**: 入库自动降级 + `/library` multimodal 检索不受影响(检索不依赖解析服务)。

## 10. 验收标准

1. **零改动**: `git diff` 中 `vectorizer.py`/`dense_milvus.py`/`ask_api.py`/`rag_nodes` 相关 = 0 行改动; 70 存量单测全绿。
2. **env 切换**: `DENSE_BACKEND=milvus` 一切如旧; `=milvus_multimodal` 时 factory 返回新后端, `/ask` 得到清晰 ValueError, `/library` 正常。
3. **混配 fail-fast**: `milvus_multimodal` + `sparse=milvus` 启动即报错(新单测覆盖)。
4. **入库**: 课件 PDF → `rag_documents` +1, `rag_multimodal` 点数=文本块+图片块; 重跑幂等; 停 DOTS·OCR 降级入库成功。
5. **查询**: `POST /agent/api/library/ask {collection:"multimodal", question:"有界流和无界流的定义"}` → 200, answer 非空, evidence 相关页 `image_url` 可 GET 到字节。
6. **单测**(新增): 维度校验失败路径 / 429 退避 / 路径穿越拒绝 / factory 混配矩阵 / layout JSON→md 转换 / fitz 降级分支。

## 11. 里程碑

| 阶段 | 内容 | 交付物 |
|---|---|---|
| **MM-1 解析+入库** | `dots_ocr_client.py` + 分块 + `multimodal_vectorizer.py` + `dense_milvus_multimodal.py` + `multimodal_asset_store.py`(抽象+local 实现) + CLI | 真 PDF 入库, 验收 1/3/4 |
| **MM-2 检索 API** | `library_api.py` + `library_service.py` + `SPARSE_BACKEND=none` + ask 守卫 | 验收 2/5/6 |
| **MM-3 前端** | 第三部分 `/library` 页(独立设计, 见迁移文档 §5) | 验收 5 的 UI 面 |
| **二期 MinIO 资产迁移** | `MinioAssetStore` + `scripts/multimodal_assets_migrate.py` + compose `rag-minio`; env 切 `MULTIMODAL_ASSET_STORE=minio` | 图片字节抽样校验全等, API/前端零改动 |

MM-1/MM-2 后端可独立验收; 前端不阻塞。二期 MinIO 在一期验收后独立排期(asset store 接口已定型, 迁移成本=实现类+搬迁脚本)。

## 12. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| dots.mocr vLLM 显存(3.4B 级, ~8-10GB)与本地 4B reranker 争卡 | 解析是离线批任务, 与在线服务错峰; `DOT_OCR_BASE_URL` 指向远端亦可 |
| 多模态 embedding 维度与现有 1536 不同 | 独立 collection 隔离; `MULTIMODAL_EMBEDDING_DIM` 探测+强校验 |
| layout JSON 偶发解析失败 | 官方 filtered 降级路径照抄(存原始响应入失败清单, 不中断整批) |
| DashScope 限流 | 固定窗口+退避复刻(参考已验证 120RPM 稳定) |
| 中文 PDF 的 BM25 jieba 分词质量 | v1 检索不走 sparse, 无影响; P2 启用 sparse 时再评 |
| **开放**: vLLM 服务部署位置(本机 4090/远端) | 不阻塞设计; MM-1 前用户确认, 文档已给 docker 启动命令 |

## 13. 对 v1 §4 的覆盖汇总

| v1 条目 | v2 处置 |
|---|---|
| §4.1 env 表 | 扩充(§5), `SPARSE_BACKEND=none` 新增 |
| §4.2 六步管线 | 沿用, 解析器明确为自实现 `dots_ocr_client.py`(不引 dots_ocr 包) |
| §4.3 library API | 契约沿用, multimodal 检索改经新 DenseBackend |
| §4.5 验收 | 扩为 §10(增加零改动回归门禁与混配矩阵) |
| (新增) | 零改动边界 / factory 切换与校验矩阵 / ask 守卫 / 模型支持矩阵 |
