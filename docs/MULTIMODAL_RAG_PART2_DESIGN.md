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
| **多模态 embedding** | 火山方舟 `doubao-embedding-vision`(订阅 plan 端点, 实测解析为 251215 版, dim=2048) | `MULTIMODAL_EMBEDDING_PROVIDER=ark`(默认)\|`dashscope` + `MULTIMODAL_EMBEDDING_MODEL` / `_DIM`(0=自动探测) | ark: httpx 直调 `{base}/embeddings/multimodal`, input=content-block 数组 — text 块 `[{type:text}]`, image 块 `[{type:image_url, url:"data:image/jpeg;base64,..."}, {type:text}]`; **响应 `data` 为单对象**(非 OpenAI 数组, 实现须兼容) | dashscope 备选: `multimodal-embedding-one-peace-v1` / `tongyi-embedding-vision-flash-*`(SDK `{image, text}`); 换模型必须核维度 |
| **图片描述 VLM** | **`doubao-seed-2-0-lite-260428`(plan 端点实测通过, 零新增配置)** — 高并发批量(RPM 30000)匹配离线入库场景 | `MULTIMODAL_VLM_MODEL` + `MULTIMODAL_VLM_BASE_URL` / `_API_KEY`(可选, 默认复用 OPENAI_* 即 plan 端点) | OpenAI 兼容 chat + image_url base64 block; **须 `extra_body={thinking:{type:disabled}}`**(该系模型 thinking 默认开, 实测带 reasoning_content, 批量描述禁用省时省 token — 复用 RAGAS judge 修复经验) | 备选 `glm-5.3-flash`(实测通过, 原生多模态, thinking 不可禁每图多 ~580 tokens; **双订阅通道**: ark plan 或 zhipu coding `ZHIPU_BASE_URL`, 实测⑫)、`doubao-seed-evolving`(实测通过); GLM `glm-4v-flash`(免费)/`glm-4v-plus`(`ZHIPU_API_KEY` 现成, bigmodel v4 端点); `qwen-vl-plus` 末选(`QWEN_API_KEY` 实际为空) |

**实测记录(2026-09-15, 订阅 plan 端点)**: ①`doubao-embedding-vision` 文本/图片-base64/图文联合三种输入均 200, dim=2048; ②base64 data URI 被服务端解码(1×1 图报"最小 14px", 换 320×240 通过 — 一期本地图片无需公网 URL, 二期 MinIO 亦不必开公网); ③响应 `data` 为单对象含 `embedding`, 与 OpenAI 数组结构不同; ④`doubao-seed-1-6-vision-250815` 与 `glm-4.5v` 在 plan 端点 404 UnsupportedModel — VL 描述模型若走方舟需正式按量端点+独立 key, 或走 GLM bigmodel。⑤**`doubao-seedream-5.0-pro` 不适用图片描述** — 属图像生成模型(文生图/图生图), 与视觉理解(图→文)是方舟两条独立产品线; 评估排除(2026-09-15)。⑥`doubao-seed-2-0-lite-260428` 与 `doubao-seed-evolving` 在 plan 端点 chat+image_url 直读成功(描述准确, finish=stop), 但 thinking 默认开(reasoning_content 非空) — 实现须禁用; `doubao-seed-2-1-pro-260628` 404。⑦key 盘点: `QWEN_API_KEY` 为空(原默认 qwen-vl-plus 的"key 现成"假设不成立), `ZHIPU_API_KEY` 现成(GLM 备选可用)。⑨GLM 视觉系 API 实测(bigmodel v4, ZHIPU key): `glm-4v-flash` **免费可用且读图内文字准确**(实测完整读出测试图英文标注); `glm-4.5v`/`glm-4v-plus` 429 余额不足(**需充值**才可用); 用户所列 GLM 私有实例价格表(100-200 元/算力单元/天)为专属部署方案, 与 API 按量调用的场景不同 — 本项目离线批任务不值得私有实例, GLM 备选定格 `glm-4v-flash`(免费), 质量升级路径=充值 4.5v 或 plan 内 `doubao-seed-evolving`。⑩GLM-5.3 系视觉探测(plan 端点): `glm-5.3`(当前 DEFAULT_MODEL) **仅文本**(400 Model only support text input — 生成模型看不了图, 印证 VLM 描述层必要); `glm-5.3-flash` **原生多模态**(image_url 通过, 描述准确), 但 **thinking 不可禁**(400 not supported, 只能带跑, 实测每图 ~580 reasoning tokens) — 列为质量优先备选, 与 DEFAULT_MODEL 同家族; 其原生多模态+高智能亦是 §7 P2"生成阶段 LLM 直接看图"的天然候选。⑪`glm-4.6v-flash`(bigmodel 免费档, 128K 上下文): 实测**可用**(描述准确含图内数据), 但有**间歇性 429"访问量过大"**(免费档拥挤, 重试即过) — 入库批任务须带重试; `glm-4v-flash` 输出上限 1024 tokens 对 ≤300 字描述契约定量够用(≈450 token), 天花板场景(千 token 结构化图表解析)切 doubao-lite。免费=智谱拉新策略(输入/输出/缓存全免, 限并发 10, 无每日额度)。⑫**ZHIPU_API_KEY 判定为 Coding Plan 订阅 key**(2026-09-15 双端点对照实测): `glm-5.3-flash` 图片理解在 **coding 端点 `/api/coding/paas/v4` 订阅内可用**(content 准确, reasoning=323), 而标准端点 `/api/paas/v4` 按量 429 余额不足; `glm-4v-flash` 两端点均通(免费档)。⇒ glm-5.3-flash 具**双订阅通道**(ark plan + zhipu coding), 质量档备选更稳; `.env` 已显式配置 `ZHIPU_BASE_URL=coding 端点`。⑧GLM `embedding-3`(2048 维)为**纯文本**嵌入(input 仅 string, 无 image) — 不能用于 multimodal collection(图文须同空间); 且与 doubao-embedding-vision 维度巧合相同但**维度相同≠同空间**, 禁止混用。**文本 embedding 备选记录(2026-09-15)**: 现有文本链路主力 = OpenRouter `nvidia/nemotron-3-embed-1b:free`(2048 维, 免费); GLM `embedding-3` 为备选 — 代码已完整支持(`vectorizer.py` zhipu 分支, `EMBEDDING_PROVIDER=zhipu` + `ZHIPU_EMBEDDING_MODEL=embedding-3`, `ZHIPU_API_KEY` 现成, 批量上限 64/单条 8000 字符); **切换=换向量空间**, 须 `EMBEDDING_DIMENSION` 对齐(2048)后 `reindex-vectors` 全量重嵌, 本期不执行(零改动约束)。**架构注**: 多模态 embedding 管"找得到"(图→向量, 检索用), VLM 描述管"讲得出"(图→文, 生成 LLM 是纯文本模型看不了图 + BM25 text 字段 + 证据卡 preview) — 二者不可互替。

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

### 4.5 GLM 接入端点总表 (2026-09-15 记录, 用户提供的 Coding Plan 官方端点 + 实测)

| 协议 | Base URL | 说明 |
|---|---|---|
| Anthropic Message | `https://open.bigmodel.cn/api/anthropic` | Claude Code 类客户端 |
| OpenAI Chat Completion | `https://open.bigmodel.cn/api/coding/paas/v4` | **Coding Plan 订阅通道**(我们的 ZHIPU_API_KEY 属此类, 实测 glm-5.3-flash 视觉可用) |
| OpenAI Response | `https://open.bigmodel.cn/api/v1` | Responses API 形态 |
| 标准按量 API | `https://open.bigmodel.cn/api/paas/v4` | embeddings 走此端点(不在 Coding Plan 订阅内); glm-4v-flash 免费档亦通 |

项目内 GLM 消费路径: 生成(DEFAULT_MODEL)走 ark plan; 多模态 VLM 备选走 `ZHIPU_BASE_URL`(.env 已配 coding 端点); zhipu embedding 备选走标准端点(vectorizer 硬编码, 语义正确不动)。

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

# --- 多模态 embedding (provider 可切, 默认方舟) ---
MULTIMODAL_EMBEDDING_PROVIDER=ark       # ark(默认, doubao-embedding-vision) | dashscope(备选)
MULTIMODAL_EMBEDDING_MODEL=doubao-embedding-vision
MULTIMODAL_EMBEDDING_BASE_URL=          # 空=复用 OPENAI_BASE_URL(订阅 plan 端点); 可指正式 v3 端点
MULTIMODAL_EMBEDDING_API_KEY=           # 空=复用 OPENAI_API_KEY
MULTIMODAL_EMBEDDING_DIM=0              # 0=首响应自动探测并建 collection; 显式值=强校验(ark 默认 2048)
MULTIMODAL_EMBED_RPM=120
MULTIMODAL_EMBED_MAX_RETRIES=5
MULTIMODAL_EMBED_BACKOFF_BASE=2.0
DASHSCOPE_API_KEY=                      # 仅 provider=dashscope 时需要; 密钥仅经 env, 严禁硬编码

# --- 图片描述 VLM (默认方舟 seed-2.0-lite, plan 端点实测; 可切 GLM/qwen) ---
MULTIMODAL_VLM_MODEL=doubao-seed-2-0-lite-260428   # 备选: glm-5.3-flash(原生多模态,thinking不可禁) / doubao-seed-evolving / glm-4v-flash(免费,ZHIPU key)
MULTIMODAL_VLM_BASE_URL=                # 空=复用 OPENAI_BASE_URL(plan 端点); GLM=ZHIPU_BASE_URL(标准 /api/paas/v4 或 Coding Plan /api/coding/paas/v4); qwen=https://dashscope.aliyuncs.com/compatible-mode/v1
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 标准 API; Coding Plan 订阅改 /api/coding/paas/v4 (chat 订阅内, embeddings 不在内)
MULTIMODAL_VLM_API_KEY=                 # 空=复用 OPENAI_API_KEY; GLM=ZHIPU_API_KEY; qwen=QWEN_API_KEY
MULTIMODAL_DESCRIBE_MODE=vlm            # vlm(默认, 基础层+VLM 增强) | context(纯上下文零 API 成本)

# --- 存储 (图片资产两期策略见 §6) ---
MULTIMODAL_ASSET_STORE=local             # local(一期默认) | minio(二期)
MULTIMODAL_COLLECTION=rag_multimodal
MULTIMODAL_PAGES_DIR=tools/data/multimodal_pages   # local 模式根目录; minio 模式忽略
LIBRARY_ASK_DEFAULT_TOP_K=8

# --- 切换 (§4) ---
DENSE_BACKEND=milvus                  # milvus(默认,零改动) | milvus_multimodal(新)
SPARSE_BACKEND=milvus                 # milvus | postgres | none(新, 仅配套 multimodal dense)
```

新增依赖: **零**(默认 ark provider 用 httpx 直调 — openai 依赖自带 httpx; 响应 data 单对象兼容在 vectorizer 内处理)。`dashscope` 仅 `MULTIMODAL_EMBEDDING_PROVIDER=dashscope` 时安装。**不新增**: `dots_ocr`、`torch`、`qwen_vl_utils`、LangChain 系。

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
| `book_id` | VARCHAR(索引) | **书籍层级**(§6.1): 所属书名规范化串; 独立文档=其自身书名; filter 下推用 |
| `chapter_label` | VARCHAR | 章节标签(`{book_id} · 第{N}章 · {章节名/文件名}`); 章=一个 PDF 文档 |
| `sparse` | SPARSE_FLOAT_VECTOR | Milvus BM25 Function 自动生成, **v1 查询不用**(预留) |
| `dense` | FLOAT_VECTOR(dim=探测) | 多模态向量, COSINE + AUTOINDEX |

v1 检索仅 dense; sparse 字段为 §7 P2 预留(与 rag_nodes 的 text 加权策略不同: image 块的 text=VLM 描述, 权重设计延后)。

### 6.1 书籍层级模型与 metadata filter (2026-09-15 需求扩展)

```
Library(全库) ─ Book(逻辑书, book_id) ─ Chapter(章 = 一个 PDF = document_id) ─ Chunk(text|image)
```

**自动合并语义**: 同一次 CLI 调用 + 同 `--book` 值 = 一本书; 章节 = 匹配文件按**文件名自然排序**编号 1..N; 不做文件名启发式猜书(脆、易错聚合)。未指定 `--book` 的 PDF = 独立文档(自成一书, book_id=文件名去扩展名)。

| 存储 | 承载 |
|---|---|
| Milvus `book_id`/`chapter_label` 标量字段(索引) | filter 下推: `book_id in [...]`、`book_id in [...] and chapter_label in [...]`、`kind in [...]` |
| PG `rag_documents.metadata` 追加 `{book_id, chapter_index, chapter_label}` | 事实源; `/library/filters` 聚合数据源。**层级关系用 metadata 不建表的理由**(2026-09-15 决策): 书当前无独立属性(仅分组标签, 全部需求在读路径)、零 schema 变更、删除无孤儿行; 引用完整性由 CLI 写入时 trim 规范化 book_id 保证。**升级 `library_books` 表的触发条件**(任一出现即建表, metadata 保留冗余做 Milvus 下推): ①书需独立属性(作者/封面/权限) ②重命名成高频操作 ③书过百本且 /filters 聚合变慢 ④第五部分动态 collection 落地。对照: 用户集合(§6.2)跨文档+chunk 级+CRUD 生命周期+唯一名约束, 故必须建表 — 两者的分界 = 分组标签 vs 一等实体 |
| 检索执行路径 | **filter 下推 Milvus 标量字段**(书/章/kind/筛选集合 → `expr` 与向量搜索一次完成, 归属关系写入时已物化到每个 chunk 点上, 检索零 join); 枚举集合 = PK 列表下推(`id in [...]`, ≤500); **否决"PG 倒查 chunk id 集合再查"** — 表达式爆炸(一书几千 id)+热路径多一跳 PG+把引擎原生标量过滤搬到应用层 |

### 6.2 用户自定义集合 (PG 新表 `library_sets`, 多模态域专用, 不碰现有表)

前端动态 filter 圈选或搜索结果手动勾选 chunk → 保存为命名集合 → 集合成为可检索单位("选集/书架"模式):

```sql
CREATE TABLE library_sets (
  set_id      TEXT PRIMARY KEY,            -- uuid
  name        TEXT NOT NULL UNIQUE,
  kind        TEXT NOT NULL,               -- 'filter'(动态) | 'enumerated'(静态勾选)
  filter_json JSONB,                       -- kind=filter: {books?, chapters?, kinds?} → 查询时展开为 Milvus 表达式; 库变化集合自动跟随
  chunk_ids   JSONB,                       -- kind=enumerated: 显式 chunk PK 列表(上限 500, 超限提示改用 filter 型)
  created_at  TIMESTAMPTZ DEFAULT now()
);
```

枚举型检索 = Milvus `id in [chunk_ids]`(PK 过滤); 删除文档时其 chunk 留在 chunk_ids 中但检索自然落空(惰性失效, 不做级联清理 — 集合详情页显示失效计数提示)。

### PG `rag_documents` (复用, 不改表)

新行 `metadata`: `{source: "multimodal_pdf", pages: N, chunks: {text: X, image: Y}, dpi: 200, embedding_model: ..., book_id, chapter_index, chapter_label}`。catalog/删除工具复用。

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

`--data-dir --glob "*.pdf" --document-id-start N [--book "书名"] [--chapter-start N]`(§6.1: 同 `--book` 多 PDF 自动聚合为一本书, 章节按文件名自然排序编号; `--chapter-start` 设起始章号默认 1):

1. **解析**(`dots_ocr_client.py`): 探测 `DOT_OCR_BASE_URL` 可达 → 逐页 layout JSON + md(线程池, 页序排序, 失败页记清单继续); 不可达且降级开 → fitz 文本(`## 第 N 页` 前缀)+整页 jpg, 无插图。
2. **分块**(自实现, 不引 LangChain, v1 §4.2 已定): 每页 md → H1-H3 标题边界分块(~60 行) → 正则抽插图占位单独成 image 块、正文去图 → >1000 字语义分块(切句→相邻余弦→percentile 断点, 复用现有**文本** embedding 仅作分块判据, 不入库)→ 标题层级补全拼 title(~40 行)。
3. **图片描述(分层策略, 2026-09-15 修订)**: 基础层 = caption+紧邻上下文文字从 layout 直接拼接(零成本, 永远有); 增强层 = `MULTIMODAL_DESCRIBE_MODE=vlm`(默认)时带上下文调 `MULTIMODAL_VLM_MODEL` 生成 ≤300 字读出**图内内容**(数字/趋势/极值 — Picture 元素的 text 字段被 dots.ocr prompt 省略, 图内信息只在像素里, 纯上下文拼不出); `=context` 时跳过 VLM 零 API 成本(接受图内信息缺失)。VLM 失败 → 降级为基础层描述(非占位符)。
4. **向量化**: `multimodal_vectorizer.py` — text 块 `{text: title+"："+正文}`; image 块 `{image: base64, text: 描述}`; 固定窗口限流+429 退避。
5. **存储**: `MilvusMultimodalDenseBackend.upsert_document_nodes()`(含 collection 首建/维度校验) + `rag_documents` 登记 + 落盘。幂等: 先 `document_id==X` 删点再全量写。
6. **可观测**: `log_rag` 各 stage; 失败页/失败块清单 + 截断标记(`text_truncated: true`)。

## 8. 查询 API (`/agent/api/library`)

v1 §4.3 契约沿用(ask/collections/page-image 三端点、请求响应模型、错误契约 422/404/502、路径穿越防护), 修正与扩展:

- **multimodal 检索改走 `MilvusMultimodalDenseBackend.search()`**(§4.4 显式实例化), 不再是"library_service 内联调用 embedding+Milvus" — 后端化统一了删除工具/factory/维度校验的复用面。
- **ask 请求扩展**(2026-09-15 需求, 仅 multimodal 路; text 路忽略并告警): `filters: {books?: string[], chapters?: string[](=document_id), kinds?: ("text"|"image")[]}` 与 `set_id?: string` **二选一**(同传 422); 均省略 = 全库(现状)。filter → Milvus 标量表达式下推(`book_id in [...] and chapter_label in [...] and kind in [...]`); set_id → §6.2 展开。evidence 不变。
- **新端点 `GET /agent/api/library/filters`**: 页面初始化拉取可用 filter 标签面 — 从 `rag_documents.metadata` 聚合: `{books: [{book_id, title, chapter_count, chunk_count?, chapters: [{document_id, chapter_index, chapter_label, filename, pages, chunks}]}], kinds: ["text","image"]}`(chunk_count 惰性: 书量大时按需)。前端据此渲染级联筛选器。
- **新端点 集合 CRUD**(§6.2): `POST /library/sets {name, filter?|chunk_ids?}`(二选一, chunk_ids ≤500) / `GET /library/sets`(含 chunk 失效计数) / `DELETE /library/sets/{set_id}`。重名 422; 集合检索直接走 ask 的 set_id。
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
6. **单测**(新增): 维度校验失败路径 / 429 退避 / 路径穿越拒绝 / factory 混配矩阵 / layout JSON→md 转换 / fitz 降级分支 / filter 表达式构造 / 集合展开。
7. **书籍聚合**(§6.1): 同 `--book` 入库 3 个章节 PDF → `/library/filters` 返回 1 本书 3 章; `filters:{books:[...]}` 检索命中仅该书; `filters:{chapters:[doc_id]}` 命中仅该章。
8. **自定义集合**(§6.2): 前端圈选 filter 保存集合 → `set_id` 检索结果 ⊆ 集合范围; 勾选 chunk 保存枚举集合 → 检索恰含所选; 删除集合后 set_id 检索 404; chunk_ids>500 → 422。
9. **零改动不变**: filter/集合全部能力不影响 `/ask` 与 text 路(text 路传 filters → 忽略+告警日志, 契约不报错)。

## 11. 里程碑

| 阶段 | 内容 | 交付物 |
|---|---|---|
| **MM-1 解析+入库** | `dots_ocr_client.py` + 分块 + `multimodal_vectorizer.py` + `dense_milvus_multimodal.py`(schema 含 book_id/chapter_label) + `multimodal_asset_store.py`(抽象+local 实现) + CLI(`--book`/`--chapter-start` 聚合) | 真 PDF 入库(含多章节书), 验收 1/3/4/7 入库面 |
| **MM-2 检索 API** | `library_api.py` + `library_service.py`(filters/set_id 下推) + `GET /library/filters` 聚合 + PG `library_sets` 表 + 集合 CRUD + `SPARSE_BACKEND=none` + ask 守卫 | 验收 2/5/6/7 检索面/8 |
| **MM-3 前端** | 第三部分 `/library` 页 + 书籍/章节级联筛选器(启动拉 /filters) + 证据卡"加入集合"勾选 + 集合管理下拉(见迁移文档 §5) | 验收 5/7/8 的 UI 面 |
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
| bbox 坐标系(smart_resize vs 页图像素)映射错位 | §14.1-1: MM-1 首跑真 PDF 打印 bbox 与页图尺寸比对验证; 错位则加比例修正 |
| **开放**: vLLM 服务部署位置(本机 4090/远端) | 不阻塞设计; MM-1 前用户确认, 文档已给 docker 启动命令 |

## 13. 对 v1 §4 的覆盖汇总

| v1 条目 | v2 处置 |
|---|---|
| §4.1 env 表 | 扩充(§5), `SPARSE_BACKEND=none` 新增 |
| §4.2 六步管线 | 沿用, 解析器明确为自实现 `dots_ocr_client.py`(不引 dots_ocr 包) |
| §4.3 library API | 契约沿用, multimodal 检索改经新 DenseBackend |
| §4.5 验收 | 扩为 §10(增加零改动回归门禁与混配矩阵) |
| (新增) | 零改动边界 / factory 切换与校验矩阵 / ask 守卫 / 模型支持矩阵 |

## 14. 实现蓝图 (文件级, MM-1 自底向上)

| # | 文件 | 职责与关键签名 | 依赖 |
|---|---|---|---|
| 1 | `tools/multimodal_asset_store.py` | `MultimodalAssetStore` 协议(save/open/delete_document) + `LocalFileAssetStore`(防穿越内聚) + `get_asset_store()` env 分发(lru_cache) | 无 |
| 2 | `tools/dots_ocr_client.py` | `ParsedPage{page_no, layout, md_content, page_image_jpg}`; `DotsOcrClient.healthy()/parse_pdf()/parse_pdf_fitz_fallback()`; `layout_to_md(page)` 自实现 | openai(现有), fitz, Pillow |
| 3 | `tools/multimodal_chunker.py` | `MmChunk{kind, page_no, title, text, image_name, category}`; `chunk_document(pages)`; `_semantic_split()` 复用现有文本 embedding 仅作分块判据 | 2 |
| 4 | `tools/multimodal_vlm.py` | `describe_image(image_b64, prev_text, next_text) -> str`(≤300字) | openai(现有) |
| 5 | `tools/multimodal_vectorizer.py` | provider 抽象(ark 默认: httpx 直调 `/embeddings/multimodal`, data 单对象兼容, 复用 OPENAI_* env 或独立覆盖; dashscope 备选); `detect_dim()/embed_texts()/embed_image_with_text()`; RPM 窗口+429 退避+截断标记 | httpx(自带); dashscope 可选 |
| 6 | `tools/retrieval_backends/dense_milvus_multimodal.py` | `ensure_collection()/upsert_document_nodes()/search()/replace_document_nodes()`; 连接复用 milvus_store 单例模式, collection 操作独立方法 | 5 |
| 7 | `scripts/ingest_multimodal_pdf.py` | 六步编排 CLI(§16) | 1-6 |
| 8 | `tools/library/library_service.py` + `library_api.py` | §8 契约; multimodal 路显式实例化新 backend | 6 |
| 9 | 接缝×3 | factory `milvus_multimodal` 分支 + `NoneSparseBackend` + 校验矩阵; `rag_service.answer_question` 3 行守卫; `delete_ingested_document` 多模态分支(删点+`assets.delete_document`) | 6 |
| 10 | `tools/library/library_repository.py` | `library_sets` 表 CRUD + `/library/filters` 的 rag_documents.metadata 聚合查询 + filter→Milvus 表达式构造(单测重点) | PG |

### 14.1 关键实现决策 (设计期定死, 实现期不再议)

1. **插图来源 = bbox 裁剪**: dots.ocr 的 Picture 元素只有 bbox 无 bytes → PIL 在 dpi=200 整页图上裁剪。**坐标系风险**: 模型输出 bbox 可能基于 `smart_resize` 后分辨率, 裁剪前按 `input_height/input_width ↔ 页图高宽` 比例映射; MM-1 首跑用真 PDF 打印 bbox vs 页图尺寸验证一次。
2. **md 中图片表达**: `layout_to_md` 对 Picture 输出 `![image_{n}](image_{md5}.jpg)` 占位; chunker 按占位正则抽取, 裁剪后的 jpg 经 asset store 落盘, `image_name` 记入 MmChunk。
3. **跨页章节**: v1 不跨页合并 section; 页首无标题时 title 层级继承上一页末层级(课件类 PDF 章节跨页高频, 简化可接受)。
4. **协议兼容**: `MilvusMultimodalDenseBackend` 实现现有 `DenseBackend` 协议(方法签名兼容, 入参为结构化 dataclass); `search` 返回 `MmHit{kind, document_id, filename, title, page_no, score, text_preview, image_ref?}`。
5. **filtered 页处理**: layout JSON 解析失败的页保留原始响应作 md(纯文本降级语义), layout 置空 → 该页无插图块, 不中断整批。

## 15. 错误处理矩阵 (各阶段失败行为)

| 阶段 | 失败 | 行为 | 载体 |
|---|---|---|---|
| 解析 | vLLM 不可达 | `DOT_OCR_FALLBACK_FITZ=true` → fitz 降级整批; false → CLI 退出码 2 | log_rag stage=parse |
| 解析 | 单页 JSON 解析失败 | filtered 降级(§14.1-5), 计入失败页清单 | 失败清单 JSON |
| 解析 | 单页请求超时/异常 | 同 filtered; 重试 1 次 | 同上 |
| 描述 | VLM 单图失败 | 降级为**基础层描述**(caption+上下文拼接, §7-3), 不向量化阻塞; 计入失败块清单 | 失败清单 |
| 向量化 | 429 | 指数退避 5 次(120RPM 窗口内) | log |
| 向量化 | 退避耗尽/其它异常 | 整批 fail-fast(半入库状态由幂等重跑覆盖) | CLI 退出码 3 |
| 存储 | 维度不符 collection | upsert 前 fail-fast, 报"换模型须重建 collection" | ValueError |
| 检索 | embedding 失败/Milvus 故障 | API 502 | /library |
| 检索 | asset 缺失(image_ref 指向不存在) | 证据卡降级为纯文字(text_preview), `image_url` 省略 | /library |
| 检索 | collection 空/未建 | 200 + evidence=[] + answer=null (`available=false` 见 /collections) | /library |

## 16. CLI 契约 (`ingest_multimodal_pdf.py`)

```
参数: --data-dir --glob "*.pdf" --document-id-start N
      [--no-fallback]  # 禁 fitz 降级(默认开)
输出: 每文档一行摘要 {document_id, filename, pages, chunks_text, chunks_image, filtered_pages, failed_images, elapsed_s}
      失败清单: {document_id}/ingest_report.json (页级/块级失败明细, 供重跑定位)
退出码: 0=成功(含 filtered 降级页) | 2=解析不可达且禁降级 | 3=向量化/存储失败 | 4=参数/文件错误
幂等: 同 document_id 重跑 = 先删点再全量写, 点数不变(验收 §10-4)
上限防护: 单文档 >500 页 或 PDF>200MB → 拒绝并提示(防打爆磁盘/批任务时长)
```

## 17. 测试与验证计划

单测(不依赖外部服务, 按依赖顺序):

| 模块 | 用例 |
|---|---|
| `layout_to_md` | 11 类 category 各一行 → md 断言(table→HTML/formula→latex/Picture→占位/header-footer 跳过); 空布局; filtered 页 |
| chunker | 标题边界/层级串; 跨页标题继承; 插图占位抽取+正文去图; >1000 字语义分块(mock embedding); 无标题页兜底单块 |
| asset store | save/open roundtrip; 路径穿越拒绝(`../`/绝对路径/符号链接); delete_document 幂等 |
| vectorizer | 429 退避节奏(mock 时钟); RPM 窗口; dim 探测; 截断标记 |
| backend | FakeClient(仿 test_milvus_store) 幂等删插; 维度校验失败路径; search 字段映射 |
| factory | 混配矩阵: mm+none ✓ / mm+milvus raise / none 单独+text dense 合法性 |
| library_service | text/multimodal 分支(mock); 502/空集合契约; image_ref 缺失降级 |

端到端(需服务): 真 PDF 入库(点数/幂等/停 vLLM 降级); bbox 坐标验证(§14.1-1); curl 验收 §10-5; 70 存量单测回归 + `/ask` 冒烟(零改动证明)。

## 18. 性能预算与可观测性

| 项 | 预算/口径 |
|---|---|
| 解析吞吐 | vLLM 单卡 ~15-30 页/分钟(官方 batch 推理量级); 100 页课件 ≈ 4-7 分钟, 线程池 16 并发请求 |
| embedding 吞吐 | 120RPM 上限 → 100 页(~200 块) ≈ 2 分钟; 入库总时长主要受解析支配 |
| 检索延迟 | multimodal 路 = 1 次 embedding(API ~200ms) + 1 次 Milvus search(<50ms); P95 目标 <1.5s(不含生成) |
| 磁盘 | dpi=200 页图 ~150-300KB/页 + 插图; 100 页课件 ≈ 30-60MB(文档注明, 二期 MinIO 解耦) |
| 可观测 | `log_rag` 新 stage: parse/chunk/describe/embed/upsert 各记 {document_id, counts, elapsed}; /library 响应带 trace_id+latency_ms(沿用现有中间件) |

## 19. 系统架构总览 (跨部分横切, 2026-09-15 补全)

### 19.1 服务拓扑

```
[浏览器 Next.js :3000]
   ├─ /                文档问答(现有, /agent/api/ask 代理)
   └─ /library         全库查询(MM-3, /agent/api/library 代理)
[FastAPI :8000]
   ├─ ask_api(现有, 零改动)          rag_nodes 路(dense=milvus+sparse=milvus)
   ├─ library_api(MM-2 新增)         text 路(rag_nodes 全库 RRF) / multimodal 路(rag_multimodal)
   └─ ingest CLI(离线)               EDGAR 管线(现有) / multimodal_pdf 管线(MM-1)
[基础设施]
   ├─ PostgreSQL(rag_documents/rag_nodes/eval_jobs — 事实源)
   ├─ Milvus(rag_nodes: dense+BM25 7541 点 | rag_multimodal: 多模态, MM-1 建)
   └─ 资产: 一期本地盘 MULTIMODAL_PAGES_DIR → 二期 rag-minio
[模型服务(全部外部)]
   ├─ ark plan 端点: glm-5.3(生成) / doubao-embedding-vision(多模态向量) / doubao-seed-2-0-lite(VLM 描述)
   ├─ zhipu coding 端点(ZHIPU_BASE_URL): glm-5.3-flash 备选通道
   └─ vLLM(自起, 解析 dots.mocr) + OpenRouter(文本 embedding) + 本地 4B reranker(GPU)
```

### 19.2 Collection 全景

| collection | 维度 | 检索方式 | 写入方 | 消费方 |
|---|---|---|---|---|
| `rag_nodes` | 1536(OpenRouter nvidia) | dense+BM25 混合+scope 下推 | EDGAR 管线(现有) | /ask + /library text 路 |
| `rag_multimodal` | 2048(ark doubao-embedding-vision, 探测) | dense-only(v1) | multimodal 管线(MM-1) | /library multimodal 路 |

跨 collection 混合检索 = 非目标(§7 P2: 两库向量空间不同)。

### 19.3 数据流全景(两入库两查询)

入库A(文本, 现有): EDGAR HTML → section tree → rag_nodes(PG) → Milvus rag_nodes(dense+text)
入库B(多模态, MM-1): PDF → dots.ocr/vLLM 页解析 → 分块 → VLM 描述(增强层) → ark 多模态向量 → rag_multimodal + rag_documents + 资产落盘
查询A(/ask, 现有): 问题 → scope 下推 → 混合检索 → rerank → 生成
查询B(/library): collection=text → 全库混合+RRF+rerank; =multimodal → ark 向量 → dense top-k → (可选)生成

## 20. 数据生命周期与一致性

### 20.1 文档删除级联矩阵 (delete_ingested_document 统一)

| 存储对象 | 文本文档(现有路径) | 多模态文档(MM-2 接缝#9) | 失败恢复 |
|---|---|---|---|
| PG rag_documents | 删行 | 删行(source=multimodal_pdf) | 幂等重跑 |
| PG rag_nodes | 删行 | 不涉及 | |
| Milvus rag_nodes | dense 删点(sparse 同行) | 不涉及 | |
| Milvus rag_multimodal | 不涉及 | `replace(id, [])` 删点 | |
| 资产(页图/插图) | 不涉及 | `assets.delete_document(id)` | **级联顺序: 先 Milvus 后 PG 后资产** — 检索面先失效(用户不再看到半删状态证据卡), 资产残留由重删或清理脚本兜底; 三步各自幂等, 无跨存储事务(接受最终一致, 残留只占磁盘不影响正确性) |

### 20.2 幂等与并发

| 操作 | 幂等机制 | 并发防护 |
|---|---|---|
| 文本入库 | 同 document_id 重跑 replace(现有) | 人工分配 ID 段 |
| 多模态入库 | 同 document_id 先删点再全量写(§7-5) | **同 ID 并发跑 = 竞态**: v1 加运行前检查 — `rag_documents` 已存在同 ID 且 source=multimodal_pdf → 拒绝并提示 `--replace` 语义(先删后入); 不做分布式锁(单机离线 CLI 场景过度设计) |
| 文档删除 | 各步幂等(上表) | 重复执行无害 |

## 21. 回滚与发布预案

| 场景 | 回滚动作 | 成本 |
|---|---|---|
| 多模态后端缺陷影响文本主链 | `DENSE_BACKEND=milvus` 本就是默认 — **零改动边界保证文本链路从未变过**, 无需回滚动作 | 0(设计保证) |
| /library 缺陷 | 前端下掉入口或 server 移除 router 注册(1 行) | 分钟级 |
| 多模态数据污染 | delete 工具清 document + 资产; collection 可整删重建(`ensure_collection` 幂等) | 取决于入库量 |
| 模型故障(ark/zhipu) | env 切备选链(§2 实测矩阵), embedding 换模型须重建 collection(维度校验拦截误配) | 分钟级 + 重嵌成本 |
| 发布顺序 | MM-1(纯增量, 默认 env 不激活) → MM-2(默认 env 下 /ask 冒烟+70 单测回归为发布门禁) → MM-3 | 每步可独立回退 |

## 22. 可观测性设计

| 层 | 机制 | 覆盖 |
|---|---|---|
| 结构化日志 | `log_rag` 现有机制, 多模态新增 stage: parse/chunk/describe/embed/upsert(§18) | 入库全链路每文档 |
| 请求追踪 | trace_id + latency_ms(沿用现有中间件), /library 响应契约内建 | 两查询路 |
| LLM 观测 | langfuse: /ask 现有覆盖; /library 生成调用纳入同一 client 包装 | 生成+VLM 描述(批量入库的 VLM 调用记 langfuse, 关联 document_id) |
| 质量门禁 | m4_benchmark.py 基建复用: MM-2 后跑 text 路零回归; 多模态 R4 指标(MultiModalFaithfulness)接入后纳入 | 发布门禁+趋势 |
| 资源水位 | Milvus count(/collections 端点) + 资产目录体积(CLI 摘要输出) | 容量规划 |

## 23. 安全与密钥矩阵

| env | 用途 | 端点 | 暴露面 |
|---|---|---|---|
| OPENAI_API_KEY(ark) | 生成/多模态向量/VLM 描述 | ark plan | 后端 only |
| ZHIPU_API_KEY + ZHIPU_BASE_URL | GLM 备选通道(coding plan) | bigmodel coding | 后端 only |
| OPENROUTER_API_KEY | 文本 embedding(现有) | openrouter | 后端 only |
| BOCHA_API_KEY | rerank 回退(现有) | bocha | 后端 only |
| DASHSCOPE_API_KEY | 多模态 embedding 备选(预留, 空) | dashscope | 后端 only |
| DOT_OCR_API_KEY | vLLM 解析(默认占位 0) | 自起 vLLM | 内网 |
| LANGFUSE_* | 观测(现有) | langfuse | 后端 only |

密钥纪律: 仅经 env(参考仓库硬编码 key 为反面教材, §9); `.env` 不入 git(现状); 前端仅经同源代理访问后端, 密钥零前端暴露。
第五部分上传安全(届时实现): PDF magic bytes 校验(%PDF 头)+扩展名白名单+大小上限(§16 同源 200MB)+并发入库互斥(§20.2)+page-image 防穿越(§6 资产接口已内聚)。

## 24. 需求追踪矩阵 (验收 ↔ 测试 ↔ 里程碑)

| 验收(§10) | 测试来源(§17) | 里程碑 |
|---|---|---|
| 1 零改动回归 | 70 存量单测 + /ask 冒烟 | MM-2 发布门禁 |
| 2 env 切换 | factory 混配矩阵单测 | MM-2 |
| 3 混配 fail-fast | 同上 | MM-2 |
| 4 入库幂等/降级 | backend FakeClient 幂等 + CLI 端到端(含停 vLLM) | MM-1 |
| 5 查询图文证据 | library_service 单测 + curl 验收 | MM-2(+MM-3 UI 面) |
| 6 新单测六类 | §17 表全部 | MM-1/MM-2 各自交付 |
| bbox 坐标验证(§14.1-1) | 真 PDF 首跑人工比对 | MM-1 第一周 |
| 资产两期迁移(§11) | 抽样字节校验 | 二期 |
