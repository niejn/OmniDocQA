# 多模态 RAG 与 Milvus 迁移需求文档 (v1.17)

- 日期: 2026-09-11
- 状态: 第一部分完成并定论(dense=Milvus 唯一, sparse=milvus 默认|postgres; M4 接受误差; qdrant/opensearch 已移除); 第二~五部分未进入开发
- 参考: `D:\mashibing\RAG_RAGAS\code` (DOTS·OCR 解析器 + Multimodal_RAG 链路 + Milvus collection 建法)

## 0. 背景与目标

**项目愿景升级(2026-09-09)**: 从 SEC 财报专用 RAG 系统 → **通用文档智能问答平台**。用户可提交任意 PDF(书籍/课件/报告/说明书), 系统切分入库到向量库, 用户按文档提问获取答案； 同时用 RAGAS 从入库的 PDF 自动生成评测集 → 验证 RAG 质量 → 驱动模型/组件选型。

```
用户提交 PDF(书籍/文档/报告)
  → 切分+向量化 → 存入 Milvus collection
  → 用户按文档/全库提问 → 检索+生成答案
  → RAGAS TestsetGenerator 从 PDF 自动生成评测题
  → RAGAS 指标验证检索/生成质量
  → R5/R6/R7 驱动 LLM/reranker/embedding 选型与回归防护
```

分五个部分实施:

1. **第一部分(先行, 分步)**: 向量存储迁到 Milvus。1A 稠密(本期)→ 1B 稀疏 → 1C 融合(可选)。Postgres `rag_nodes` 表结构零变更。
2. **第二部分**: 多模态 PDF 入库管线(通用解析, 不限于 SEC)→ 独立 Milvus collection; 全库查询 API。
3. **第三部分**: 前端 `/documents` 全库查询页面(通用文档 QA 入口)。
4. **第四部分**: RAGAS 评测体系(R1 修复→R2 数据集→R3 指标→R5 模型选型→R6 组件 A/B→R7 持续评测)。
5. **第五部分(新增)**: 通用化改造 — 前端上传 PDF 入库 + 动态 collection 管理 + 从 PDF 自动生成评测集。

现有 finance 问答管线(`/agent/api/ask/generate`, document_ids 必填)**行为零回归**。

## 1. 总体架构与顺序

| 部分 | 内容 | 依赖 |
|---|---|---|
| 第二部分: 多模态后端 | DOTS·OCR/vLLM(+fitz 降级)入库 → 复刻参考分块 → 多模态 embedding → Milvus 新 collection; 全库查询 API | 第一部分 |
| 第三部分: 前端 | `/documents` 页 + 导航 + collection 选择 + 双模式 | 第二部分 API |
| 第四部分: RAGAS 评测体系重构(§6, 高优先级) | R1 修环境(阻塞 M4) → R2 数据集 LLM 增强 → R3 全量指标 → R4 专项 | R1 立即; R2 与 1B 并行 |
| 第五部分: 通用化改造(§8.5) | 前端上传 PDF + 动态 collection + 从 PDF 自动生成评测集 | 第二/三/四部分 |

```mermaid
flowchart LR
  subgraph 存储层
    M[Milvus<br>rag_nodes: dense COSINE + BM25 稀疏<br>text=BM25 输入副本 + 过滤标量<br>rag_multimodal: 多模态 collection]
    P[(Postgres rag_nodes 表结构零变更<br>节点事实源: text/title/metadata/树关系<br>业务表 sec_financial_observations 等不动)]
  end
  APP[检索/问答管线<br>NodeHybridRetriever] -->|node_id + score<br>factory 选择 backend| M
  APP -->|上下文/证据/兄弟扩展<br>按 id 回表| P
  U[前端 /documents] --> LIB[/agent/api/documents/*] --> M
```

核心原则:
- **PG 仍是节点数据唯一事实源** — 上下文组装、证据抽取、章节树遍历、兄弟扩展全部照旧读 PG。
- **Milvus 是纯向量检索层** — 命中返回 `node_id + score`, 正文回 PG 取。
- 迁移是存储层替换, 不是算法重构; 融合位置切换(M6)是独立门禁的单独阶段。

## 2. 参考实现 → 本项目映射

| 参考代码 (RAG_RAGAS/code) | 内容 | 本项目落点 |
|---|---|---|
| `dots_ocr/parser.py` `DotsOCRParser` | vLLM 服务逐页解析 PDF → 每页 `.md`+`.jpg`+`.json`, dpi=200, 线程池并行 | 新解析器, vLLM 地址配置化, fitz 降级 |
| `Multimodal_RAG/splitters/splitter_md.py` | 标题分块(H1–H3)→ 抽 base64 插图单独成块 → 超长块语义分块(percentile)→ 标题层级补全(`Header 1 --> Header 2`) | 分块策略完全复刻 |
| `Multimodal_RAG/milvus_db/db_operator.py` `generate_image_description` | VLM(qwen-vl-plus)结合前后文生成图片描述(≤300 字) | 复用 `QWEN_API_KEY`(OpenAI 兼容接口) |
| `Multimodal_RAG/utils/embeddings_utils.py` | DashScope 多模态 embedding: 文本块 `{text}`, 图片块 `{image, text}` 联合向量; 固定窗口限流 120RPM + 429 指数退避(5 次, base 2.0s) | 复刻限流/重试; 模型与维度可配置 |
| `创建一个Collection.py` / `collections_operator.py` | Milvus 混合 collection(BM25 Function + SPARSE_FLOAT_VECTOR + AUTOINDEX/IP) | 第二部分多模态 collection 直接复刻建法(PK 改 UUID, 非 auto_id) |
| 硬编码 API Key / Milvus 凭据 | 参考代码缺陷 | 禁止拷贝, 全部走 `.env` |

## 3. 第一部分: Milvus 迁移

### 3.0 分步计划(v1.5 起)

| 步骤 | 范围 | 切换开关 | 状态 |
|---|---|---|---|
| **Step 1A** | 仅稠密: Qdrant → Milvus; 稀疏仍 `SPARSE_BACKEND=postgres`(tsvector), OpenSearch 不动 | `DENSE_BACKEND=milvus` | **代码已落地(2026-09-09)**: M1/M2 完成, env 已切 milvus; M4 质量验收未跑(见 §3.6) |
| **Step 1B** | 稀疏: PG tsvector/OpenSearch → Milvus BM25(`MilvusSparseBackend`) | `SPARSE_BACKEND=milvus` | **已落地(2026-09-10, 用户决策先于 M4 执行)**: `milvus_store.sparse_search`(原始查询文本→`anns_field=sparse`, analyzer 服务端分词)+ `retrieval_backends/sparse_milvus.py`(async 包装, replace 为 no-op——行由 dense 路径统一写入)+ factory `milvus` 分支(强制 `DENSE_BACKEND=milvus`, 混配 fail-fast)+ `delete_ingested_document.py` 补 milvus 分支(含 1A 遗留的 dense=milvus raise 修复)+ env 切换。`SparseQueryPlan` 字段加权无法映射单 BM25 字段→按 §3.9 预案在 stage log 记 `query_plan_applied=false`。验证: 54 单测全绿(新增 7 项); 真机冒烟: 68 文档(9801–9870)BM25 命中相关性正常(summary 5.0ms/leaf 4.9ms), E2E `ask-multi` 检索链路与 OpenSearch 对照 `retrieve_done` 逐位一致(dense 43/sparse 43/12 节点, 生成阶段 LLM 404 与后端无关); **M4 评测门禁(±2%/重叠度≥90%)未跑, 质量验收待补** |
| Step 1C(可选) | 融合下沉 Milvus(`SupportsHybridSearch` + `hybrid_search`) | `FUSION_BACKEND=milvus` | 待 M6 门禁 |

关键设计: Milvus collection **schema 一次建全**(含 text + BM25 sparse 字段), Step 1A 迁移时即写入 text(Milvus BM25 Function 随 insert 自动生成稀疏向量)— Milvus schema 不可变, 1B 因此无需重建 collection/二次迁移, 只需新增 backend + 切 env。1A 的评测变量唯一: 仅稠密源变化(Milvus COSINE 与 Qdrant COSINE 分数语义一致)。

### 3.1 目标与边界

- 稠密向量: Qdrant(`rag_nodes` collection) → Milvus 同名 collection, **PK = 节点 UUID(VARCHAR, auto_id=False)**, 与 Postgres `rag_nodes.id` 对齐(参考项目用 INT64 auto_id, 不可照抄)。
- 稀疏检索: Postgres tsvector(`node_repository.sparse_search`)与 OpenSearch 全部废弃 → Milvus **BM25 Function**(`SPARSE_FLOAT_VECTOR` + `SPARSE_INVERTED_INDEX`)。
- **Postgres 零 DDL**: `rag_nodes` 保留全部列(含 `text, title, metadata, search_vector, has_vector`)。`search_vector` 列在切换后不再被查询路径使用, 但列保留(GENERATED STORED, PG 写入自动维护, 无需处理)。
- Analyzer: 默认 `english`(语料为 EDGAR 英文), 可配置 `jieba`(中文课件场景)。现网 PG 用 `simple`、OpenSearch 有 finance 调优 profile — 迁移验收用评测集对比, 不达标再调 analyzer。
- 问答管线逻辑(finance 路由、RRF 融合、本地 rerank、兄弟扩展、上下文预算)行为不变。

### 3.2 实现方式: factory + 协议扩展

走现有 `retrieval_backends/factory.py` 缝隙, 不改管线调用结构:

> 分步交付: **Step 1A 只需 `MilvusDenseBackend`(search + replace_document_nodes)**; `MilvusSparseBackend` 属 Step 1B, `SupportsHybridSearch`/`hybrid_search` 属 Step 1C(§3.0)。本节描述最终形态。

**协议层**(`retrieval_backends/types.py`, 旧协议原样保留):

```
DenseBackend(Protocol)          # 不动: search() + replace_document_nodes()
SparseBackend(Protocol)         # 不动: search() + replace_document_nodes()
SupportsHybridSearch(Protocol)  # 本期新增, 可选能力:
    hybrid_search(query_vector, query, *, document_ids, limit,
                  levels, parent_ids, metadata_filters,
                  log_stage) -> list[NodeHit]
```

**实现层**:
- `MilvusDenseBackend` 实现全部三个方法(dense ANN / 写入 / 融合检索)。写入无需额外逻辑: Milvus BM25 Function 在 insert 时自动从 `text` 生成稀疏向量, `replace_document_nodes` 插 dense+text 即同时维护两个索引。
- `hybrid_search` 内部一次调用 `client.hybrid_search(reqs=[dense AnnSearchRequest, sparse AnnSearchRequest(原始查询文本)], ranker=RRFRanker(k=现网 RRF 的 k), filter=过滤表达式, limit)` — 服务端融合, 单次往返。
- `MilvusSparseBackend` 独立实现(BM25 文本检索, 查询侧用原始查询文本搜稀疏字段, 实现时按 pymilvus 版本验证 API), 承担两路回退路径。

**管线层**(`llamaindex_retrieval.py`, 唯一改动点):
- `NodeHybridRetriever` 在 4 组混合调用点前判断: `isinstance(self.dense_backend, SupportsHybridSearch) and config.fusion_backend == "milvus"` → 一次 `hybrid_search`; 否则现有两路 + 应用层 RRF **原样保留为回退**。
- dense-only 调用点(section-tree 种子、narrative 目标)继续 `search()`, 零改动。

**factory**: 不变, `get_dense_backend()/get_sparse_backend()` 各加 `milvus` 分支。

factory 的价值(决策依据, 记录备查):
1. `NodeHybridRetriever`(2100+ 行)有 4 处 dense + 4 处 sparse 调用点, 经 factory 换实现则管线 8 个调用点零改动。
2. 配置级回滚: `DENSE_BACKEND=milvus ↔ qdrant`, 迁移期旧数据未删, 改 env 即回退。
3. 写路径(ingest/`reindex-vectors`/删除文档)经 `replace_document_nodes` 自动切换。
4. 仓库既有惯例, 单测可按协议 mock。

### 3.3 Milvus schema(`rag_nodes`)

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR PK(=节点 UUID) | 非 auto_id |
| `document_id` | INT64 | 过滤主键, 建索引 |
| `parent_id` / `level` | VARCHAR / INT64 | 现网过滤语义 |
| `retrieval_fields` | JSON | `_retrieval_fields` 过滤字段副本, 过滤表达式对齐现网 |
| `text` | VARCHAR(≤65535, 启用 analyzer) | BM25 Function 输入, 检索副本; 非事实源 |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function(`text_bm25_emb`)生成 |
| `dense` | FLOAT_VECTOR(dim=EMBEDDING_DIMENSION, **COSINE**) | 与 Qdrant 分数语义一致, 不改用参考项目的 IP |

不含 `title/node_type/order_index/metadata` 全量副本 — PG 是事实源, 不需要。

Step 1A 仅启用 `dense` 检索; `text`/`sparse` 字段随 1A 迁移一并写入(schema 一次建全, 供 1B 启用 BM25 检索)。

Analyzer: 默认 `english`(语料为 EDGAR 英文), 可配置 `jieba`(中文课件场景)。现网 PG 用 `simple`、OpenSearch 有 finance 调优 profile — 迁移验收用评测集对比, 不达标再调 analyzer。

### 3.4 配置与依赖

| 新增 env | 默认 | 说明 |
|---|---|---|
| `MILVUS_URI` | `http://127.0.0.1:19530` | 参考项目同款连接方式 |
| `MILVUS_USER` / `MILVUS_PASSWORD` / `MILVUS_DATABASE` | — / — / `default` | 凭据必须走 env(参考代码硬编码密码禁止带入) |
| `MILVUS_COLLECTION` | `rag_nodes` | |
| `MILVUS_TEXT_ANALYZER` | `english` | `english\|jieba` |
| `FUSION_BACKEND` | `app_rrf` | `app_rrf\|milvus`。默认应用层融合; `milvus` 仅 M6 门禁通过后启用 |

依赖: +`pymilvus`(≥2.5, BM25 Function 所需; 与服务端 v2.6.22 兼容)。M5 后移除: `-qdrant-client`(opensearch 客户端保留至 1B)。

依赖边界: 参考项目虽用 LangChain 编排, 但其 Milvus 读写(`collections_operator.py`/`db_operator.py`/`db_retriever.py`)本身是裸 `pymilvus`; 本项目无 LangChain, 对接 Milvus **仅新增 `pymilvus`, 不引入 LangChain / langchain-milvus**(自研 backend 协议需精确控制过滤/删除/BM25 schema, 官方封装反而加黑盒)。LangChain 仅在第二部分分块处可能出现(§4.2"锁效果不锁依赖", 届时单独评估 `langchain-text-splitters`)。

### 3.5 docker-compose 与 env 配置 — 已完成(2026-09-09)

**两个 compose 文件均已加入 Milvus 三容器**(镜像 pin 自官方 v2.6.22 standalone compose): `milvusdb/milvus:v2.6.22` + `quay.io/coreos/etcd:v3.5.25` + `minio/minio:RELEASE.2024-12-18T13-15-44Z`; 命名卷 `rag_milvus_etcd_data / rag_milvus_minio_data / rag_milvus_data`; healthcheck + `depends_on: service_healthy`; 内存限制 etcd 512M / minio 1G / milvus 4G; 仅 19530 暴露宿主(`MILVUS_HOST_PORT` 可覆盖), etcd/minio 不暴露; minio 凭据 `MILVUS_MINIO_USER/PASSWORD` 可覆盖。`docker compose config` 双文件校验通过。

| 文件 | 说明 |
|---|---|
| `docker-compose.rag.yml` | 服务名 `etcd/minio/milvus` |
| `docker-compose.ragall.yml`(开发用) | 服务名 `milvus-etcd/milvus-minio/milvus`(避让 Langfuse 已有 `minio` 服务); **Milvus 对象存储独立于 Langfuse minio, 互不共享**; `restart: always` 对齐本文件惯例 |

- **Attu**(Milvus Web 管理界面, `zilliz/attu:v2.6.5`, 兼容 Milvus 2.6.x): 两个 compose 均已加入; `MILVUS_URL=milvus:19530`(compose 内服务名), 宿主端口 **8300**(`ATTU_HOST_PORT` 可覆盖, 8000 留给 FastAPI 后端); 启动 `docker compose -f docker-compose.ragall.yml up -d attu` → http://127.0.0.1:8300 (HTTP 200 已验证)。另: ragall 注释中 Langfuse 版本已更正为 v4(镜像即 `langfuse:4`)。

**env 已配置**:
- `src/agent/.env`: **`DENSE_BACKEND=milvus` 已生效(M2 起)**; Qdrant 块降级为回滚备份(M5 删除); `env.example` 模板默认亦为 milvus。
- `src/agent/env.example`: 同步补 Milvus 模板块。
- compose 插值变量(`MILVUS_HOST_PORT` 等)有内联默认值, 根目录无需 `.env`。
- ~~`config.py` 尚无 `MILVUS_*` 字段~~ **已落地(2026-09-09)**: config 字段、`milvus_store.py`、`dense_milvus.py`、factory `milvus` 分支、`pymilvus>=2.5,<3` 均已实现; 38 项单测 + 真机 Milvus v2.6.22 冒烟 7 项全过(建库/文档过滤排序/json_contains_any/level 过滤/幂等 replace/删除)。

qdrant/opensearch 本期保持默认启动(切换前仍是现网后端); **1A 验收后删 qdrant service/卷**, 1B 验收后删 opensearch(若在用)。

### 3.6 迁移计划(每阶段可回退; 原 M3 切换已并入 M2 第一行, 原 M2 迁移脚本已废弃)

| 阶段 | 内容 | 回退方式 |
|---|---|---|
| M1 并行准备 — **已完成(2026-09-09)** | `MilvusDenseBackend`(`retrieval_backends/dense_milvus.py`)+ `milvus_store.py`(schema 一次建全含 BM25, dense COSINE, document_id INVERTED 索引)+ factory `milvus` 分支 + config `MILVUS_*` + `pymilvus>=2.5,<3`; 单测 `tests/unit_tests/test_milvus_store.py`(11 项)+ 真机冒烟 7 项通过。不含 SparseBackend/HybridSearch(属 1B/1C)。现网不动(DENSE_BACKEND=qdrant) | 无需回退, 现网未动 |
| M2 重建 — **已完成(2026-09-09)** | 切 `DENSE_BACKEND=milvus` → `ingest-edgar-local --document-id-start 999001` 重跑 70 文档(同 ID 原地 replace, next_document_id=999071, success:false=0)→ 对账 PASS: PG has_vector 节点 7541 = Milvus 行数 7541, 逐文档数量零偏差, 53 个抽样 text 哈希零不一致; E2E 检索验证: 真实管线 dense_hits=60(Milvus)+ sparse_hits=45(OpenSearch)→ RRF → Bocha rerank → 兄弟扩展, 命中相关章节正常 | 改 env 回 qdrant(未重跑无损语义见 §3.9 回滚行) |
| M2-① 盘点结果(2026-09-09) | 70 文档全部 `sec_edgar_html`, 源 .htm 与 accession 一一对应, 不可重建清单为空; 规模 7541 向量节点; **勘误**: 999036–999066 老文档"向量少"非当年 embedding 故障 — 源 .htm 本身仅封面页(word_count≈550–660, element_count=1), 重入库确定性保持 2–3 节点; 999063/999065 源无文本 → 零向量, 与重建前一致 | — |
| M4 质量验收 — **已跑(2026-09-14), 部分达标** | 两轮 100 题+RAGAS 零失败: faithfulness 0.9145 vs 0.8954(+2.14% PASS, 正向); context_precision 0.6984 vs 0.7271(**-3.95% FAIL**)。归因: 单列 BM25 的 TF enrichment 无法完全复刻 OS 字段加权的头部排序质量。处置选项: 调 enrichment 次数重评 / 接受差距 / M6 WeightedRanker 补偿 | 不达标回 M2 前(改回 qdrant) |
| M5 收尾清理(1A) — **已完成(2026-09-15)** | 删 qdrant compose service/卷(两文件)+容器/卷; 删 `dense_qdrant.py`、`vector_store.py`(零引用死文件); factory milvus-only(default 改 milvus); 删 `qdrant-client` 依赖; config 删 QDRANT_* 字段; env.example 清理; delete 脚本/测试同步。70 单测全绿, 双 compose config 校验通过 | git revert |
| M5' opensearch 移除 — **已完成(2026-09-15, M4 定论后)** | M4 定论: **接受 context_precision -3.95%**(用户决策, 归因见 DEV_PROGRESS 09-14); 删 `sparse_opensearch.py`/`inspect_opensearch.py`/config opensearch_* 12 字段/factory 分支(sparse 默认 milvus, 保留 postgres 回退)/delete 脚本分支/`opensearch-py`/compose service+卷+容器。检索栈最终形态: **dense=Milvus 唯一, sparse=milvus(默认)|postgres** | git revert |
| M6 融合切换(= Step 1C, 可选, 1B 之后) | 前置: `RRFRanker(k)` 与现网 RRF 的 k/归一化对齐(不等价则调 k 或放弃); `SparseQueryPlan` 字段权重映射决策(见 3.9)。同一评测集 `app_rrf` vs `milvus` 对比, 指标不回退(±2%)且 P95 延迟下降方达标 → 切 `FUSION_BACKEND=milvus`; 不达标保持默认 | 改 env |

### 3.7 写路径改造

- `_store_nodes`: PG 写入完全不变; 向量写入从 Qdrant 换成 Milvus(delete by expr `document_id == X` + insert)。
- `has_vector` 含义改为"Milvus 已写入", 逻辑位置不变。
- 删除文档: PG 级联照旧 + Milvus expr 删除。
- `reindex-vectors` CLI 改写 Milvus。

### 3.8 第一部分验收标准

1. 重建对账: 每文档 Milvus 点数 = PG `rag_nodes` 中该文档 `has_vector=true` 节点数; 抽样 100 节点 PG vs Milvus text 哈希一致; 不可重建清单为空或已经用户确认接受。
2. 评测不回退: 现有评测集(`run_mixed_narrative_questions_parallel.py` 100 题 + `run_evaluate_pending_parallel.py`)迁移前后指标对比, 容差 ±2%; 另抽样 20 题做 top-8 检索重叠度 ≥90%。
3. `ask-multi` CLI 与 `/agent/api/ask/generate` 冒烟: 答案与证据正常; document_ids 过滤语义不变(含 level/parent/metadata 过滤)。
4. 重入库/删除文档幂等: 同文档重跑 Milvus 点数不变; 删文档后 Milvus 无残留点、PG 行级联删除。
5. `rag_nodes` 表结构零变更(迁移前后 `\d rag_nodes` 一致); 1A 的 M5 后 `docker compose ps` 无 qdrant(opensearch 保留至 1B)。
6. 单测: Milvus backend 过滤表达式映射(document/level/parent/metadata)、删除幂等。
7. `FUSION_BACKEND=app_rrf` 时管线行为与迁移前逐位一致 — 单测断言 hybrid 分支未进入(Step 1C 引入开关时执行)。
8. `hybrid_search` 单测: 双路结果与"分别调 dense/sparse 后应用层 RRF"在相同 k 下排序一致性; 过滤表达式在融合路径同样生效(Step 1C 时执行)。

### 3.9 第一部分风险

| 风险 | 缓解 |
|---|---|
| BM25 分词与 PG `simple`/OpenSearch profile 的召回差异 | 评测集门禁 + analyzer 可配; 不达标先调 analyzer 再验收 |
| Milvus 按需反查/过滤的性能 | 过滤字段建索引; 验收含延迟对比(latency_ms 不明显回退) |
| text 在 PG 与 Milvus 双份, 存在不一致可能 | 唯一写路径 `_store_nodes → replace_document_nodes`(先删后插); M2 对账抽样比对哈希 |
| Milvus 三容器资源占用(etcd/minio) | compose 限内存; 文档注明最低配置 |
| `RRFRanker` 与应用层 RRF 语义差异(k、分数归一、tie-break) | M6 前对齐 k 并用排序一致性单测验证; 不等价即不启用 |
| `SparseQueryPlan` 领域字段权重无法映射到 Milvus 单 text 字段 BM25 | M1 起 Milvus 稀疏路径记日志标明 plan 未生效; **写入侧拼接已执行(2026-09-11, 用户决策先于 M4)**: `text=title×2+search_hints×3+正文`(TF 代理, `MILVUS_TEXT_TITLE/HINTS_REPEATS` 可配, 0=禁用), 前缀长度记 `metadata._milvus_text_prefix_chars` 读取时剥离(reranker 吃纯正文), 存量经 `scripts/milvus_rebuild_text.py` upsert 重建(7541 行对账零偏差); M4 仍为最终门禁 |
| 源数据缺失的文档无法重建(M2 新) | 执行前盘点出不可重建清单, 由用户决定补源/放弃; 未确认前不进入 M4 |

| 重入库产生新节点 UUID, 历史报告中的旧 node_id 引用失效(M2 新) | 仅影响旧报告回溯展示, 不影响检索/新报告; 可接受, 已记录 |
| 重算全库 embedding 的 API 成本(M2 新) | 用户已选此方案即接受; 可按 document_ids 分批执行控制节奏 |
| 回滚语义变化(M2 重建方案固有): 已重跑文档回切 qdrant 后 dense 失效(qdrant 旧点指向已被替换的旧节点 UUID) | 回滚仅对未重跑文档无损; 全量回滚 = 改回 qdrant 后用 qdrant 再重跑一遍(幂等); Qdrant 旧数据保留至 M5 仅作对照, 不承诺无损回切 |

### 3.10 Reranker 本地化(2026-09-09 落地)

`tools/rerank.py` 组合式选型(对管线透明, 调用点 `llamaindex_retrieval`/`server` 仅换 import):
- **`RERANKER_BACKEND=local`(默认)**: `tools/local_reranker.py` — sentence-transformers CrossEncoder, 复用 `tools/qwen_reranker.py`(原 qwen_reranker_demo, 已更名转正)的加载器(BNB 4/8bit 可选, CUDA)与 Sigmoid 打分; 懒加载+失败缓存+启动预热(server startup 后台 create_task, 锁防重复加载); 模型 = **本地 4B**(`LOCAL_RERANKER_MODEL` 指向 `C:/Users/julien/.cache/huggingface/hub/Qwen3-Reranker-4B`, 7.6GB 完整缓存, RTX 5080 bf16); 备选 `Qwen/Qwen3-Reranker-0.6B`(HF 自动下载)。
- 验证: 9 项单测 + **0.6B/4B 双真模型冒烟 PASS**(4B: 加载 13.4s 一次性, 纯推理 56ms, liquidity 段 0.995 居首/cover 淘汰)。
- `RERANKER_BACKEND=none` 可显式关闭（bocha 远程后端已于 2026-09-16 移除）。
- 注意: API 服务首个 rerank 请求会触发模型加载(~20s), 可选启动预热; Windows 下 HF 下载用 `HF_ENDPOINT=https://hf-mirror.com`。

全部 47 项单测通过。

> **v2 实施设计已定稿(2026-09-15)**: 见 `MULTIMODAL_RAG_PART2_DESIGN.md` — 基于 dots.ocr 源码级调研细化本文 §4, 并新增硬约束: 现有密集向量链路零改动、`DENSE_BACKEND=milvus_multimodal` env 切换。冲突处以 v2 为准。

## 4. 第二部分: 多模态后端

### 4.1 新增配置

| env | 默认 | 说明 |
|---|---|---|
| `DOT_OCR_BASE_URL` | 空 | vLLM 服务地址(如 `http://host:6006`); 空=直接降级 |
| `DOT_OCR_MODEL` / `DOT_OCR_DPI` / `DOT_OCR_MAX_THREADS` | `dots_ocr` / `200` / `16` | 复刻参考参数 |
| `DOT_OCR_FALLBACK_FITZ` | `true` | 不可达时降级为 fitz 文本解析 |
| `MULTIMODAL_COLLECTION` | `rag_multimodal` | Milvus 新 collection |
| `MULTIMODAL_EMBEDDING_MODEL` | `multimodal-embedding-v1` | DashScope 多模态表征(参考项目亦实测 `tongyi-embedding-vision-flash-2026-03-06`) |
| `MULTIMODAL_EMBEDDING_DIM` | `0`=自动探测 | 首次调用取实际维度建 collection; 显式配置则强校验 |
| `MULTIMODAL_EMBED_RPM` / `_MAX_RETRIES` / `_BACKOFF_BASE` | `120` / `5` / `2.0` | 复刻限流与退避 |
| `MULTIMODAL_VLM_MODEL` | `qwen-vl-plus` | 图片描述, 复用 `QWEN_API_KEY` |
| `MULTIMODAL_PAGES_DIR` | `tools/data/multimodal_pages` | 页图+插图落盘根目录 |
| `DOCUMENT_ASK_DEFAULT_TOP_K` | `8` | 全库查询默认 top_k |

新增依赖: `dashscope`(多模态 embedding SDK)、`dots_ocr`(进 `requirements.txt` 前评审体积)。**不新增**: LangChain 全家(编排=普通函数, 分块=自实现, LLM/VLM=现有客户端, 见 §4.2 与 §3.4 依赖边界)。

### 4.2 入库管线(新 CLI `scripts/ingest_multimodal_pdf.py`)

`--data-dir --glob "*.pdf" --document-id-start N`, 六步:

1. **解析**: 探测 `DOT_OCR_BASE_URL` 可达 → DOTS·OCR 逐页输出 md/jpg(线程池并行, 页序排序); 不可达且降级开启 → fitz 提取文本(`## 第 N 页` 前缀)+渲染整页 jpg, 无插图抽取(无版面信息), 仅文本块向量化, 页图仅作展示。
2. **分块(效果完全复刻参考, 依赖自实现)**: 每页 md → 标题分块(逐行扫 H1–H3 边界, ~60 行自实现)→ 正则抽 base64 插图(md5 命名落盘 `MULTIMODAL_PAGES_DIR/{document_id}/`)单独成 image 块、正文去图 → 超 1000 字做语义分块(切句→相邻句余弦→percentile 断点, 复用现有 `generate_embedding`, ~100 行自实现)→ 标题层级补全拼 `title`(`Header 1 --> Header 2 --> Header 3`, 移植参考 `add_title_hierarchy` ~40 行)。**不引入 LangChain**(参考的 `MarkdownHeaderTextSplitter`/`SemanticChunker`/`Document` 均为可自实现的教学级算法, 且引入将拖进 langchain-core 依赖树); 兜底: 仅当自实现对齐失败时评估引入轻量纯文本包 `langchain-text-splitters`。
3. **图片描述**: image 块带前后文调 VLM 生成 ≤300 字描述(复刻参考 prompt: 结合前文/后文综合分析; 调用用项目现有 OpenAI 兼容客户端 + image_url content block, 不经 ChatOpenAI/HumanMessage)。
4. **向量化**: text 块 `{text: title+"："+正文}`; image 块 `{image: base64, text: 描述}`; 固定窗口限流 + 429 指数退避重试。
5. **存储**:
   - `rag_documents` 新行(metadata 标 `source=multimodal_pdf`, 记录页数/块数)— catalog、删除工具复用;
   - Milvus `rag_multimodal` collection 按参考项目建混合 collection: `id`(VARCHAR PK=块 UUID, 非 auto_id)、`document_id`、`filename`、`title`、`kind`(text|image)、`page_no`、`text`(BM25+jieba analyzer)、`category`、`image_ref`、`sparse`(BM25 Function)、`dense`(dim=多模态模型探测维度); v1 查询仅用 dense, BM25 字段为后续预留;
   - 幂等: 重跑同一 PDF 先按 `document_id == X` 表达式删点, 再全量写。
6. **可观测**: 复用 `log_rag` 记录各 stage; 失败页/失败块清单输出。

### 4.2.1 查询流程总览(两种模式)与图片的角色

```
问题 → ① embedding → ② Milvus 全库向量搜索(不指定文档)
     → 命中混排列表: [文本块…, 图片块…](按分数降序)
     ├── generate_answer=false(仅检索): 直接返回命中列表, 不调 LLM
     └── generate_answer=true(默认, 标准 RAG):
            ③ context = 命中块的【文字】(文本块正文 + 图片块的 VLM 描述)
            ④ DEFAULT_MODEL 生成答案 → ⑤ 前端: 答案卡 + 证据卡
```

图片在三个阶段的角色(关键设计):

| 阶段 | 处理 | 结果 |
|---|---|---|
| 入库时 | 抽插图 → VLM 生成 ≤300 字描述 → 图+描述联合向量化 | 带向量的图片块 |
| 检索时 | 与文本块同向量空间, 问题文本可直接命中图片 | 平等候选 |
| 生成答案时 | **LLM 不看原图**, context 中图片 = 其 VLM 文字描述 | 一段文字 |
| 前端展示 | 证据卡显示原图缩略图(page-image 接口)+ 描述 + 页码 | 原始 jpg |

即: LLM 消费图片的文字化描述, 原图只给用户看; LLM 直接看图生成(qwen-vl 类模型)为 §7 P2。

### 4.3 全库查询 API(新 router `/agent/api/documents`, 不动 ask_api)

| 端点 | 契约 |
|---|---|
| `POST /agent/api/documents/ask` | `{question: 1–1000字, collection: "text"\|"multimodal"(必填), top_k: 1–50 默认8, generate_answer: bool 默认true}` |
| `GET /agent/api/documents/collections` | `[{id, label, milvus_collection, points_count, document_count}]`(Milvus count + rag_documents), 供前端下拉 |
| `GET /agent/api/documents/page-image?document_id=&name=` | 返回 jpg; 校验解析后路径必须位于 `MULTIMODAL_PAGES_DIR/{document_id}/` 内(防穿越) |

- **multimodal 检索**: 问题 → 多模态 embedding(纯文本)→ `rag_multimodal` dense top-k, 无任何预过滤。
- **text 全库检索**: 落 Milvus — 空过滤表达式即全库, dense+BM25 双路 → 应用层 RRF; 不做 sibling/section-tree 扩展(依赖单文档上下文); 上下文按 `CONTEXT_CHAR_BUDGET` 拼接。
- **generate_answer=true**: context = 命中块文本(image 块用描述+title)→ `DEFAULT_MODEL` 生成, 证据引用 `[文件名:p页]`; false=跳过 LLM。
- 响应: `{question, collection, answer?, evidence: [{kind, document_id, filename, title, page_no?, score, text_preview, image_url?}], trace_id, latency_ms}`。
- finance SQL 路由、`/ask/generate`(document_ids 仍必填)不参与。

#### 4.3.1 接口模型明细

文件落点(镜像 `tools/asks/ask_api.py` 组织): `src/agent/tools/documents/document_api.py`(router, prefix="/api/documents")+ `document_service.py`(业务编排, 角色=简化版 rag_service, 不含 finance 路由/SQL 证据/section-tree)。

```
DocumentAskRequest:
  question: str          # 1–1000, 必填
  collection: "text" | "multimodal"   # 必填
  top_k: int = 8         # 1–50
  generate_answer: bool = true        # false = 仅检索

DocumentAskResponse:
  question, collection, answer(str|null), trace_id, latency_ms
  evidence: [ {kind: "text"|"image", document_id, filename,
               title, page_no, score, text_preview,
               image_url(仅 image 块)} ]

GET /collections 返回:
  [{id, label, milvus_collection, points_count, document_count,
    available}]   # multimodal 未建/为空 → available=false, 前端置灰
```

内部流程: text 路 = 现有 vectorizer 文本向量 → Milvus dense + BM25 双路(均无 document 过滤)→ 应用层 RRF → 本地 CrossEncoder rerank(复用单例)→ CONTEXT_CHAR_BUDGET 拼接; multimodal 路 = DashScope 多模态 embedding(纯文本)→ rag_multimodal dense top-k, image 块以 VLM 描述作上下文; generate_answer=true 时 DEFAULT_MODEL 生成, 证据标 [文件名:p页]。

错误契约: 422 参数校验; collection 未初始化 → 404; embedding/Milvus 上游故障 → 502; 无命中 → 200 + evidence=[] + answer=null。

page-image 端点: 解析为 `MULTIMODAL_PAGES_DIR/{document_id}/{name}`, `Path.resolve()` 后必须仍以该目录为前缀(防穿越), FileResponse 按扩展名给 content-type, 缺失 404。

### 4.4 非功能需求

- **维度防护**: 首个多模态向量返回后校验维度 = collection 维度, 不符 fail-fast 并报清晰错误。
- **密钥安全**: 全部走 env; 参考代码中的硬编码 DashScope key/Milvus 凭据严禁带入。
- **长文本截断**: 块文本超过多模态 embedding API 上限时截断并在 payload 记 `text_truncated: true`。
- Windows 路径兼容; 页图 dpi=200 的磁盘占用在文档注明。

### 4.5 第二部分验收标准

1. 课件 PDF 入库: `rag_documents` +1 行, `rag_multimodal` 点数 = 文本块+图片块数; 重跑同文件点数不变(幂等)。
2. 停 DOTS·OCR + 开降级: 入库成功(纯文本块)。
3. `curl POST /agent/api/documents/ask {collection:"multimodal", question:"有界流和无界流的定义", generate_answer:true}` → 200, answer 非空, evidence 含相关页, 其 `image_url` 可 GET 到图片字节。
4. `collection:"text"` 不传 document_ids → 200 全库命中; 同时 `/agent/api/ask/generate` 缺 document_ids 仍 422(零回归)。
5. 单测: 维度校验失败路径、429 退避逻辑、路径穿越拒绝。

## 5. 第三部分: 前端

### 5.1 导航与新页面

- 新建共享 Header 组件(两页复用), 链接"文档问答(`/`) / 全库查询(`/documents`)" — 现为单页无导航, 需补此组件。
- 新页面 `src/frontend/src/app/documents/page.tsx`。

### 5.1.1 隔离边界(互不干扰的硬约束)

- 前端: `/documents` 独立路由页面, 不 import 现有 `HomePage` 的状态逻辑; 原首页 `/` 唯一改动是挂共享 Header。
- 后端: 新 FastAPI router `/agent/api/documents`(独立文件注册到 server.py), 不 import/修改 `ask_api.py`; `/agent/api/ask/*` 契约(含 document_ids 必填)零改动。
- 语义: "全库查询" = **选定单个 collection 内**不指定文档/分组; 跨 collection 混合排序(两库 embedding 模型与维度不同)为 §7 P2 非目标。

### 5.2 查询交互

- collection 下拉(native `<select>`, 数据来自 collections 端点, 沿用 `DocumentScope.tsx` 的 select 模式); 无 DocumentScope/文档选择。
- 问题输入 + Enter 提交、top_k(沿用 `page.tsx` 交互与 clamp 逻辑)。
- 模式开关: "生成答案 / 仅检索"分段按钮(沿用 `DocumentScopeMode` 切换样式)。

### 5.2.0 标签筛选器(filter 生效与查询数据流, 2026-09-15 定稿)

**数据流**:

```
① 页面挂载 ──GET /documents/filters──▶ 标签面 {books[{id,title,chapters[]}], kinds[]}
② 用户点选 chips ──▶ 前端状态 filters={books?,chapters?,kinds?}(或选中集合 set_id)
③ 点[提问]/Enter ──POST /documents/ask {question, collection, top_k, filters|set_id}──▶
④ 后端 filters → Milvus 标量表达式下推(检索热路径不碰 PG) ──▶ evidence 仅含 filter 范围内 chunk
⑤ 激活 filter 数量 Badge("筛选 2") + [清除全部]; 空结果时提示"当前筛选范围内无命中, 试试放宽"
```

**交互规则**:

| 规则 | 行为 |
|---|---|
| 全量常驻(2026-09-15 修正, 否决级联) | **书与章始终全部可见可选, 无"先选书才展章"门槛** — 按章筛是独立诉求(例: 只要某几章, 不限书); 形态 = 按书分组的折叠面板(默认全展开, 章多时区内滚动+搜索框过滤标签名); 选中书**不约束**其章 — 二者独立勾选 |
| kind | 筛选面板 kind(全部/文本/图片)是**检索前下推**(改检索本身); 与结果区过滤 chips(§5.3, 显示层)并存且视觉区分 — 面板放检索区上方, 结果 chips 在证据区标题行 |
| 归一语义 | 书/章勾选原样随请求下发(books/chapters), **后端 document_service 单点归一**为 document_id 并集(选书=该书全部章; 选章=单个 doc; 去重) → `document_id in [...] and kind in [...]`, 无书章交集陷阱(选书A+书B第c章 = A全部 ∪ B.c章)。归一放后端: filter 型集合动态跟随需同一展开逻辑(§6.2), 单点可测 |
| 触发时机 | filter 变更**不自动重查**(避免连点打爆), 下次[提问]生效; 若已有结果, 面板显示"筛选已变更, 重新提问生效"提示条 |
| 集合 | "保存为集合"按钮(当前 filter 组合命名保存) → "自定义集合"下拉(含 chunk 计数/失效数); 选中集合 = set_id 检索(面板其余 chips 置灰禁用, 二选一语义) |
| 持久化 | filters/选中集合/collection 记 localStorage, 刷新恢复; URL 参数化(可分享筛选链接)为 P2 |
| 边界 | 标签面为空(无多模态文档)→ 面板显示"暂无书籍标签, 先入库 PDF"; 已选书/章在最新 /filters 中消失(被删)→ 挂载时清洗选中态并 toast 提醒; collection=text 时面板整体置灰(filters 仅 multimodal 路, text 路忽略) |

### 5.2.1 组件树与线框

```
app/documents/page.tsx             # "use client", 页面骨架 + 状态机 idle→loading→result|error
components/DocumentControls.tsx   # collection 下拉 + 模式开关 + top_k
components/FilterBar.tsx         # 标签筛选器: 书/章级联 chips + kind 分段 + 集合下拉 + 保存为集合 + 清除全部
components/EvidenceCard.tsx      # text/image 两种变体(image 变体含 <img> 懒加载缩略图 + 勾选框入集合)
components/SaveSetDialog.tsx     # 命名保存集合(新名/选已有枚举集合)
components/Header.tsx            # 共享导航(原首页唯一改动点)
lib/api.ts                       # +documentAsk(filters/set_id) +fetchDocumentFilters() +sets CRUD
```

行为: collection 下拉数据来自 /collections, `available=false` 选项置灰标"(未初始化)"; 默认选第一个可用库并记 localStorage; Enter 提交、提交中禁用; 结果整体替换, v1 无历史记录/多轮会话; filters 状态同记 localStorage(挂载时对照最新标签面清洗失效项)。

```
┌────────────────────────────────────────────────┐
│ RAGAS·Finance        [文档问答] [文档库查询 ●]     │ ← Header
├────────────────────────────────────────────────┤
│ 检索库 [多模态库 ▼]  (生成答案|仅检索)   topK[8] │ ← DocumentControls
│ 筛选 [全部|文本|图片]  自定义集合[无 ▼] [存为集合]│ ← FilterBar(激活时 Badge"筛选 2" [清除])
│ ▾ Flink指南        [全书✓] 第1章☐ 第3章☑ 第4章☐ │ ← 书=全书快捷勾选; 章独立勾选
│ ▾ Kafka精讲        [全书✓] 第2章☐ 第5章☐        │ ← 全部书常驻可选(折叠+搜索)
│ ┌────────────────────────────────────────────┐ │
│ │ 有界流和无界流的定义                   [提问] │ │
│ └────────────────────────────────────────────┘ │
│ ┌─ 答案 ──────────────────────────────────────┐ │
│ │ 无界流是持续生成的数据流… [Flink概念:p3]      │ │ ← 仅 generate_answer=true
│ ├─ 证据 (N) [全部|文本|图片] ← 显示层过滤 ──────┤ │
│ │ [text] 第一章Flink概述►运行模型   score .83 ☑│ │ ← ☑ 勾选入集合
│ │ [image][缩略图] 图1-2 数据流  p3   score .81 │ │ ← /api/documents/page-image
│ └────────────────────────────────────────────────│ │ 勾选后底部浮条: [加入集合 ▼]
└────────────────────────────────────────────────┘
```

### 5.3 结果渲染(复用 Card/Badge/Button + `jsonFetch`)

- 答案卡(generate_answer=true 时)。
- 证据渲染原则: evidence 为按融合分数降序的混排数组(后端排序, 前端不重排), 渲染**单一列表**, 每张卡按 `kind` 分支; 文本块与图片块是独立向量, 一条 evidence 只有一种形态。
- 文本卡(复用现有 narrative card 模式): title Badge + filename Badge + score Badge + `text_preview` line-clamp-3 + 超出时"展开/收起"。
- 图片卡: 左侧固定缩略图框(w-32 h-24, object-contain, bg-muted), `<img loading="lazy" src={image_url}>`(同源代理, 无 CORS); `onError` 显示"图片加载失败[重试]"占位; 右侧 `[图]`/页码 Badge + title + VLM 描述(line-clamp-2)+ score; 点击缩略图新窗口打开原图(v1 不做灯箱)。
- 过滤 chips: `[全部|文本|图片]` 带计数, 纯前端过滤, 默认"全部"。
- `collection=text` 时 evidence 全为文本卡, 同一套组件无图片分支。

### 5.4 新代理路由(同 `route.ts` 模式, `BACKEND_API_BASE_URL`)

`/api/documents/ask`、`/api/documents/collections`、`/api/documents/page-image`(流式透传字节, 透传 content-type; 图片文件名 md5/页号不可变, 响应加 `Cache-Control: max-age` 便于浏览器复用缓存)。

### 5.5 第三部分验收标准(浏览器实测)

1. Header 两页互切; `/documents` 选 multimodal 提问 → 答案卡+图文证据卡渲染, 缩略图可见。
2. 切 `collection=text` → 文本证据卡; "仅检索"模式无答案卡。
3. 过滤 chips 计数正确(全部/文本/图片)且切换过滤生效。
4. 图片加载失败(onError)显示占位与重试; 后端停机 → 友好错误提示。

## 6. 第四部分: RAGAS 评测体系重构(高优先级 Future Work)

现状与动机(2026-09-09 审计): 仅接入 2 个免参考指标(`Faithfulness`/`LLMContextPrecisionWithoutReference`); 数据集 `apple_narrative_questions_100.json` 无参考答案; `rag_evaluation_jobs` 无 reference 列; **ragas 0.4.3 与新版 langchain-community 不兼容(vertexai 拆包), import 崩溃, 评测当前实际不可用**(阻塞 M4); TestsetGenerator/evaluate() 汇编/自定义指标等能力零使用。

### 6.1 R1 修复与现代化 — **已完成(2026-09-09)**

落地记录: ① ragas 安装为 **PR #2769 修复 commit**(`fc0d071`, 惰性 VertexAI import; 官方 main 尚未合并, 0.4.4 发布后可换 `ragas>=0.4.4`)— requirements 已 pin; ② **judge 模型修复**: `RAGAS_LLM_MODEL=deepseek/deepseek-chat` 走官方 key(未配)而崩 → 改 `openai/deepseek-v4-pro`(走 OPENAI_BASE_URL=ark 代理 + OPENAI_API_KEY, 与 DEFAULT_MODEL 同路); ③ **pyarrow 实测窗口**: 24.0.0 在 Py3.13.14 进程内 access violation(此前误判为 langfuse/输出管道问题, faulthandler 定位), 18.1 缺 `pyarrow.json_`, **pin 21.0.0**; evaluation_pipeline 增加模块级 pyarrow 预热(防止事件循环内惰性导入触发原生崩溃); ④ 指标导入**保持 legacy `ragas.metrics` 路径** — 新 `ragas.metrics.collections` 类要求 InstructorLLM(`llm_factory`), 拒绝 langchain `get_llm`; 迁移与类更名(`LLMContextPrecisionWithoutReference→ContextPrecisionWithoutReference`)合并到 R3 一次完成。
验收证据: `import ragas` 0.4.4.dev9 OK; 生产路径 `_score_job` 真实样本两次稳定出分(faithfulness=1.0, context_precision≈1.0); 38 项单测全过。

### 6.2 R2 数据集增强(LLM + 文本改进/扩充, 解锁全量指标的前提)

| 途径 | 内容 | 产出 |
|---|---|---|
| TestsetGenerator | 用 ragas 测试集生成器从 70 份 EDGAR filings 合成问答对(knowledge graph + scenario/persona), 带参考答案 | 新评测集(目标 ≥100 题/带 reference), 版本化存储并记录生成模型与种子 |
| LLM 补标 | 对现有 100 题用 LLM 从对应 filing 抽取参考答案 + 人工抽检(≥20%) | `apple_narrative_questions_100` 升级为带 `reference` 字段的 v2 |
| 链路改造 | `rag_evaluation_jobs` 加 `reference` 列, 入队时携带; `_score_job` 按"有/无参考"自动路由指标 | 指标覆盖 2 → ~13 |

### 6.3 R3 全量指标逐步接入

- 第一批(免参考, R1 后即可): `ResponseRelevancy`(需 embeddings)+ `AspectCritique`/`SimpleCriteriaScore`(LLM 裁判)。
- 第二批(R2 后, 需参考答案): `LLMContextRecall`/`ContextEntityRecall`/`NoiseSensitivity`/`AnswerCorrectness`/`FactualCorrectness`/`AnswerSimilarity`/`LLMContextPrecisionWithReference`。
- 每指标记录 judge LLM 成本与延迟; 指标结果按 question tags 分组聚合; M4/1B/1C 门禁复用同一报告输出(per-metric 趋势对比)。
- 经典 NLG(BLEU/ROUGE 等)与 Agent 指标: 数据形态不支持, 不接(记录理由)。

### 6.4 R4 专项指标(跟随对应部分落地)

- 多模态 `MultiModalFaithfulness`/`MultiModalRelevance`: 第二部分落地后接入(评估图文混合证据的答案)。
- `SQLSemanticEquivalence`: finance SQL 路线评估(参考 SQL 与生成 SQL 语义比对)。
- ragas 0.4 自定义指标装饰器(discrete/numeric): 领域 rubric, 如引用准确性(答案引用 [文件:p页] 是否真实命中)。


### 6.5 R5 LLM 模型对比(选型依据: 用哪个大模型生成答案)

**目标**: 同一套评测题, 让 N 个候选 LLM 分别生成答案, 用同一 judge 模型打分 → 数据驱动的模型选型, 不靠直觉。

| 项 | 设计 |
|---|---|
| 测试集 | R2 产出的带 grading_notes/reference 的评测集(≥100 题) |
| 候选模型 | 通过 `get_llm()` 的 provider 前缀路由: `openai/deepseek-v4-pro`(当前默认)、`qwen/qwen-plus`、`zhipu/glm-4.7`、`openai/gpt-4o` 等 |
| 执行方式 | 同题同上下文, 仅换生成模型 → 每个(模型×题)一行入 `rag_evaluation_jobs`(metadata 记 `generator_model`) |
| 评测指标 | `AnswerRelevancy`(切题度)+ `FactualCorrectness`(事实正确, 需 reference)+ `InstanceSpecificRubrics`(领域 rubric pass/fail) |
| 报告输出 | **模型排行榜 CSV**: 每行一个模型, 列 = 各指标均值 + P95 延迟 + 每题 LLM 费用(token 用量), 排序按加权综合分 |
| 基础设施 | `scripts/run_llm_benchmark.py --models openai/deepseek-v4-pro,qwen/qwen-plus --testset tools/data/eval_set_v2.json` → `experiments/llm_benchmark_<date>.csv` |

### 6.6 R6 组件级 A/B 评测(选型依据： 用哪个 reranker/embedding/检索策略)

**目标**: 端到端指标告诉你"整体好不好", 组件级 A/B 告诉你"哪个零件在拖后腿"。固定其他组件, 只变一个 → 量化该组件的贡献。

| 对比维度 | 变量 | 固定量 | 评测指标 | 预期产出 |
|---|---|---|---|---|
| **Reranker 选型** | `RERANKER_BACKEND=local/none` + `LOCAL_RERANKER_MODEL=4B/0.6B` | 同一 embedding, 同一融合池 | `ContextPrecision`(检索精度)+ `AnswerRelevancy` | "4B 比 0.6B 好 X%, 比 no-rerank 好 Y%" |
| **Embedding 选型** | `EMBEDDING_PROVIDER=qwen/zhipu/openrouter` | 同一 reranker, 同一 chunking | `ContextRecall`(需 reference)+ `ContextPrecision` | "qwen text-embedding-v3 比 zhipu 召回高 X%" |
| **检索策略** | dense-only vs sparse-only vs hybrid | 同一 embedding + reranker | `ContextRecall` + `ContextPrecision` + 延迟 | "hybrid 比单独 dense 好 X%" |
| **Chunking 策略** | `EMBEDDING_SAFE_CHARS` / 语义分块阈值 | 同一 embedding + reranker | `ContextRecall` + `AnswerCorrectness` | "3000 chars 比 2000 召回高 X%" |
| **Analyzer(BM25)** | `MILVUS_TEXT_ANALYZER=english/jieba` | 同一 dense 路径 | `ContextPrecision`(sparse 路) | "english vs jieba 对英文财报的召回差异" |

执行基础设施: `scripts/run_ab_test.py --dimension reranker --variants local:4B,local:0.6B,none --testset tools/data/eval_set_v2.json` → `experiments/ab_reranker_<date>.csv`(per-variant per-metric per-question)。

### 6.7 R7 持续评测与回归检测(评测驱动开发)

**目标**: 评测不是一次性报告, 而是像单测一样每次管线变更后自动跑 → 质量回归早发现。

| 能力 | 设计 |
|---|---|
| 触发时机 | ① 手动 `run_eval_experiment.py` ② CI hook(管线代码变更后自动) ③ 定时(每日跑最新评测集) |
| 回归检测 | 每次评测结果与上一次 baseline 对比; 任一指标降幅 > 阈值(默认 3%)→ 标记 regression 并输出差异明细 |
| 历史趋势 | 评测结果持久化(`experiments/` 目录按时间戳命名), 支持画趋势图(时间×指标) |
| 切片分析 | 按 question tags 分组(mda/risk_factors/sql vs narrative), 检测"整体不降但某一类下降"的隐性回归 |
| 报告格式 | 统一 CSV: `experiment_id, timestamp, pipeline_hash, model, reranker, embedding, metric, value, delta_vs_baseline, regression_flag` |

### 6.8 执行顺序与优先级(更新)

```
R1(已完成) → M4 评测(1A+1B 一并验收) → R2 → R3 第一/二批
→ R5 LLM 选型(与 R3 并行, 只需评测集) → R6 组件选型(R3 后, 需全量指标)
→ R7 持续评测(R5/R6 后, 自动化) → R4 多模态(第二部分后)
```

验收基线: R5 = ≥2 个模型排行榜报告 + 推荐结论; R6 = ≥1 个维度的 A/B 对比报告; R7 = 回归检测 + 趋势持久化可运行。

## 7. 非目标 (P2)

- 以图搜图(any-to-any 查询)。
- ~~前端上传 PDF 入库 UI~~ **已升级为第五部分核心功能**。
- 多模态 RAGAS 评估(已升级为第四部分 R4, 不再是 P2)。
- 跨 collection 混合检索。
- OpenSearch analyzer 语义完全等价复刻。
- 图片块 VLM 问答(答案生成用文本上下文)。
- 图片卡跳转所在页整页预览(页图已落盘, v1 未设入口)。
- 图片证据卡灯箱大图(v1 为新窗口打开原图)。

## 8. 开放问题(开发前需确认)

| # | 问题 | 影响 |
|---|---|---|
| 1 | 可用的 DOTS·OCR vLLM endpoint(参考代码 `172.22.93.49:6006` 为课程内网, 可达性未知); 或确认先走 fitz 降级 | 第二部分解析质量 |
| 2 | `multimodal-embedding-v1` 实际输出维度(参考 Milvus schema 用 1024, 未证实) | 第二部分 collection 维度; 已设计运行时探测, 实现首日实测锁定 |
| 3 | 多模态 embedding 文本长度上限未证 | 已有截断+标记策略兜底 |


## 8.5 第五部分: 通用化改造(用户提交 PDF → 入库 → 问答 → 评测)

### 8.5.1 目标

用户不需要写代码/CLI, 通过**前端界面**完成: 上传 PDF → 自动切分入库 → 按文档提问 → 查看质量报告。

### 8.5.2 前端 PDF 上传入库(从 P2 升级为核心功能)

| 端点 | 契约 |
|---|---|
| `POST /agent/api/documents/upload` | multipart/form-data: `{file: <pdf>}` → 后端调第二部分入库管线(fitz 或 DOTS·OCR)→ 返回 `{document_id, status, node_count, page_count}` |
| `GET /agent/api/documents/documents` | 列出所有已入库文档(collection 过滤, 含状态/页数/块数) |
| `DELETE /agent/api/documents/documents/{id}` | 删除文档(级联 Milvus + PG + 磁盘页图) |

前端 `/documents` 页增加:
- "上传 PDF" 按钮(drag & drop / file picker), 显示上传进度+入库状态
- 已入库文档列表(卡片, 含标题/页数/状态/删除按钮)
- 上传完成后可立即提问(该文档已在 collection 中可检索)

### 8.5.3 动态 collection 管理

当前设计只有 2 个固定 collection(`rag_nodes` + `rag_multimodal`)。通用化后需要:

| 能力 | 设计 |
|---|---|
| 按域分 collection | 用户可选 collection(如 `finance` / `books` / `manuals`), 每个 collection 独立 schema+embedding 模型 |
| API | `POST /agent/api/documents/collections` `{name, embedding_provider, description}` → 创建 |
| 切换 | `/documents` 页 collection 下拉从硬编码改为动态获取(现有 `GET /collections` 改为查库) |
| 隔离 | finance 管线(`rag_nodes`)不受影响; 新 collection 走 `/agent/api/documents/*` 通用路径 |

### 8.5.4 从 PDF 自动生成评测集(打通 R2 与第五部分)

```
用户上传 PDF → 入库完成
  → 点击"生成评测题"(或自动触发)
  → 后端调 RAGAS TestsetGenerator:
      从该 PDF 的 Milvus 节点构建 KnowledgeGraph
      → default_transforms(LLM + embeddings)
      → 按 query_distribution 合成 N 题(带 grading_notes + reference)
  → 输出 eval_set_<collection>_<date>.json
  → 用户可立即"运行评测"(调 R5/R6/R7 的评测管线)
  → 输出质量报告(指标 CSV)
```

| 端点 | 契约 |
|---|---|
| `POST /agent/api/documents/generate-testset` | `{collection, testset_size: 10-100}` → 异步生成, 返回 job_id |
| `GET /agent/api/documents/testset/{job_id}` | 轮询状态 → 完成后返回题目列表(JSON) |
| `POST /agent/api/documents/evaluate` | `{collection, testset_path}` → 异步执行评测(R3 指标) → 返回报告 |

### 8.5.5 与其他部分的关系

| 部分 | 关系 |
|---|---|
| 第二部分(PDF 入库管线) | 第五部分的入库引擎 = 第二部分的管线, 增加通用化参数(不自 SEC 元数据) |
| 第三部分(前端 /documents) | 第五部分的 UI 基座 = 第三部分的页面, 增加上传/文档管理/评测面板 |
| 第四部分 R2(TestsetGenerator) | 第五部分的评测题生成 = R2 的 TestsetGenerator, 增加按 collection 触发 |
| 第四部分 R3(全量指标) | 第五部分的质量报告 = R3 的指标体系 |
| 第四部分 R5/R6/R7 | 通用化后选型能力适用于任何文档类型(不只 SEC) |

### 8.5.6 执行顺序

```
第二部分(入库管线) → 第三部分(前端基座) → 第五部分(上传+动态collection+自动评测)
                                        ↑
                          第四部分 R1-R3(评测能力就绪)
```

验收基线: 用户在前端上传一份课件 PDF → 30 秒内入库 → 立刻提问得到答案 → 点击"生成评测题"→ 10 分钟内获得质量报告(≥5 个指标)。
## 9. 决策记录

| 日期 | 决策 | 备选与否决理由 |
|---|---|---|
| 2026-09-09 | OCR 采用"配置化 vLLM + fitz 降级" | 否决"仅 fitz"(丢失多模态意义)、"仅 vLLM"(无 GPU 时不可入库) |
| 2026-09-09 | 向量化粒度完全复刻参考项目(标题+语义分块+图片单独向量化) | 否决"页级"(实现简单但召回粒度粗) |
| 2026-09-09 | 查询双模式: `generate_answer` 开关 | — |
| 2026-09-09 | 前端新增独立页面 `/documents` + 共享 Header | 否决"同页模式切换" |
| 2026-09-09 | Postgres `rag_nodes` 表结构零变更, 不删 text/title/metadata/search_vector | 否决 v1.1 "PG 瘦身"(用户决策; PG 保持事实源, 迁移风险大幅下降) |
| 2026-09-09 | M3 采用配置开关灰度切换(非一次性切换) | 保留回滚窗口至 M5 |
| 2026-09-09 | 采用扩展协议 `SupportsHybridSearch` + `FUSION_BACKEND` 开关, 随本期实现但默认关闭, M6 门禁后启用 | 否决"删除 factory 直连 Milvus"(失去灰度回滚/写路径统一/单变量评测); 否决"融合一次到位"(迁移评测无法单变量归因) |
| 2026-09-09 | 多模态 collection 存 Milvus(原 v1.0 为 Qdrant, 因第一部分迁移而改) | 与参考项目一致, 免二次迁移 |
| 2026-09-09 | evidence 单一混排列表(按融合分数降序), 卡片按 kind 分支 + 过滤 chips | 否决"文本/图片分区显示"(丢失相关性排序) |
| 2026-09-09 | 生成答案用纯文本 context: 图片 = 入库时 VLM 预生成描述, LLM 不看原图; 原图仅前端证据展示 | VLM 看图生成为 P2(成本/复杂度高) |
| 2026-09-09 | `/documents` 页面与 `/agent/api/documents` router 同现有首页/ask 完全隔离(互不 import) | 落实"互不干扰"硬约束, 见 §5.1.1 |
| 2026-09-09 | 第一部分分步: 1A 仅稠密 Qdrant→Milvus, 稀疏留 PG; schema 一次建全(text+BM25 随 1A 写入) | 单变量评测更干净; 避免 Milvus schema 不可变导致二次迁移 |
| 2026-09-09 | compose 加入 Milvus v2.6.22 三容器(版本取官方 standalone compose), qdrant/opensearch 保持默认启动直至各自切换验收 | 切换前旧后端仍现网在用 |
| 2026-09-09 | 读 PDF 全链路不引入 LangChain: 解析=纯 dots_ocr, 编排=普通函数(不借鉴 LangGraph), LLM/VLM=现有 OpenAI 兼容客户端, 分块=自实现(标题 60 行+语义断点 100 行+层级传播 40 行) | 参考 LangChain 仅为课程技术栈; 引入将拖 langchain-core 依赖树违背单一惯例; `langchain-text-splitters` 仅作对齐失败兜底 |
| 2026-09-09 | M2 改为"源数据重入库"重建 Milvus(同 document_id 原地 replace), 废弃 Qdrant 向量迁移脚本 | 用户决策; 免迁移工具, 稀疏索引顺带重建; 代价 = 重算 embedding 费用 + 依赖源数据可用 + 节点 UUID 换新 |
| 2026-09-09 | 新增第四部分 RAGAS 重构(高优先级): R1 修 ragas 0.4.3 兼容(M4 阻塞) → R2 TestsetGenerator/LLM 补标扩充数据集 → R3 指标 2→~13 → R4 多模态/SQL 专项 | 审计发现评测实际不可用且仅用 2/24 指标; 用户决策列为高优先级 |
| 2026-09-09 | R1 用 git 安装 ragas PR#2769 修复 commit(用户选定安装方式); pyarrow pin 21.0.0(24 原生崩溃/18 缺 API); judge 模型改走 ark 路由; metrics 维持 legacy 路径, collections+InstructorLLM 迁移并入 R3 | 官方 main 未合并修复; 双最新组合不可用已实锤 |
| 2026-09-09 | 库存清空后重建改用 `--document-id-start 9801`(原 999001 为历史遗留): 70 文档恰好 = `document_groups.json` F 组 9801–9870, 前端分组下拉恢复可用, 与 AGENTS.md 标准用法对齐 | 旧 999xxx 批次与分组文件(9002–9870)完全脱节, 分组形同虚设; 用户确认换 9801 |
| 2026-09-09 | Reranker 默认改本地 sentence-transformers CrossEncoder(复刻 qwen_reranker_demo), Bocha 保留为级联回退, `RERANKER_BACKEND` 三态开关 | 用户决策; 免 API 费/降延迟; 接口零改动靠组合器; CUDA 可用, 4/8bit 量化留配置 |
| 2026-09-09 | 新增 R5(LLM 模型选型)/R6(组件级 A/B)/R7(持续评测) — 评测体系从"验证质量"升级为"驱动选型+回归防护" | 用户需求: 用评测数据选 LLM/reranker/embedding; 补充: CI 集成/趋势/切片分析(@experiment 模式启发) |
| 2026-09-09 | 项目愿景升级为通用文档智能问答平台: 用户提交任意 PDF → 入库 → 问答 → RAGAS 自动评测; 新增第五部分(前端上传/动态 collection/TestsetGenerator 集成) | 用户决策: 项目通用化, 不限于 SEC; 前端上传从 P2 升级为核心; 评测能力(R5-R7)因此适用于任何文档 |
| 2026-09-10 | Step 1B 先于 M4 执行(用户决策); MilvusSparseBackend 的 replace 为 no-op, factory 强制 dense=milvus+sparse=milvus 混配 fail-fast; SparseQueryPlan 不生效仅记日志(§3.9 预案) | 用户要求先切稀疏; 行(text+BM25)只能由 dense 路径写入(schema dense 非空), 混配必空结果故显式报错优于静默; 质量差异留 M4 门禁统一验收 |
| 2026-09-11 | BM25 text 加权(写入侧拼接)先于 M4 启用(用户决策, 推翻"门禁触发才做"的默认顺序): title×2+hints×3 拼入 text, 前缀长度记 metadata 读取剥离, `milvus_rebuild_text.py` upsert 重建存量 | 用户判断章节导向查询收益值得提前做; 风险(词法噪声同步放大、reranker 输入污染)以后者剥离方案规避, 净效果由 M4 统一验收 |

| 2026-09-16 | 移除 Bocha 远程 reranker: `bocha_reranker.py` 删除, `rerank.py` 仅 local/none, `BOCHA_TOP_N`→`RERANKER_TOP_N`, `/health` key `bocha_rerank`→`reranker`; leads 的 Bocha web-search 保留 | 用户决策: 本地 4B 已稳定, 远程回退无调用价值 |

## 10. 版本历史

| 版本 | 变更 |
|---|---|
| v1.0 | 初稿: 多模态入库(Qdrant)+全库查询+前端 |
| v1.1 | 新增第一部分 Milvus 迁移(含 PG 瘦身方案); 多模态 collection 改 Milvus |
| v1.2 | 撤销 PG 瘦身: `rag_nodes` 表结构零变更, PG 保持事实源 |
| v1.3 | 协议扩展 `SupportsHybridSearch` + `FUSION_BACKEND` 开关(M6 门禁); M3 定为灰度切换 |
| v1.4 | 接口模型明细(§4.3.1)、查询流程总览与图片角色(§4.2.1)、前端组件树/线框/证据渲染细则(§5.2.1/§5.3)、隔离边界(§5.1.1)、page-image 缓存策略 |
| v1.5 | 第一部分分步: 1A 仅稠密(本期)/1B 稀疏/1C 融合; compose 已实际加入 Milvus v2.6.22(§3.5); M1–M6 阶段重排 |
| v1.6 | M1 落地: milvus_store/dense_milvus/factory/config/pymilvus; 38 单测 + 真机冒烟(建库/过滤/幂等/删除)全绿; 存量文件 lint 问题(config.py/旧 backends 共 54 项)不在范围未动 |
| v1.7 | M2 重定义为源数据重入库(切 env → 同 ID 重跑 ingest → 对账); 新增三条风险(源缺失/UUID 换新/embedding 成本); 原 M3 并入 M2 |
| v1.8 | M2 完成: 70 文档同 ID 重入库 Milvus, 对账 7541=7541 零偏差, E2E 检索(dense+sparse+rerank)通过; 勘误老文档向量少的原因(源文件为封面页) |
| v1.9 | 新增第四部分 RAGAS 评测体系重构(R1-R4); 原 §6-9 顺延为 §7-10; 多模态 RAGAS 从 P2 升级为 R4 |
| v1.10 | R1 完成: ragas@fc0d071 + pyarrow==21.0.0 + RAGAS_LLM_MODEL 走 ark + legacy metrics 路径; 生产 _score_job 出分验证; M4 解锁 |
| v1.11 | Reranker 本地化: local_reranker + rerank 组合工厂(local 默认/bocha 回退/none), 真模型冒烟 PASS, 47 单测全绿 |
| v1.12 | 本地 reranker 切换 Qwen3-Reranker-4B(本地完整缓存, LOCAL_RERANKER_MODEL 支持绝对路径); 4B 冒烟 PASS(56ms 暖推理) |
| v1.13 | 第四部分扩展 R5-R7: LLM 排行榜/reranker+embedding A/B/持续评测; §6.5→§6.8 重排 |
| v1.14 | 新增第五部分"通用化改造"(前端上传/动态 collection/自动评测集); §0 愿景升级; "前端上传 PDF"从 P2 移入核心 |
| v1.15 | Step 1B 落地: `milvus_store.sparse_search`(BM25)+`sparse_milvus.py`+factory milvus 分支(混配校验)+delete 脚本 milvus 分支(修 1A 遗留 dense raise)+env 切 `SPARSE_BACKEND=milvus`; 54 单测全绿(新增 7)+真机冒烟+E2E 对照(检索链路逐位一致); M4 质量门禁待跑; rag_service 稀疏零命中诊断文案补 Milvus 分支 |
| v1.17 | M4 评测执行: 两轮 100 题+评分零失败; faithfulness +2.14% PASS / context_precision -3.95% FAIL(未过 ±2% 门禁); 评测链路修 4 处(thinking:disabled judge/NaN 过滤/分数落库/孤儿 server); m4_benchmark.py 落地 |
| v1.16 | BM25 text 加权落地(§3.9 预案提前执行, 用户决策): `build_enriched_text`(title×2+hints×3, repeats 可配/0 禁用)+前缀标记 `metadata._milvus_text_prefix_chars`+dense/sparse 命中剥离+`scripts/milvus_rebuild_text.py` upsert 重建(7541 行 26s, 对账零偏差, PK 唯一); stats row_count 滞后为显示层现象; 60 单测(新增 6)+冒烟(liquidity top5 全中章节)+E2E 通过; ingest 写路径同步生效 |
| 2026-09-14 | M4 评测落地: m4_benchmark.py 两轮 A/B(断点续跑+评分+报告); faithfulness PASS(+2.14%), context_precision FAIL(-3.95%) → 整体未过门禁, 处置待决策 | 顺带修复: ark 推理模型 thinking 耗尽致 judge 空 content(注入 thinking:disabled), RAGAS NaN 落库, 分数入 jobs.metadata, 孤儿 server 竞态 |
| v1.18 | M5 执行: qdrant 全量移除(代码/依赖/compose/容器/卷); factory dense=milvus 唯一; opensearch 保留待 M4 定论 |
| 2026-09-15 | M4 定论: 接受 context_precision -3.95% 不再对照(faithfulness +2.14% PASS; 差距归因 OS 假阳性+源数据缺失); 执行 M5' opensearch 全量移除 | 用户决策: 误差可接受, 不再投入对照成本; 检索栈收敛 Milvus |
| v1.19 | M4 定论(接受误差) + M5' opensearch 移除; 检索栈最终形态 dense=Milvus only / sparse=milvus 默认+postgres 回退 |
| v1.20 | 第二部分 v2 实施设计定稿(`MULTIMODAL_RAG_PART2_DESIGN.md`): dots.ocr 源码级借鉴映射、三类多模态模型支持矩阵、密集向量后端 env 切换(milvus_multimodal)与零改动边界 |
| v1.21 | 二期设计增量: 图片资产两期存储策略 — 一期 local(MULTIMODAL_PAGES_DIR) / 二期 MinIO, 接口一期定型(`multimodal_asset_store.py`), asset key 两期同构保证 Milvus 数据与 API 契约零迁移 |
| v1.22 | 二期设计补全: 实现蓝图(9 文件级职责+签名, 关键实现决策: bbox 裁剪插图/跨页标题继承/filtered 页/协议兼容)、错误处理矩阵、CLI 契约(退出码/幂等/上限防护)、测试计划、性能预算与可观测性 |
| v1.23 | 模型栈定稿(实测驱动): 多模态 embedding 默认火山方舟 doubao-embedding-vision(plan 端点实测 dim=2048/base64 data URI/图文联合/响应 data 单对象), provider=ark\|dashscope 可切, 新增依赖归零; 图片描述 VLM 可一行切方舟正式端点(doubao-seed-1-6-vision-250815, plan 端点实测 404)或 GLM glm-4.5v; doubao-seedream-5.0-pro 属图像生成模型不适用图片描述, 评估排除 |
| v1.24 | 图片描述 VLM 默认定稿 doubao-seed-2-0-lite-260428(plan 端点实测 vision 直读通过, RPM 30000 匹配批量入库, thinking 须禁用); qwen-vl-plus 降末选(QWEN_API_KEY 实际为空); GLM glm-4v-flash 备选(ZHIPU key 现成); 补充架构说明 — 多模态 embedding 管"找得到", VLM 描述管"讲得出", 不可互替 |
| v1.25 | VLM 备选链实测补全: glm-5.3 仅文本(生成模型看不了图)、glm-5.3-flash 原生多模态可用(thinking 不可禁)、glm-4v-flash 免费档可用(读图内文字准确)、4.5v/4v-plus 需充值; 默认维持 doubao-seed-2-0-lite(可禁 thinking+RPM 30000 批量最优) |
| v1.26 | GLM Coding Plan 端点记录(anthropic/coding-chat/response/标准四端点总表 §4.5); 实测判定 ZHIPU_API_KEY 为 Coding Plan 订阅(glm-5.3-flash 视觉在 coding 端点订阅内可用, 标准端点按量 429) → glm-5.3-flash 双订阅通道; `ZHIPU_BASE_URL` 入 config/.env/env.example |
| v1.27 | 系统设计补全(PART2_DESIGN §19-24): 服务拓扑/collection 全景/数据流、删除级联矩阵与并发防护(先 Milvus 后 PG 后资产, 最终一致)、回滚预案(零改动边界=文本链天然免回滚)、可观测(log_rag/langfuse/m4 门禁)、密钥矩阵与上传安全、需求追踪矩阵 |
| v1.28 | 书籍层级与集合需求(用户): ①Book→Chapter(=PDF)→Chunk 层级, 同 `--book` 多章节 PDF 自动聚合, Milvus 加 book_id/chapter_label 标量 filter 下推, `/documents/filters` 聚合端点; ②前端启动拉 filter 标签 + 动态圈选保存自定义集合(PG 新表 `document_sets`, filter 型/枚举型) + chunk 勾选入集合; ask 扩 filters/set_id 参数(PART2_DESIGN §6.1/6.2/§8) |
| v1.29 | 前端标签筛选器交互定稿(§5.2.0): 挂载拉 /filters 标签面 → 书/章级联 chips 多选(选中书才展开章) → kind 检索前下推(与结果区显示层过滤区分) → 点提问才生效(不自动重查, 变更提示条) → 集合下拉/存为集合/勾选入集合; 线框与组件树更新(FilterBar/SaveSetDialog/勾选浮条); localStorage 持久化+失效清洗; 组件树/线框同步 |
| v1.30 | Bocha 远程 reranker 移除(见 2026-09-16 决策): rerank 仅本地 CrossEncoder, CompositeReranker 级联简化为工厂直选 |
| v1.30 | 修正 filter 语义与交互: 否决"选书才展章"级联(按章筛是独立诉求) → 书/章全量常驻可选(折叠面板+搜索); books/chapters 归一为 document_id **并集**下推(单表达式 `document_id in [...] and kind in [...]`), 消除书章交集陷阱(选书A+选书B的章=并集而非空集); book_id/chapter_label 降级为 evidence 冗余展示字段(零回查 PG) |
| v1.31 | 详细设计定稿: 归一实现位置修正(前端→后端 document_service 单点, 保 filter 集合动态跟随); 蓝图签名对齐(book/chapter 字段/search expr 构造/MmHit 扩展/repository 四方法); 实现决策补 6-8(自然排序算法/chapter_label 生成/MmHit 字段); 错误矩阵+测试计划扩充(归一 422/set 404/超限/重名); §11.1 任务级 WBS(MM-1×5/MM-2×4/MM-3×3, 估时+依赖+验证映射) |
| v1.32 | 命名定稿(用户决策): 文档库服务=document_service 族 — `tools/documents/{document_service,document_api,document_repository}.py`, 路由 `/agent/api/documents`(ask/collections/page-image/filters/sets), 前端 `/documents` 页, `document_sets` 表, `DOCUMENT_ASK_DEFAULT_TOP_K`; 否决 library(软件语境歧义)——代码未写的零成本窗口完成全局替换 |
| v1.33 | 设计评审补全: T2.5 多模态 gold set 门禁(M4 假阳性教训, 10-20 题+R4 指标)、CLI --dry-run 成本预估+token_usage 汇总、MM-4 章节浏览与人工策展、资产 GC(二期同批)、显式不做四项附升级条件(版本/权限/格式/多轮) |
| v1.34 | T2.5 题目生成方法定稿: LLM 起草+人工校准(gen_multimodal_evalset.py, 分层采样→逐 chunk 起草→人工 ~1h→缩编 10-20 题); gold set 五字段含 gold_chunk_ids+scope 硬断言(不经 LLM 的检索正确性判据, 直击 M4 假阳性根因); 与 R2 TestsetGenerator 分工=小集门禁 vs 大集合成 |
| v1.35 | T2.5 题型补全: 跳数=生成时设计(喂 N chunk 即 N 跳) — 单跳(HitRate)/同章聚合(gold=集合, GoldRecall@k+MRR 测排序)/跨书比较(filter+多 gold), 配比 60/30/10; 指标纯计算不经 LLM; 真 multi-hop P2(单轮检索不支持) |
| v1.37 | 评测讨论收口(§17.1): 代码/数据解耦(脚本可与 T1.x 并行开发, 运行硬依赖 T1.5 chunk); 评测执行四步(检索→硬断言→RAGAS 软评分→±阈值门禁, 沿用 M4 模式); 出题防坑四项对照表(指代词/原文复述/假多跳/VLM 描述错) |
| v1.38 | 新增 `MULTIMODAL_DEV_PLAN.md`(施工图): 终态设计一页总览(硬约束/架构/数据模型/检索路径/管线/API/前端/门禁/横切精要) + WBS 任务表(MM-1×5/MM-2×5/MM-3×3/MM-4, 依赖/验证/里程碑映射) + 发布门禁 + 开工前置清单; PART2_DESIGN 挂指针定位为决策档案 |
