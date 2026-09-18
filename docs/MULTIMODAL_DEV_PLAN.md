# 多模态文档库 · 系统设计与开发计划（施工图）

> 定位：开工执行文档——终态设计精要 + 任务级开发计划。决策依据与完整论证见 `MULTIMODAL_RAG_PART2_DESIGN.md`（36 节详设，下称【详设】），历史沿革见 `MULTIMODAL_MILVUS_MIGRATION.md`。
> 日期：2026-09-16 · 状态：设计收口，待开工 · 前置阅读：本文 + 【详设】§0（硬约束）

## 1. 一页总览

**目标**：PDF 书籍/文档 → 多模态入库（dots.ocr 解析 + VLM 图片描述 + 图文联合向量）→ `rag_multimodal` collection → `/documents` API 检索（书籍/章节 filter + 自定义集合）→ 前端 `/documents` 页（图文证据卡 + 筛选器 + 策展）。

**四条硬约束**（【详设】§0，违反即返工）：

1. 文本主链零改动：`vectorizer.py`/`dense_milvus.py`/`ask_api.py`/`rag_nodes` 一行不改
2. 新建多模态密集向量实现（`DenseBackend` 协议），env 切换 `DENSE_BACKEND=milvus_multimodal`
3. 依赖边界：新增依赖=0（dashscope 仅备选 provider），不引 dots_ocr 包/LangChain 系/torch
4. 模型栈全部实测锁定（【详设】§2 12 条实测记录）

**终态架构**：

```
[前端 Next.js :3000]  / (SEC问答,不动)      /documents (文档库查询)
[FastAPI :8000]       ask_api(不动)          document_api(新router)
                       └ rag_nodes 路(不动)    ├ text 路: rag_nodes 全库 RRF+rerank
                                               └ multimodal 路: rag_multimodal dense+filter
[存储]  PG: rag_documents(+metadata 层级) + document_sets(新表) | Milvus: rag_nodes(1536) + rag_multimodal(2048)
[资产]  一期本地盘 MULTIMODAL_PAGES_DIR → 二期 rag-minio(asset key 两期同构, 零迁移)
[模型]  解析=dots.mocr/vLLM · 向量=ark doubao-embedding-vision(2048) · 描述=ark doubao-seed-2-0-lite
        (备选链全实测: glm-5.3-flash 双订阅通道 / glm-4v-flash 免费 / evolving)
```

## 2. 系统设计精要

### 2.1 数据模型（【详设】§6）

层级：`Documents(全库) → Book(book_id, metadata 分组标签, 不建表) → Chapter(=1 个 PDF=document_id) → Chunk(text|image)`。

- **Milvus `rag_multimodal`**：`id(PK=uuid) / document_id / filename / title / kind / page_no / text(BM25 预留) / category / image_ref(asset key) / book_id / chapter_label(冗余展示字段) / sparse(预留) / dense(2048, COSINE)`
- **PG**：`rag_documents.metadata` 追加 `{book_id, chapter_index, chapter_label}`（零改表）；新表 `document_sets(set_id, name, kind=filter|enumerated, filter_json, chunk_ids≤500)`（用户集合）
- **升级触发**（任一出现才建 `document_books` 表）：书需独立属性 / 重命名高频 / 书过百 / 动态 collection 落地

### 2.2 检索执行路径（【详设】§6.1, v1.30-31 定稿）

```
filter 归一(document_service 单点): books→PG 展开 doc_ids ∪ chapters(doc_ids) → 去重并集
    ↓ 显式传入但展开为空 → 422(防拼写错误静默全库)
Milvus 单表达式: document_id in [并集] and kind in [...]     ← 向量+过滤一次完成, 零 join
枚举集合: id in [chunk_ids](PK 过滤, ≤500)
标量字段 book_id/chapter_label = evidence 卡展示冗余(零回查 PG), 不参与 filter 表达式
```

### 2.3 入库管线（【详设】§7 六步 + §14 蓝图）

```
CLI ingest_multimodal_pdf.py --data-dir --glob --document-id-start [--book "书名"] [--chapter-start N]
  ①解析   dots_ocr_client: vLLM 逐页(dpi=200, 16线程, OpenAI兼容+imgpad前缀) → layout JSON→md;
          不可达→fitz 降级(纯文本+页图); 单页失败→filtered 不中断
  ②分块   标题边界(H1-3)→插图占位抽成 image 块→>1000字语义分块→标题层级继承; 文件名自然排序定章号
  ③描述   分层: 基础层(caption+上下文, 零成本, 永远有) + 增强层(VLM 读图内内容≤300字,
          DESCRIBE_MODE=vlm|context 可切; VLM 失败→降基础层)
  ④向量   ark /embeddings/multimodal(content-block 数组, data 单对象兼容, RPM 120+429 退避)
  ⑤存储   先删后插(幂等) + rag_documents 登记 + 资产落盘; 维度校验 fail-fast
  ⑥可观测 log_rag 六 stage + 失败清单 + token_usage; --dry-run 成本预估; >500页/200MB 拒绝
```

### 2.4 API 面（【详设】§8, 前缀 /agent/api/documents）

| 端点 | 要点 |
|---|---|
| `POST /ask` | `{question, collection: text\|multimodal, top_k, generate_answer, filters?{books,chapters,kinds}\|set_id?}` 二选一; 422/404/502 契约 |
| `GET /collections` | 两库状态, available 置灰逻辑 |
| `GET /page-image` | 资产字节流, 防穿越内聚 asset store |
| `GET /filters` | 书→章层级标签面(metadata 聚合), 页面初始化数据源 |
| `GET/POST/DELETE /sets` | 集合 CRUD(重名 422/超限 422/失效计数) |
| （MM-4）`GET /chapters/{id}/chunks` | 分页浏览 + 人工策展 |

### 2.5 前端 `/documents` 页（迁移文档 §5）

`DocumentControls + FilterBar(书/章全量常驻折叠面板+搜索, 独立勾选非级联) + EvidenceCard(勾选入集合) + SaveSetDialog`；filter 变更不自动重查；localStorage 持久化+失效清洗；kind 检索下推与显示层过滤视觉分离。

### 2.6 质量门禁（【详设】§17.1）

```
gold set(10-20题): LLM 反向合成(chunk→考题, 喂 N chunk 即 N 跳) + 人工校准 1h
  题型: 单跳60%(HitRate) / 同章聚合30%(GoldRecall@k+MRR) / 跨书10%(filter)
  字段: question+reference+gold_chunk_ids+scope = 硬断言(不经 LLM, 直击 M4 假阳性根因)
  防坑: 无指代词/不复述答案/聚合全员贡献/n-gram 泄漏自检/图题人工对原图核
评测四步: 检索(generate_answer=false) → 硬断言 → RAGAS 软评分(现有基建) → ±2% 门禁
```

### 2.7 横切（【详设】§19-24）

删除级联=先 Milvus 后 PG 后资产（最终一致）；回滚=零改动边界保证文本链天然免回滚；可观测=log_rag/trace_id/langfuse/m4 门禁/水位；密钥矩阵=7 key 全 env（§23 表）。

## 3. 开发计划

### 3.1 WBS（【详设】§11.1 展开）

**MM-1 解析+入库（4-5 天）**

| # | 任务 | 产出 | 验证 | 依赖 |
|---|---|---|---|---|
| T1.1 | 资产存储 | `tools/multimodal_asset_store.py`（协议+Local+防穿越） | 单测 | — |
| T1.2 | 解析客户端 | `tools/dots_ocr_client.py`（vLLM/layout→md/fitz 降级） | 单测+真 PDF 冒烟（**bbox 首验**） | — |
| T1.3 | 分块+描述 | `multimodal_chunker.py` + `multimodal_vlm.py`（分层描述） | 单测 | T1.2 |
| T1.4 | 向量化+后端 | `multimodal_vectorizer.py`(ark) + `dense_milvus_multimodal.py` | 单测（FakeClient） | T1.1 |
| T1.5 | CLI 编排 | `scripts/ingest_multimodal_pdf.py`（六步+--book 聚合+dry-run） | 真 PDF 端到端：幂等/降级/书聚合 | T1.1-1.4 |

**MM-2 检索 API（3-4 天）**

| # | 任务 | 产出 | 验证 | 依赖 |
|---|---|---|---|---|
| T2.1 | 仓储 | `tools/documents/document_repository.py`（sets 表/聚合/展开） | 单测 | PG 池 |
| T2.2 | 切换守卫 | factory 分支+NoneSparseBackend+校验矩阵+ask 守卫 | 单测+70 存量回归 | T1.4 |
| T2.3 | 服务路由 | `document_service.py`(归一单点) + `document_api.py` 五端点 | 单测+curl 验收 | T2.1/2.2 |
| T2.4 | 删除级联 | delete 脚本多模态分支 | 手工验收 | T2.3 |
| T2.5 | 评测门禁 | `scripts/gen_multimodal_evalset.py`+校准+基线 | 基线分入库 | **T1.5**(运行) |

**MM-3 前端（3 天）**：T3.1 页面基座 → T3.2 FilterBar → T3.3 证据策展（依赖 MM-2）

**MM-4 浏览（1 天）**：章节分页端点 + 内容抽屉（MM-3 后小迭代）

**并行线**：T1.1/T1.4 ∥ T1.2→T1.3；gen 脚本代码 ∥ T1.x（FakeClient 单测，运行等 T1.5）。

### 3.2 里程碑与验收映射

| 里程碑 | 演示物 | 验收条目（【详设】§10） |
|---|---|---|
| T1.5 完成 | curl 演示入库+检索（首个可演示） | 1/3/4 + 7 入库面 |
| MM-2 完成 | 完整 API（含 filter/集合） | 2/5/6/7 检索面/8/9 |
| MM-3 完成 | 前端完整体验 | 5/7/8 UI 面 |

### 3.3 发布门禁（每 MM 收口）

70 存量单测全绿 + `/ask` 冒烟一例（零改动证明）+ 本 MM 验收条目 + ruff +（MM-2 起）gold set 基线无回归。

### 3.4 开工前置清单

| 项 | 状态 | 阻塞 |
|---|---|---|
| vLLM 解析服务位置（本机/远端） | 待 T1.2 冒烟时确认 | 不阻塞 T1.1/T1.3/T1.4（fitz 降级路可先行） |
| 测试 PDF 样例 | ✅ 已定位（下表, 无需新找） | T1.5 前 |
| ark/zhipu key | ✅ 已实测 | — |

### 3.5 参考项目与测试素材（2026-09-16 落实）

**参考项目**（迁移文档 §2 借鉴映射的实物来源，对照实现时查证用）：

- `D:\mashibing\RAG_RAGAS\code` — 课程配套多模态 RAG 工程：`dots_ocr/`(DOTS·OCR 解析器), `Multimodal_RAG/`(分块 `splitters/splitter_md.py`、图片描述 `milvus_db/db_operator.py`、embedding 限流 `utils/embeddings_utils.py`), collection 建法(原 `创建一个Collection.py`)
- dots.ocr 官方仓库（解析协议权威源）：https://github.com/rednote-hilab/dots.ocr —【详设】§1 有源码级借鉴映射表

**测试 PDF 样例**（已核实页数/体积，均在 500 页/200MB 上限内）：

| 样例 | 路径 | 用途 |
|---|---|---|
| **Flink 第一章（14 页/1MB）** | `D:\mashibing\RAG_RAGAS\code\第一章 Apache Flink 概述.pdf` | **真实"章节 PDF"样例**（参考项目自带，分章节书需求的现实来源）；概念密度高，单章入库=1 本书 1 章；与拆分章节合成《Apache Flink》书测多章聚合 |
| 主课件（84 页/7MB） | `D:\mashibing\RAG_RAGAS\课件\GraphRAG+多模态RAG+Ragas的项目开发.pdf` | 内容自指（讲多模态 RAG 的多模态文档，图文最丰富）；T2.5 gold set 主出题源 |
| demo_pdf1（2 页） | `D:\mashibing\RAG_RAGAS\code\demo_pdf1.pdf` | dots.ocr 官方 demo 同款；**T1.2 首次真机冒烟 + bbox 坐标首验的最小样本**（秒级反馈） |
| 英文论文（25 页/1MB） | `D:\mashibing\large-model-finetuning-and-deployment-course-courseware\Parameter-EfficientFine-TuningforLargeModels.pdf` | 无 `--book` 入库测**独立文档成书** + 英文语料 + dense 检索跨语言 |

**书籍聚合测试组合**：`--book "Apache Flink"` 收 Flink 第一章 + 主课件拆出的 chapter-02/03（fitz 3 行拆分 → `tools/data/multimodal_test/`）→ 入库后 `/documents/filters` 应返回 1 本书 3 章，其中第 1 章为真实章节样例。T2.5 gold set 主出题源=主课件 chunk（中文/图文/跨章天然齐备），Flink 章节补概念题。
