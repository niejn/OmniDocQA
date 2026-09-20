# OmniDocQA：多模态文档问答与评测平台

> 原名 RAGAS-FINANCE / FinanceRAG，起步于 SEC 财报问答，现已泛化为通用的多模态文档 RAG + 评测闭环平台。

OmniDocQA 是一个评测优先（evaluation-first）的节点式 RAG 系统：把文档解析成带层级的 section tree，在 Milvus 中建立 dense + BM25 混合索引，经 RRF 融合与本地重排序后由 LLM 生成带证据卡的答案；同时提供完整的多模态文档库（任意 PDF → OCR/视觉解析 → 图文分块 → 图文联合向量 → 图文证据检索）与 RAGAS 评测闭环（金标集、硬断言门禁、参考答案指标）。

项目同时提供 FastAPI 后端和 Next.js 前端：`/` 面向 SEC 财报问答（SQL 财务事实 + 叙述性 RAG 双路由），`/documents` 面向文档库（上传 PDF、书/章筛选、集合策展、图文证据、一键生成评测集并出质量报告）。

## 项目特点

- **节点化文档结构**：文档解析为 section tree，叶子节点作为检索 chunk，保留章节路径与上下文关系；查询侧自动解析年份/Form 并收窄检索范围。
- **混合检索**：Milvus 提供 dense 向量（COSINE）与 BM25 稀疏检索，应用层 RRF（k=60）融合；可选 `FUSION_BACKEND=milvus` 下沉服务端 `hybrid_search`（经排序一致性对账，默认关闭）。
- **本地重排序**：Qwen3-Reranker-4B CrossEncoder 本地推理（`local_reranker.py`），面向叙述类问题的多维度重排序。
- **金融领域路由**：规则优先判断问题需要 SQL 财务事实、叙述性 RAG 或两者；查询 `sec_financial_observations` 并以 SQL 证据收窄 RAG 结果。
- **多模态文档库**：PDF → dots.ocr/vLLM 解析（fitz 降级）→ 标题/语义分块 → VLM 图片描述 → ark 多模态 embedding（2048 维）→ `rag_multimodal` collection；图文混合证据卡、书/章聚合、用户集合策展、动态 collection。
- **通用化接入**：前端拖拽上传 PDF（`POST /documents/upload`）→ 自动入库 → 立即检索提问 → 一键生成评测集 → 运行评测出质量报告。
- **评测闭环**：RAGAS faithfulness / context precision / recall / factual correctness，参考答案增强（context_recall 等），多模态指标（MultiModalFaithfulness/Relevance，flag 门控），硬断言（HitRate/GoldRecall@k/MRR/scope 违规率）±2% 回归门禁。
- **可观测**：log_rag 六阶段日志、trace_id、可选 Langfuse 追踪。
- **前后端分离**：FastAPI 提供问答与文档接口，Next.js 15 提供交互界面。

## 系统架构

```text
[文本路：SEC 财报]
EDGAR HTML / Company Facts JSON
        │ 解析 → section tree
        ├── PostgreSQL：rag_nodes（节点、稀疏全文）、财务事实、评测任务
        └── Milvus rag_nodes：1536 维 dense + BM25（title/hints 加权拼接）

[文档库路：任意 PDF]
上传 PDF ──► dots.ocr/vLLM 解析（fitz 降级）──► 标题/语义分块 + VLM 图描述
        ├── ark 多模态 embedding（2048 维）
        ├── Milvus rag_multimodal（动态 collection 可选）
        └── 资产（页图/插图裁剪）→ 本地盘 / MinIO

用户问题 ──► 意图路由 ──► SQL 事实 / 混合检索（RRF）──► 本地重排 ──► 上下文组装
                                                                    │
                                              LLM 生成答案 + 证据卡 ◄┘
                                                                    │
                                          RAGAS 评测队列 + 硬断言门禁 + Langfuse

FastAPI :8000 ──► Next.js :3000（/ SEC 问答 · /documents 文档库）
```

## 技术栈

- Python 3.11+、FastAPI、Uvicorn、Pydantic
- LangChain、LlamaIndex、RAGAS（pinned）、Milvus（pymilvus 2.6+）
- PostgreSQL、Milvus 2.6、MinIO（可选，多模态资产二期）
- Next.js 15、React 18、Tailwind CSS、Radix UI
- 本地模型：Qwen3-Reranker-4B（CrossEncoder）；解析/描述/向量走 vLLM 与 ark OpenAI 兼容端点

## 目录结构

```text
.
├── docker-compose.rag.yml       # PostgreSQL、Milvus、rag-minio
├── requirements.txt             # Python 依赖
├── src/
│   ├── agent/
│   │   ├── api/                 # FastAPI 服务入口（含 documents router 注册）
│   │   ├── core/                # 配置和运行时基础设施
│   │   ├── tools/
│   │   │   ├── asks/            # 问答 API
│   │   │   ├── documents/       # 文档库：repository/service/api（上传/集合/评测集）
│   │   │   ├── finance/         # 金融路由、SQL 计划、事实查询
│   │   │   ├── retrieval_backends/  # dense_milvus / sparse_milvus / dense_milvus_multimodal …
│   │   │   ├── multimodal_*.py  # 资产存储/分块/VLM 描述/向量化/dots.ocr 客户端
│   │   │   ├── multimodal_ingest.py / multimodal_cleanup.py   # 共享入库/级联删除核心
│   │   │   ├── llamaindex_retrieval.py / rag_service.py / node_repository.py
│   │   └── scripts/             # 导入、问答、多模态入库、评测、GC/迁移脚本
│   └── frontend/                # Next.js 前端（/ 与 /documents，同源代理路由）
└── tests/                       # 单元测试与集成测试
```

## 环境要求

- Docker Desktop
- Python 3.11 或更高版本
- Node.js 18 或更高版本
- 可用的 LLM API Key、Embedding API Key（多模态链路用 ark OpenAI 兼容端点）
- 本地重排序需要 GPU（RTX 40/50 系实测）；无 GPU 时设 `RERANKER_BACKEND=none`

## 快速启动

### 1. 启动基础设施

```bash
docker compose -f docker-compose.rag.yml up -d
```

默认端口：PostgreSQL `127.0.0.1:5433`，Milvus `127.0.0.1:19530`，rag-minio `9002/9003`。

### 2. 配置后端

```bash
cp src/agent/env.example src/agent/.env   # Windows: Copy-Item
```

至少确认以下配置：

```dotenv
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/rag
MILVUS_URI=http://127.0.0.1:19530

DEFAULT_MODEL=openai/glm-5.3        # 走 OPENAI_BASE_URL 指向的 OpenAI 兼容端点
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=你的端点
EMBEDDING_PROVIDER=qwen
QWEN_API_KEY=你的密钥

DENSE_BACKEND=milvus
SPARSE_BACKEND=milvus
RERANKER_BACKEND=local              # 无 GPU 改 none
```

完整配置见 [`src/agent/env.example`](src/agent/env.example) 和 [`src/agent/core/config.py`](src/agent/core/config.py)。不要提交 `.env` 或任何 API Key。

### 3. 安装依赖并启动后端

Windows：

```powershell
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m uvicorn api.server:app --app-dir src/agent --host 0.0.0.0 --port 8000
```

Unix / macOS：

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m uvicorn api.server:app --app-dir src/agent --host 0.0.0.0 --port 8000
```

启动后可访问 Swagger：<http://localhost:8000/docs>，服务状态：<http://localhost:8000/agent/health>。

### 4. 启动前端

```bash
cd src/frontend
npm install
npm run dev
```

如需指定后端地址，创建 `src/frontend/.env.local`：

```dotenv
BACKEND_API_BASE_URL=http://127.0.0.1:8000
```

访问 <http://localhost:3000>。

## 导入 SEC 财报（文本路）

导入脚本从 `src/agent` 目录执行：

```powershell
cd src/agent
..\venv\Scripts\python scripts\run_sec_finance_pipeline.py ingest-edgar-local `
  --document-id-start 9801 `
  --data-dir tools\data `
  --edgar-glob "EDGAR_320193_*.htm" `
  --companyfacts-json tools\data\CIK0000320193.json
```

过程包括 HTML 解析、章节树构建、节点写入 PostgreSQL、dense/BM25 写入 Milvus（BM25 text 按 title/hints 加权拼接）。也可用 `ingest-edgar` 直接从 SEC 下载（需设置 `SEC_HTTP_USER_AGENT`）。

注意：`DENSE_BACKEND=milvus_multimodal` 仅服务文档库；文本入库请保持 `milvus`（入口有守卫）。

## 上传 PDF 到文档库（多模态路）

无需 CLI，直接在前端 `/documents` 页拖拽上传；或调 API：

```bash
curl -X POST http://localhost:8000/agent/api/documents/upload \
  -F "file=@your.pdf" -F "collection=rag_multimodal"
```

返回 `{document_id, status, node_count, page_count, filename}` 后即可在 `/documents` 检索提问。dots.ocr 服务不可达时自动 fitz 降级（纯文本块 + 整页图）。支持书/章聚合（`--book`）、用户集合、动态 collection（`POST /agent/api/documents/collections`）。

## 评测闭环

```bash
# 1. 生成/补标参考答案（checkpoint 断点续跑）
python scripts/generate_reference_answers.py --concurrency 3

# 2. 跑批提问并入队评估
python scripts/run_mixed_narrative_questions_parallel.py --questions tools/data/apple_narrative_questions_100.json
python scripts/run_evaluate_pending_parallel.py

# 3. 文档库金标评测（硬断言 + 可选 RAGAS + ±2% 门禁）
python scripts/run_multimodal_eval.py --baseline tools/data/multimodal_eval/reports/baseline.json
```

有参考任务自动路由到 faithfulness + context precision/recall + factual correctness；`EVAL_EMBEDDINGS_METRICS_ENABLED=true` 追加 embeddings 系指标；`--ragas-multimodal` 追加多模态指标（需含图块语料）。

## HTTP API（前缀 `/agent`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/ask/generate`（`/stream`） | SEC 问答（流式可选） |
| `POST` | `/api/documents/ask` | 文档库问答（text / multimodal / 动态 collection，filters/set_id） |
| `POST` | `/api/documents/upload` | 上传 PDF 入库 |
| `GET`/`DELETE` | `/api/documents/documents[/{id}]` | 文档列表 / 级联删除 |
| `GET`/`POST` | `/api/documents/collections` | 两库状态 + 动态 collection 创建 |
| `GET`/`POST`/`DELETE` | `/api/documents/sets` | 用户集合 CRUD |
| `POST`/`GET` | `/api/documents/generate-testset`、`/testset/{job_id}`、`/evaluate` | 生成评测集 / 查询 / 运行评测 |
| `GET` | `/api/documents/filters`、`/chapters/{id}/chunks`、`/page-image` | 书章筛选面 / 章节分页 / 页图 |
| `GET` | `/api/health` | 服务状态 |

## 测试与代码检查

```bash
.venv/Scripts/python -m pytest tests/unit_tests -q     # 280 项，全离线
python -m ruff check <改动的文件>
pytest tests/integration_tests                          # 需要运行中的服务与 API Key
```

## 重要配置

| 配置项 | 作用 |
| --- | --- |
| `DEFAULT_MODEL` / `OPENAI_BASE_URL` | 生成模型与 OpenAI 兼容端点 |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | 文本 embedding（维度须匹配 Milvus collection） |
| `MULTIMODAL_*` | 多模态链路：dots.ocr 地址、embedding 模型/维度、资产存储 local/minio |
| `DENSE_BACKEND` / `SPARSE_BACKEND` / `FUSION_BACKEND` | `milvus|milvus_multimodal` / `milvus|postgres|none` / `app|milvus` |
| `RERANKER_BACKEND` / `LOCAL_RERANKER_MODEL` / `RERANKER_TOP_N` | 本地重排序开关（local/none）/ 模型 / top n |
| `MILVUS_TEXT_TITLE_REPEATS` / `MILVUS_TEXT_HINTS_REPEATS` | BM25 text 拼接加权（改动后跑 `milvus_rebuild_text.py`） |
| `CONTEXT_CHAR_BUDGET` / `RETRIEVE_TOP_K` | 上下文预算 / 检索数量 |
| `EVAL_EMBEDDINGS_METRICS_ENABLED` / `RAGAS_ENABLED` / `LANGFUSE_ENABLED` | 评测指标与追踪开关 |
| `FINANCE_SQL_ROUTING_ENABLED` / `FINANCE_SQL_NARROW_RAG_ENABLED` | 金融 SQL 路由与证据收窄 |

## 注意事项

- 不要将 `src/agent/.env`、API Key、数据库密码或本地数据提交到 Git。
- 修改 Embedding 模型或维度后，必须重新索引已有文档（`reindex-vectors`）。
- `document_id` 需要自行规划，避免财报文档、company facts 与上传 PDF 编号冲突（上传自动分配）。
- SEC 下载功能应配置有效的 `User-Agent`，并遵守 SEC 服务使用规范。
- 当前项目更适合作为本地开发和简历项目展示，生产环境还需要补充鉴权、限流、密钥管理、数据备份和部署配置。

## License

MIT
