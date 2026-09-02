# FinanceRAG：SEC 财报智能检索与问答系统

FinanceRAG 是一个面向 SEC 风格金融披露文件的 RAG（Retrieval-Augmented Generation，检索增强生成）系统。项目将 EDGAR HTML 财报解析为带层级关系的文档节点，结合向量检索、稀疏检索、财务事实 SQL 查询和可选重排序，为用户提供带证据的财报问答能力。

项目同时提供 FastAPI 后端和 Next.js 前端，支持本地文件导入、SEC 数据下载、混合检索、财务指标查询、报告保存以及 RAGAS 评估。

## 项目特点

- **节点化文档结构**：将财报解析为 section tree，叶子节点作为检索 chunk，同时保留章节路径和上下文关系。
- **混合检索**：Qdrant 提供 dense vector search，PostgreSQL 全文检索或 OpenSearch 提供 sparse search，并进行结果融合。
- **金融领域路由**：根据问题判断是否需要 SQL 财务事实、叙述性 RAG，或同时使用两者。
- **财务事实增强**：查询 `sec_financial_observations` 中的 SEC company facts，并可根据 accession、指标等信息缩小 RAG 证据范围。
- **上下文组装**：支持章节后代展开、相邻节点扩展、字符预算控制和标题匹配保障。
- **可选重排序**：支持 Bocha reranker，以及面向叙述类问题的多维度重排序。
- **可观测与评估**：可选接入 Langfuse 追踪，并通过 RAGAS 评估 faithfulness、context precision 等指标。
- **前后端分离**：FastAPI 提供问答和数据接口，Next.js 提供用户交互界面。

## 系统架构

```text
EDGAR HTML / Company Facts JSON
              │
              ▼
     文档解析与 section tree 构建
              │
              ├── PostgreSQL：文档节点、财务事实、评估任务
              ├── Qdrant：dense vectors
              └── OpenSearch / PostgreSQL：sparse index
                              │
用户问题 ──► 金融意图路由 ──► SQL 财务事实查询
                    │              │
                    └──► 混合检索 ──┘
                              │
                    上下文组装与可选 rerank
                              │
                              ▼
                       LLM 生成答案与证据
                              │
                    FastAPI ──► Next.js 前端
```

一次问答的主要流程是：接收问题和文档 ID → 金融意图路由 → SQL 和/或混合检索 → 上下文组装 → 可选 rerank → LLM 生成答案、置信度、来源和证据 → 可选写入 Langfuse 或 RAGAS 评估队列。

## 技术栈

- Python 3.11+、FastAPI、Uvicorn、Pydantic
- LangChain、LlamaIndex、LangGraph
- PostgreSQL、Qdrant、OpenSearch（可选）
- Next.js 15、React 18、Tailwind CSS、Radix UI
- Langfuse、RAGAS（可选）

## 目录结构

```text
.
├── docker-compose.rag.yml       # PostgreSQL、Qdrant、OpenSearch
├── requirements.txt             # Python 依赖
├── src/
│   ├── agent/
│   │   ├── api/                 # FastAPI 服务入口
│   │   ├── core/                # 配置和运行时基础设施
│   │   ├── tools/
│   │   │   ├── asks/            # 问答 API
│   │   │   ├── finance/         # 金融路由、SQL 计划、事实查询
│   │   │   ├── retrieval_backends/
│   │   │   ├── ingestion_service.py
│   │   │   ├── llamaindex_retrieval.py
│   │   │   └── rag_service.py
│   │   └── scripts/             # 导入、问答和评估脚本
│   └── frontend/                # Next.js 前端
└── tests/                       # 单元测试与集成测试
```

## 环境要求

- Docker Desktop
- Python 3.11 或更高版本
- Node.js 18 或更高版本
- 可用的 LLM API Key 和 Embedding API Key
- 如果启用 OpenSearch sparse backend，需要额外运行 OpenSearch

## 快速启动

### 1. 启动基础设施

在项目根目录执行：

```bash
docker compose -f docker-compose.rag.yml up -d
```

默认端口：PostgreSQL `127.0.0.1:5433`，Qdrant `127.0.0.1:6433`，OpenSearch `127.0.0.1:9200`。使用非默认端口是为了避免与本机已有服务冲突。

### 2. 配置后端

```powershell
Copy-Item src/agent/env.example src/agent/.env
```

至少确认以下配置：

```dotenv
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/rag
DB_HOST=127.0.0.1
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=rag
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6433

DEFAULT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=你的密钥
QWEN_API_KEY=你的密钥

DENSE_BACKEND=qdrant
SPARSE_BACKEND=postgres
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

```powershell
cd src/frontend
npm install
```

如需指定后端地址，创建 `src/frontend/.env.local`：

```dotenv
BACKEND_API_BASE_URL=http://127.0.0.1:8000
```

然后启动：

```powershell
npm run dev
```

访问 <http://localhost:3000>。

## 导入 SEC 财报

导入脚本从 `src/agent` 目录执行，并要求 PostgreSQL、Qdrant 和环境变量已配置。

### 导入本地 EDGAR HTML

```powershell
cd src/agent
..\..\venv\Scripts\python scripts/run_sec_finance_pipeline.py ingest-edgar-local `
  --document-id-start 9801 `
  --data-dir tools\data `
  --edgar-glob "EDGAR_320193_*.htm" `
  --companyfacts-json tools\data\CIK0000320193.json
```

导入过程包括 HTML 解析、章节树构建、节点写入 PostgreSQL、向量写入 Qdrant，以及 sparse 索引写入 PostgreSQL 或 OpenSearch。company facts JSON 可补充 accession、表单类型、申报日期和实体名称等元数据。

### 从 SEC 下载并导入

```powershell
cd src/agent
..\..\venv\Scripts\python scripts/run_sec_finance_pipeline.py list-accessions `
  --json-path tools\data\CIK0000320193.json

..\..\venv\Scripts\python scripts/run_sec_finance_pipeline.py ingest-edgar `
  --document-id-start 9100 `
  --max-filings 5 `
  --json-path tools\data\CIK0000320193.json
```

请设置 `SEC_HTTP_USER_AGENT`，并遵守 SEC 的访问频率要求。

### 重新生成向量

修改 Embedding 模型或向量维度后，重新索引相关文档：

```powershell
cd src/agent
..\..\venv\Scripts\python scripts/run_sec_finance_pipeline.py reindex-vectors --document-id 9801
```

`EMBEDDING_DIMENSION` 必须与 Qdrant collection 的向量维度一致。

## 使用 CLI 提问

```powershell
cd src/agent
..\..\venv\Scripts\python scripts/run_sec_finance_pipeline.py ask-multi `
  --document-ids 9801 `
  --question "What does Apple's 2024 10-K say about liquidity and capital resources?" `
  --top-k 8
```

常用命令：

| 命令 | 作用 |
| --- | --- |
| `ingest-edgar-local` | 导入本地 EDGAR HTML |
| `ingest-edgar` | 从 SEC 下载并导入财报 |
| `ingest-direct` | 直接处理文档或 company facts |
| `ask-direct` | 针对单个文档提问 |
| `ask-multi` | 针对多个文档提问 |
| `reindex-vectors` | 重新生成指定文档的向量 |

完整参数说明见 [`run_sec_finance_pipeline.py`](src/agent/scripts/run_sec_finance_pipeline.py)。

## HTTP API

后端 API 前缀为 `/agent`：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/agent/api/ask/generate` | 生成完整问答结果 |
| `POST` | `/agent/api/ask/generate/stream` | 流式生成答案 |
| `POST` | `/agent/api/ask/search-documents-vector` | 文档向量检索 |
| `GET` | `/agent/api/documents/ids` | 获取可用文档 ID |
| `GET` | `/agent/api/documents/catalog` | 获取文档目录 |
| `GET` | `/agent/api/finance/observations` | 查询财务事实 |
| `POST` | `/agent/api/documents/revectorize` | 重新生成文档向量 |
| `GET` | `/agent/api/health` | 查看服务状态 |

问答请求示例：

```bash
curl -X POST http://localhost:8000/agent/api/ask/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the main changes in operating expenses?",
    "document_ids": [9801],
    "top_k": 8,
    "detail_level": "detailed",
    "report_locale": "en"
  }'
```

## RAGAS 评估与 Langfuse

在 `src/agent/.env` 中配置：

```dotenv
RAGAS_ENABLED=true
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=你的公钥
LANGFUSE_SECRET_KEY=你的私钥
LANGFUSE_HOST=http://localhost:3001
```

启动 API 后执行：

```powershell
cd src/agent
..\..\venv\Scripts\python scripts/run_mixed_narrative_questions_parallel.py `
  --questions tools/data/apple_narrative_questions_100.json

..\..\venv\Scripts\python scripts/run_evaluate_pending_parallel.py
```

评估结果会根据配置写入 Langfuse。分数会受到语料、模型、提示词、`top_k` 和上下文预算影响。

## 测试与代码检查

```powershell
uv run ruff check tests
uv run mypy --strict src/agent/tools/finance/report_locale.py
uv run pytest tests/unit_tests
```

集成测试需要已启动的服务和有效 API Key：

```powershell
uv run pytest tests/integration_tests
```

## 重要配置

| 配置项 | 作用 |
| --- | --- |
| `DEFAULT_MODEL` | 默认生成模型 |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | Embedding 服务和模型 |
| `EMBEDDING_DIMENSION` | 向量维度，必须匹配 Qdrant |
| `DENSE_BACKEND` / `SPARSE_BACKEND` | dense / sparse 检索后端 |
| `RETRIEVE_TOP_K` | 默认检索数量 |
| `CONTEXT_CHAR_BUDGET` | 上下文字符预算 |
| `FINANCE_SQL_ROUTING_ENABLED` | 是否启用金融 SQL 路由 |
| `FINANCE_SQL_NARROW_RAG_ENABLED` | 是否使用财务事实缩小 RAG 结果 |
| `BOCHA_RERANKER_URL` | Bocha 重排序服务地址 |
| `LANGFUSE_ENABLED` / `RAGAS_ENABLED` | 追踪和评估开关 |

## 注意事项

- 不要将 `src/agent/.env`、API Key、数据库密码或本地数据提交到 Git。
- 修改 sparse backend 后，需要确认对应的 OpenSearch analyzer、索引名称和索引数据已准备完成。
- 修改 Embedding 模型或维度后，必须重新索引已有文档。
- `document_id` 需要自行规划，避免财报文档和 company facts 文档编号冲突。
- SEC 下载功能应配置有效的 `User-Agent`，并遵守 SEC 服务使用规范。
- 当前项目更适合作为本地开发和简历项目展示，生产环境还需要补充鉴权、限流、密钥管理、数据备份和部署配置。

## License

MIT
