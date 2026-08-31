# RAGAS-FINANCE 使用命令

本文档记录本项目在 Windows PowerShell 下的常用启动、写入、检索和验证命令。

## 1. 启动基础设施

在仓库根目录执行：

```powershell
docker compose -f docker-compose.rag.yml up -d
```

The Compose files pin Qdrant server to `v1.18.0`, matching the currently
installed Python client `qdrant-client 1.18.x`.

To pull and recreate only the upgraded Qdrant service:

```powershell
docker compose -f docker-compose.rag.yml pull qdrant
docker compose -f docker-compose.rag.yml up -d qdrant
```

If the existing `rag_qdrant_data` volume was created by a much older Qdrant
server, back it up and follow Qdrant's documented consecutive-minor upgrade
path before jumping across several minor versions. For a disposable local
environment, the simplest safe approach is to preserve the PostgreSQL source
data, recreate the Qdrant data volume, and run `reindex-vectors` for each
document. Do not remove a data volume unless its data is backed up or can be
recreated.

Check the running server version:

```powershell
Invoke-RestMethod http://127.0.0.1:6433/
```

默认端口：

| 服务 | 地址 | 用途 |
|---|---|---|
| PostgreSQL | `127.0.0.1:5433` | 完整节点、树关系、全文索引、SEC facts |
| Qdrant | `127.0.0.1:6433` | dense embedding 向量索引 |
| OpenSearch | `127.0.0.1:9200` | 可选 sparse/BM25 索引 |

查看容器状态：

```powershell
docker compose -f docker-compose.rag.yml ps
```

查看日志：

```powershell
docker compose -f docker-compose.rag.yml logs -f postgres
docker compose -f docker-compose.rag.yml logs -f qdrant
docker compose -f docker-compose.rag.yml logs -f opensearch
```

停止服务但保留数据卷：

```powershell
docker compose -f docker-compose.rag.yml down
```

如果需要使用其他端口，在启动时覆盖环境变量：

```powershell
$env:POSTGRES_HOST_PORT = "5432"
$env:QDRANT_HTTP_PORT = "6333"
$env:OPENSEARCH_HOST_PORT = "19200"
docker compose -f docker-compose.rag.yml up -d
```

## 2. 配置后端环境

复制配置模板并编辑 `src/agent/.env`：

```powershell
Copy-Item src/agent/env.example src/agent/.env
```

至少确认以下配置与 Docker 端口一致：

```dotenv
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/rag
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6433
QDRANT_COLLECTION=rag_nodes
EMBEDDING_PROVIDER=auto
EMBEDDING_DIMENSION=1536
```

For OpenRouter, the character fallback is derived from the model token limit:

```dotenv
OPENROUTER_EMBEDDING_MAX_INPUT_TOKENS=4096
OPENROUTER_EMBEDDING_CHARS_PER_TOKEN=0.7
OPENROUTER_EMBEDDING_SAFE_CHARS=3000
```

The effective safe character budget is the smaller of
`max_input_tokens * chars_per_token` and `safe_chars`.

When `QDRANT_API_KEY` is configured while using plain `http://127.0.0.1`, the
Python client may warn that the API key is being sent over an insecure
connection. This is separate from the version mismatch. Use HTTPS/TLS for a
real deployment, or do not configure an API key for an isolated local Qdrant
instance.

如果使用 OpenSearch sparse backend：

```dotenv
SPARSE_BACKEND=opensearch
OPENSEARCH_HOST=127.0.0.1
OPENSEARCH_PORT=9200
```

## 3. 启动后端 API

在仓库根目录执行：

```powershell
cd src/agent
venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/agent/health
```

后端地址：

```text
http://127.0.0.1:8000
```

## 4. 启动前端

另开一个 PowerShell 窗口：

```powershell
cd src/frontend
npm install
```

如需指定后端地址，创建 `src/frontend/.env.local`：

```dotenv
BACKEND_API_BASE_URL=http://127.0.0.1:8000
```

启动开发服务器：

```powershell
npm run dev
```

访问：

```text
http://localhost:3000
```

## 5. 写入文档数据

### 写入本地 EDGAR HTML

从 `src/agent` 执行。建议使用新的 `document_id`，避免覆盖已有文档：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ingest-edgar-local `
  --document-id-start 999001 `
  --data-dir tools\data `
  --edgar-glob "EDGAR_320193_*.htm" `
  --companyfacts-json tools\data\CIK0000320193.json
```

### 从 SEC 下载并写入

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ingest-edgar `
  --document-id-start 999100 `
  --max-filings 5 `
  --json-path tools\data\CIK0000320193.json
```

写入流程为：

```text
解析文档 → 创建 section/chunk NodeRecord
→ PostgreSQL rag_nodes
→ text 生成 embedding
→ Qdrant upsert Point
→ PostgreSQL full-text 或 OpenSearch sparse index
```

`level=0` 是叶子 chunk；`parent_id` 指向直接父 section；`node_id` 同时作为 PostgreSQL `rag_nodes.id` 和 Qdrant Point ID。

## 6. 重新生成向量

当 embedding 模型或维度变化，或需要恢复向量索引时：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py reindex-vectors `
  --document-id 999001
```

`reindex-vectors` 会读取 PostgreSQL 中已有节点，复用原来的 `node_id` 和 `parent_id`，只重新生成 embedding 并写入 Qdrant。

普通重新 ingest 会重新构建节点并生成新的 UUID。

## 7. CLI 查询

单文档查询：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ask-multi `
  --document-ids 999001 `
  --question "What does the filing say about liquidity and capital resources?" `
  --top-k 8
```

多文档查询：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ask-multi `
  --document-ids 999001,999002 `
  --question "Compare revenue and operating results." `
  --top-k 8
```

## 8. 验证 PostgreSQL 写入

查看一篇文档的 section/chunk 数量：

```powershell
docker compose -f docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT document_id, node_type, level, COUNT(*) FROM rag_nodes WHERE document_id = 999001 GROUP BY document_id, node_type, level ORDER BY level DESC;"
```

查看叶子 chunk 与父 section：

```powershell
docker compose -f docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT c.id AS chunk_id, c.level, c.parent_id, p.title AS parent_title FROM rag_nodes c LEFT JOIN rag_nodes p ON p.id = c.parent_id WHERE c.document_id = 999001 AND c.level = 0 LIMIT 10;"
```

查看 embedding 是否成功标记：

```powershell
docker compose -f docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE has_vector) AS vectorized FROM rag_nodes WHERE document_id = 999001;"
```

## 9. 验证 Qdrant 写入

查看 collection：

```powershell
Invoke-RestMethod http://127.0.0.1:6433/collections/rag_nodes
```

按 `document_id` 查看 Point：

```powershell
$body = @{
    limit = 10
    with_payload = $true
    filter = @{
        must = @(
            @{
                key = "document_id"
                match = @{ value = 999001 }
            }
        )
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:6433/collections/rag_nodes/points/scroll" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

只查询 `level=0` chunk：

```powershell
$body = @{
    limit = 10
    with_payload = $true
    filter = @{
        must = @(
            @{ key = "document_id"; match = @{ value = 999001 } },
            @{ key = "level"; match = @{ value = 0 } }
        )
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:6433/collections/rag_nodes/points/scroll" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 10. 常见结果判断

```text
node_count > 0
vectorized_count > 0
success = true
```

通常表示 PostgreSQL 节点和 Qdrant 向量都已写入。

如果出现：

```text
node_count > 0
vectorized_count = 0
```

说明节点已经写入 PostgreSQL，但 embedding 生成失败或未配置 API key。此时检查 embedding 配置和日志，然后运行 `reindex-vectors`。

## 11. 运行测试和检查

```powershell
uv run ruff check tests
uv run pytest tests/unit_tests
```

需要外部服务或 API key 的集成测试：

```powershell
uv run pytest tests/integration_tests
```
