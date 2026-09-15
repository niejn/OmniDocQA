# Data Flush / Database Write Commands

本文档用于手动测试 EDGAR 文档写入 PostgreSQL、Qdrant 和 sparse index。
以下命令以 Windows PowerShell 为例。

## 1. 启动基础设施

在项目根目录 `D:\mashibing\RAGAS-FINANCE` 执行：

```powershell
docker compose -f docker-compose.rag.yml up -d
docker compose -f docker-compose.rag.yml ps
```

默认端口：

```text
PostgreSQL: 127.0.0.1:5433
Qdrant:     127.0.0.1:6433
OpenSearch: 127.0.0.1:9200
```

停止服务但保留数据卷：

```powershell
docker compose -f docker-compose.rag.yml down
```

## 2. 配置后端 `.env`

编辑 `src/agent/.env`，确认数据库和 Qdrant 配置：

```dotenv
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/rag
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6433
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_nodes

EMBEDDING_PROVIDER=openrouter
EMBEDDING_DIMENSION=2048
OPENROUTER_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free
OPENROUTER_EMBEDDING_MAX_INPUT_TOKENS=4096
OPENROUTER_EMBEDDING_CHARS_PER_TOKEN=0.7
OPENROUTER_EMBEDDING_SAFE_CHARS=3000
```

本地 Docker Qdrant 默认不需要 API key，因此保持：

```dotenv
QDRANT_API_KEY=
```

## 3. 写入本地 EDGAR HTML

进入后端目录：

```powershell
cd src/agent
```

建议使用新的 `document_id`，避免覆盖已有数据：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ingest-edgar-local `
  --document-id-start 999001 `
  --data-dir tools\data `
  --edgar-glob "EDGAR_320193_*.htm" `
  --companyfacts-json tools\data\CIK0000320193.json
```

重点观察返回结果：

```text
success = true
node_count > 0
vectorized_count > 0
```

写入流程：

```text
EDGAR HTML
  → 解析 section/chunk
  → split_chunk_payloads()
  → PostgreSQL rag_nodes
  → text 生成 embedding
  → Qdrant upsert Point
  → PostgreSQL full-text 或 OpenSearch sparse index
```

## 4. 验证 PostgreSQL

以下命令从 `src/agent` 目录执行，因此 Compose 文件路径是 `..\..\docker-compose.rag.yml`。

查看节点类型和层级：

```powershell
docker compose -f ..\..\docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT document_id, node_type, level, COUNT(*) FROM rag_nodes WHERE document_id = 999001 GROUP BY document_id, node_type, level ORDER BY level DESC;"
```

查看 level=0 chunk 和父 section：

```powershell
docker compose -f ..\..\docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT c.id AS chunk_id, c.level, c.parent_id, p.title AS parent_title, length(c.text) AS text_chars FROM rag_nodes c LEFT JOIN rag_nodes p ON p.id = c.parent_id WHERE c.document_id = 999001 AND c.level = 0 LIMIT 20;"
```

检查 PostgreSQL 中的向量标记：

```powershell
docker compose -f ..\..\docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE has_vector) AS vectorized FROM rag_nodes WHERE document_id = 999001;"
```

检查不同节点的最大文本长度：

```powershell
docker compose -f ..\..\docker-compose.rag.yml exec postgres `
  psql -U postgres -d rag -c `
  "SELECT node_type, level, MAX(length(text)) AS max_chars FROM rag_nodes WHERE document_id = 999001 GROUP BY node_type, level;"
```

## 5. 验证 Qdrant

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
            @{ key = "document_id"; match = @{ value = 999001 } }
        )
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:6433/collections/rag_nodes/points/scroll" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

只查看 `level=0` 的 Point：

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

预期 payload 包含：

```json
{
  "node_id": "...",
  "document_id": 999001,
  "parent_id": "...",
  "level": 0,
  "node_type": "chunk"
}
```

## 6. 重新生成向量

如果 PostgreSQL 中已有节点，但需要重新生成 Qdrant 向量：

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py reindex-vectors `
  --document-id 999001
```

`reindex-vectors` 会复用 PostgreSQL 中原有的 `node_id` 和 `parent_id`，只重新生成 embedding。

## 7. 查询测试

```powershell
venv\Scripts\python.exe scripts\run_sec_finance_pipeline.py ask-multi `
  --document-ids 999001 `
  --question "What does the filing say about liquidity and capital resources?" `
  --top-k 8
```

## 8. 结果判断

正常情况：

```text
node_count > 0
vectorized_count > 0
success = true
PostgreSQL rag_nodes 有记录
Qdrant document_id=999001 有 Point
```

如果出现：

```text
node_count > 0
vectorized_count = 0
```

说明节点已写入 PostgreSQL，但 embedding 失败。检查：

- `OPENROUTER_API_KEY` 是否配置
- `EMBEDDING_DIMENSION` 是否为 2048
- OpenRouter model 是否可用
- `OPENROUTER_EMBEDDING_SAFE_CHARS` 和 `OPENROUTER_EMBEDDING_CHARS_PER_TOKEN`
- `src/agent/logs/agent.log` 中的 `[Vectorizer]` 和 `[ChunkSplit]` 日志

如果看到：

```text
Api key is used with an insecure connection
```

这是本地 HTTP 连接使用 API key 的安全警告，与 Qdrant 版本或写入失败无关。Docker 本地实例可以保持 `QDRANT_API_KEY=` 为空；生产环境应使用 HTTPS/TLS。
