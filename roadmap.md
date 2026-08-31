# Embedding-safe ingestion roadmap

## Goal

让所有进入 embedding 的节点满足 provider/model 的输入限制，同时保证原文不丢失、章节树关系不破坏，并让普通文本、EDGAR parser、companyfacts 和 section 节点遵守同一套规则。

## 已确认的现状

- `CHUNK_SIZE=1500` 只覆盖 `build_retrieval_chunks()` 路径，不是所有 ingest 路径的全局上限。
- `_elements_to_chunks()` 可能直接把较大的 EDGAR parser element 转为 `ChunkPayload`。
- `_store_nodes()` 会给 section、chunk、document summary 全部生成 embedding。
- `_normalize_embedding_text()` 当前按字符截断；这能防止请求失败，但会使被截断部分无法参与该节点的向量召回。
- `node_id`、`parent_id`、`document_id` 和 metadata 不进入 embedding；它们用于持久化、索引和过滤。
- PostgreSQL 保存完整原文和父子关系；Qdrant 保存向量及过滤 payload。

## 目标约束

```text
完整原文必须保留在 PostgreSQL
embedding 输入必须不超过模型 token 上限
切分不得破坏 parent_id、section_path 和文档顺序
embedding 失败不能破坏已有 PostgreSQL 数据
普通 ingest 与 reindex 必须保持 ID 语义一致
```

## #1: 定义 embedding 输入契约

Blocked by: none  
Type: Grilling

### Question

每个 provider/model 的最大 token 数、目标 chunk token 数、overlap token 数和安全余量分别是多少？配置是否允许按 provider/model 覆盖？

### Proposed answer

新增 provider/model profile 配置，例如 `OPENROUTER_EMBEDDING_MAX_INPUT_TOKENS`、`OPENROUTER_EMBEDDING_TARGET_TOKENS`、`OPENROUTER_EMBEDDING_OVERLAP_TOKENS` 和 `OPENROUTER_EMBEDDING_SAFE_CHARS`。配置必须明确单位是 token；字符数只能作为没有 tokenizer 时的保守 fallback。目标值应明显低于最大值，例如最大 4096 token 时，目标 chunk 可先设为 700–1200 token。没有专属 profile 的 provider 才使用 `EMBEDDING_SAFE_CHARS` fallback。

## #2: 选择 tokenizer 与 token budget adapter

Blocked by: #1  
Type: Research

### Question

如何针对远程 embedding model 可靠计算 token 数？

### Proposed answer

定义一个小而稳定的 tokenizer adapter 接口，优先使用与实际 embedding model 匹配的 tokenizer；无法加载时使用保守字符预算并发出 warning。不要把 `SentenceTransformer` 当作 tokenizer。SentenceTransformer 只在后续确实需要语义切分时作为独立的 semantic-splitting adapter。

## #3: 建立统一的 embedding-safe chunking seam

Blocked by: #1, #2  
Type: Prototype

### Question

所有 `ChunkPayload` 和 section/document-summary 文本如何经过同一个检查与拆分模块？

### Acceptance criteria

- 普通文本、EDGAR element、companyfacts 和 section 输入都能通过同一接口处理。
- 输出的每个 embedding unit 都不超过 configured token budget。
- 默认按 heading → paragraph → sentence → hard boundary 的顺序拆分。
- 超长单句或表格提供确定性的 hard split fallback。
- `text` 内容完整保留；不使用静默前缀截断作为主要方案。

## #4: 保留树关系与来源信息

Blocked by: #3  
Type: Grilling

### Question

一个原始 element 被拆成多个子 chunk 后，如何保持可追踪性？

### Proposed answer

所有拆分后的子 chunk 共享原 section 的 `parent_id`，拥有独立 `node_id`，并按 `order_index` 连续排序。metadata 增加 `source_chunk_id`、`chunk_part`、`chunk_total`；如果成本可接受，再记录原文 offset 或 page range。`section_path`、title 和 finance metadata 必须复制到每个子 chunk。

## #5: 确定 section node 的表示策略

Blocked by: #1, #3  
Type: Grilling

### Question

section node 是否继续参与 embedding？超长 section 是拆分、preview，还是 LLM summary？

### Proposed answer

第一阶段不引入 LLM 摘要。优先采用：section title + 受限 preview，或只对 level=0 chunk 做 embedding。若保留 section embedding，section 文本也必须经过 embedding-safe 预算。LLM summary 作为第二阶段可选 adapter，必须有缓存、model/prompt version、来源 node IDs 和失败 fallback。

## #6: 处理 embedding 失败与双库一致性

Blocked by: #3, #4  
Type: Grilling

### Question

部分节点 embedding 失败时，PostgreSQL、Qdrant 和 `has_vector` 如何保持可恢复状态？

### Proposed answer

PostgreSQL 先保存完整节点；成功向量的节点设置 `has_vector=true`，失败节点保留并记录失败原因。不要因部分 embedding 失败清空旧 Qdrant 数据。`reindex-vectors` 必须复用已有 `node_id`/`parent_id`，支持补齐失败节点。日志至少记录 node ID、node type、level、title、text chars、token count、provider 和 model。

## #7: 增加回归测试与数据不变量

Blocked by: #3, #4, #6  
Type: Prototype

### Acceptance criteria

- 长 EDGAR element 会被拆分，不再把 15981 长文本直接送入 embedding。
- 所有 embedding 输入都满足 token 上限。
- 拆分前后原文可重组，允许且仅允许预期 overlap 重复。
- 子 chunk 的 `parent_id`、`section_path` 和顺序正确。
- section/document-summary 超限时走明确策略。
- embedding API 失败时 PostgreSQL 节点仍可查询。
- reindex 后 `node_id`、`parent_id` 不变。
- 测试覆盖普通文本、EDGAR、companyfacts 三条 ingest 路径。

## #8: 观测、迁移与发布

Blocked by: #5, #6, #7  
Type: Grilling

### Acceptance criteria

- ingestion 结果报告原始节点数、最终 embedding unit 数、失败数和最大 token 数。
- debug 信息能区分 leaf chunk、section node、document summary 的超限来源。
- 文档说明切分配置、provider/model 限制和 `reindex-vectors` 使用方式。
- 先对新 document ID 灰度 ingest，再对已有文档执行 reindex。
- embedding model 或 token budget 改变时，明确要求重新生成向量；普通 ingest 生成新 UUID，reindex 复用旧 UUID。

## 推荐实施顺序

```text
#1 配置契约
  → #2 tokenizer adapter
  → #3 统一安全切分
  → #4 关系/来源元数据
  → #6 失败恢复
  → #7 回归测试
  → #5 section 策略优化
  → #8 灰度发布与文档
```

SentenceTransformer 语义切分和 LLM section summary 都放在核心安全修复之后；它们是质量优化，不应成为解决 422 输入超限的前置依赖。
