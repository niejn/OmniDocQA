# Zhipu (BigModel) 接入端点与协议

记录 Zhipu / BigModel 平台对外暴露的 API 协议与基础 URL，供本仓库接入参考。
（端点由用户提供，作为接入事实来源。）

## 协议端点

| 协议 | 基础 URL | 兼容形态 | 备注 |
|------|----------|----------|------|
| Anthropic Message 协议 | `https://open.bigmodel.cn/api/anthropic` | Anthropic Messages API（`/v1/messages`） | 用 Anthropic SDK 接入，需 `x-api-key` + `anthropic-version` |
| OpenAI Chat Completion 协议 | `https://open.bigmodel.cn/api/coding/paas/v4` | OpenAI Chat Completions（`/chat/completions`） | `/coding/` 为编码 / CodeGeeX 模型变体入口 |
| OpenAI Response 协议 | `https://open.bigmodel.cn/api/v1` | OpenAI Responses API（`/responses`） | 较新的 Responses 形态 |

通用 OpenAI 兼容入口（chat + embeddings 共用，不含 `/coding`）：`https://open.bigmodel.cn/api/paas/v4`。

## 本仓库现有接入

- **Embedding（embedding-3）**：`src/agent/tools/vectorizer.py` 经 OpenAI 兼容 `AsyncOpenAI` 客户端接入，
  `base_url = "https://open.bigmodel.cn/api/paas/v4"`（通用入口，非 `/coding` 变体）。
  - `/embeddings` 挂在 `/v4` 下；请求体带 `dimensions=EMBEDDING_DIMENSION`。
  - 实证：该地址可达 Zhipu 并返回其自有 `429`（余额不足），说明 key 可认证、路径正确。
- **LLM 生成**：当前 `DEFAULT_MODEL=deepseek/deepseek-chat`，未走 Zhipu 聊天协议。
  若改用 Zhipu 聊天，可把 `base_url` 指向上方 OpenAI Chat Completion 协议地址（或通用 `/api/paas/v4`），
  复用 `ZHIPU_API_KEY`（聊天与嵌入同账号同 key）。
- **Anthropic 协议**：本仓库未使用，记录备查。

## 关键约束

- `EMBEDDING_DIMENSION` 必须 == Qdrant collection 维度（embedding-3 支持 256–2048）；切换维度需重新 ingest / `reindex-vectors`。
- Zhipu 账号需有可用额度或资源包，否则 embedding 与 chat 均返回 `429 余额不足`（错误码 `1113`）。
- `EMBEDDING_PROVIDER=zhipu` 时，向量化走 Zhipu；`auto` 回退链为 qwen > openai > zhipu > deepseek。
