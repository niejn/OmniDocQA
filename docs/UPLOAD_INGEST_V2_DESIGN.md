# 上传/入库解耦 + PDF 查看 + 章节切分入库 · 功能设计（施工图）

> 定位：第五部分通用化的第二轮迭代设计。决策依据：部署期实测（配额失败丢文件、上传请求卡死/刷新丢状态、失败原因不可见）+ 用户需求（上传成功可展示、可重新入库、可打开查看、上传/入库分里程碑、支持按章节切分入库）。
> 日期：2026-09-21 · 状态：设计定稿，待开工 · 关联：`MULTIMODAL_DEV_PLAN.md`（现有管线）、`DEV_PROGRESS.md` 09-21 节（部署现状）

## 0. 一页总览

**核心变更**：把现在"一次 HTTP 请求干完上传+入库"的同步管线，拆成两个独立里程碑：

```
M1 上传（存盘+登记，秒回，零模型依赖）
   → M2 入库（后台任务，可重试，状态与失败原因页面上可见）
   → M3 入库选项（整本 / 按章节拆分——TOC 自动识别，每章独立文档、book_id 归组）
```

**四个用户可见状态**（服务端占位行 state machine，页面刷新不丢）：

```
uploaded(已上传) ──ingest──► ingesting(入库中) ──► completed(已入库)
                                   │
                                   └──► failed(入库失败, metadata.ingest_error 带原因, 可重试)
uploaded/failed ──delete──► 级联清理（现有多模态级联复用）
```

**已确认决策**：章节拆分粒度 = 每章独立 document_id、book_id 归组（书章筛选/章节抽屉/分页浏览原生复用）。

## 1. 数据模型变更（零改表）

全部复用 `rag_documents.metadata` JSONB 与占位行状态机，不建新表：

| 字段 | 写入时机 | 说明 |
|---|---|---|
| `status` | 全程 | `uploaded` / `ingesting` / `completed` / `failed`（旧 CLI 行无此键 = completed，兼容不变） |
| `ingest_error` | 入库失败时 | `str(exc)[:300]`，页面"入库失败"卡片直接展示 |
| `ingest_started_at` / `ingest_finished_at` | 入库任务起止 | 排障 + 前端"入库中"时长显示 |
| `has_source_pdf` | 上传成功即 true | 前端"打开"按钮的显隐依据（旧 CLI 行无此键 → 不显示打开） |
| `chapter_split` | 入库选项 | `"off"` / `"toc"`；拆分结果记录 `split_toc_titles` |
| `source_asset` | 上传成功 | 固定 `"{document_id}/source.pdf"`（约定寻址，同 page_{n}.jpg 模式） |

**存储**：原 PDF 入资产库（asset store 一期本地/二期 MinIO 同构 key），每份 ≤200MB 上限继承上传守卫。磁盘水位提示：MinIO 模式推荐作为线上默认。

## 2. 状态机与迁移规则

| 当前状态 | 事件 | 次态 | 备注 |
|---|---|---|---|
| — | 上传成功 | uploaded | 存 source.pdf + 占位行；守卫拒绝（超尺寸/坏文件）→ 无行 |
| uploaded | 点击入库（或 auto_ingest） | ingesting | 后台任务启动，记 ingest_started_at |
| ingesting | 六步管线成功 | completed | 记页数/块数（现 extra_metadata 路径不变） |
| ingesting | 管线失败 | failed | status=failed + ingest_error；**source.pdf 保留**（重试资产） |
| failed | 重试（同参或改参） | ingesting | `POST /documents/{id}/ingest` 幂等复入 |
| uploaded/failed/completed | 删除 | — | 级联：Milvus→资产（含 source.pdf）→PG（现有 delete_document_everywhere 复用） |
| ingesting | 删除 | **拒绝（409）** | 防与在途任务竞态；任务自身失败会转 failed 后再删 |

与现状的差异：`failed` 行**从不可见变为可见**（列表四状态全展示，filters 面板仍只聚合 completed —— 已修复的可见性规则不动）；上传请求断开（刷新/断网）**不再取消任何东西**（上传里程碑秒回，入库是服务端任务）。

## 3. API 契约（`/agent/api/documents` 前缀）

### 3.1 `POST /upload`（语义变更：只做 M1 上传）

```
multipart: file=<pdf>, collection?(默认 rag_multimodal), auto_ingest?(默认 true)
→ 200 {document_id, status: "uploaded", filename, page_count, size_bytes}
422: 非 PDF / >200MB / >500 页（fitz 开页预检，无模型调用）/ 未知 collection
```

- 守卫全过 → `assets.save(id, "source.pdf")` + 占位行(status=uploaded, has_source_pdf=true) → 立即返回
- `auto_ingest=true` 时紧接着以同参触发 3.2（后台），响应不等待入库

### 3.2 `POST /documents/{id}/ingest`（新增：M2 入库/重试，后台任务）

```
json: {chapter_split?: "off" | "toc"(默认 off), collection?}
→ 202 {job_id, document_id, status: "ingesting"}
404 文档不存在 · 409 当前状态 ingesting · 422 参数非法 / 状态为 completed 且未 --force?
```

- 后台任务（复用 document_service JOBS 注册表 + asyncio.create_task 模式）：从资产库取 source.pdf → 现有六步管线（`ingest_one_pdf(replace=True)`）
- 失败：status=failed + ingest_error；**不删 source.pdf**（重试资产；Milvus/页图由既有补偿回滚清理）
- 入库状态直接反映在文档行上，前端轮询 `GET /documents/documents` 即可（不需要额外 job 轮询端点；job_id 保留用于日志关联）

### 3.3 `GET /documents/documents`（响应增强）

item 增加：`status` / `ingest_error?` / `has_source_pdf: bool` / `collection`。**四状态全返回**（含 failed）；filters 聚合继续只吃 completed。

### 3.4 `GET /documents/documents/{id}/pdf`（新增：M2 打开查看）

```
→ 200 application/pdf（inline，浏览器原生预览；资产库字节透传）
404 无 source.pdf（旧 CLI 行） / 400 非法 id
```

鉴权：走 nginx 站点既有 `_auth_verify` 门禁（新增 location `/agent/` 已就位，天然覆盖）。

### 3.5 请求/删除不变

`DELETE /documents/{id}`：ingesting → 409；其余状态照旧级联（把 source.pdf 一并清掉——asset store 按目录删，天然覆盖）。

## 4. 章节切分入库（M3，粒度=每章独立文档）

### 4.1 触发与识别

- 入库参数 `chapter_split: "toc"`：`fitz.open(source).get_toc()` 取 **level-1 条目**为章节（标题+起始页）
- **可拆判据**：level-1 条目 ≥ 2 且页区间覆盖有意义（首章起始页 ≤ 3 之内、章节页区间无重叠——TOC 解析天然有序）
- 退化（无 TOC / 条目 <2）：**整本入库**，metadata 记 `chapter_split="off"` + `split_note="no usable TOC"`，前端提示"该 PDF 无书签目录，已整本入库"

### 4.2 拆分与登记

- 页区间切物理 PDF：PyMuPDF `insert_pdf` 逐章生成子 PDF（与 09-19 课件手动拆章同法，纯本地零模型），章 N 页区间 = [toc[N].page, toc[N+1].page)
- **扉页/前言残页**（首章起始页之前的页）：非空则并入第一章或单独"前言"章（实现取"并入第一章"，少一个 ghost 文档）
- 每章一个新 document_id（复用 allocate_document_id 的防撞预留），登记规则：
  - `book_id` = 原文件名主干（整本的"书名"）
  - `chapter_index` = TOC 顺序；`chapter_label` = TOC 标题（缺失退化"第N章"）
  - 每章各自走六步管线（dots.ocr/fitz → 分块 → VLM 描述 → 向量化），**单章失败只标该章 failed**，其余章照常——部分成功天然可见
- filters 面板效果：1 本书 N 章，书章筛选/章节抽屉/分页浏览全部原生可用（09-19 PEFT 三章聚合已验证此形态）

### 4.3 成本与 Limits

- 拆分不增加模型调用量（总页数不变）；增加 N 次 ensure_collection/insert 批次（可忽略）
- 每章独立 document_id 消耗 id 段位（allocate 自增避让，无需人工规划）

## 5. 前端 `/documents` 页交互

- **DocumentList 卡片按状态渲染**：
  - `uploaded`：〔入库〕〔打开〕〔删除〕；入库按钮弹小对话框选择切分方式（整本/按章节，默认整本+无 TOC 提示）
  - `ingesting`：琥珀色"入库中…"（现状保留），删除禁用
  - `completed`：〔打开〕〔浏览〕〔删除〕（现状 + 打开）
  - `failed`：红边卡片 + 失败原因（ingest_error）+〔重试〕〔打开〕〔删除〕
- **轮询**：列表存在 ingesting 行时每 5s 刷新 documents，全部离开 ingesting 即停（复用 EvalPanel 的代际 token 防竞态模式）
- **打开 PDF**：新标签 `GET /documents/documents/{id}/pdf`（同源经 nginx 门禁），不引 PDF.js
- **上传面板**：文案改为"上传成功后自动开始入库；可在列表跟踪进度与失败原因"

## 6. 兼容性与边界

- **旧 CLI 行**：无 status/has_source_pdf → 视为 completed、无打开按钮，行为不变
- **09-24 定时复验脚本**：`e2e_after_quota.sh` 的 upload 断言需微调（响应 status 变为 "uploaded"，需轮询至 completed 再断言；或传 `auto_ingest=true` 后轮询列表）——随 M1 实施一并更新脚本
- **取消语义收紧**：上传秒回后不存在"刷新取消上传"；入库中不可删不可取消（v1），失败可重试即覆盖真实诉求
- **安全**：/pdf 端点经 nginx 门禁；asset key 走既有 validate_asset_key 防穿越；响应头 `Content-Disposition: inline` + `X-Content-Type-Options: nosniff`

## 7. 测试与验收

- 单测：状态机迁移矩阵（ uploaded→ingesting→completed/failed→retry ）；TOC 解析/拆页纯函数（有 TOC/无 TOC/单条目/跨级）；list payload 增字段；/pdf 端点契约（200/404/nosniff）；ingest 端点 409/422 矩阵
- 实机验收（配额重置后）：上传 100 页有书签 PDF → 选按章节 → 页面出现 N 章 completed + 书章树正确 → 打开每章 PDF → 按章检索命中 → 删除一本级联全清；上传无书签 PDF → 提示退化整本
- 回归：现有 281 单测全绿 + `/ask` 零改动冒烟

## 8. 实施顺序（每步可独立交付）

1. **M1a** 上传改为纯存盘 + 状态机扩展 + 列表四状态/失败原因 + ingest 后台端点（后端为主）
2. **M1b** 前端状态卡片/重试/轮询（依赖 M1a）
3. **M2** source.pdf 资产 + /pdf 端点 + 打开按钮
4. **M3** TOC 拆章选项（后端拆分纯函数 + 前端切分选择对话框）
