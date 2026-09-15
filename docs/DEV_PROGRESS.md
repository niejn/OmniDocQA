# 开发进度（Milvus 迁移 + 评测体系）

> 详细设计/决策/验收标准见 `docs/MULTIMODAL_MILVUS_MIGRATION.md`（v1.16）。本文件只记状态与证据。
> 更新日期：2026-09-15

## 当前状态总览

| 模块 | 状态 | 说明 |
|---|---|---|
| 稠密检索 | ✅ Milvus only（1A + M5 移除 qdrant，2026-09-15） | dense_qdrant/vector_store/qdrant-client/compose service+卷 全部删除；factory milvus-only |
| 稀疏检索 | ✅ Milvus BM25 + text 加权 + 查询侧 scope（2026-09-15） | `SPARSE_BACKEND=milvus`；text 加权 + 年份/form 硬过滤下推（`QUERY_SCOPE_FILTER_ENABLED`） |
| 融合 | 应用层 RRF（k=60） | 1C/M6 计划下沉 Milvus `hybrid_search`，评测门禁未过不切 |
| Reranker | ✅ 本地 Qwen3-Reranker-4B | `RERANKER_BACKEND=local`，Bocha 级联回退；不下沉 Milvus（见"关键结论"） |
| RAGAS 评测 | ⚠️ R1 已修，指标仅 2 个 | ragas 0.4.4.dev9 可用；R2 数据集增强/R3 全量指标未做 |
| 生成 LLM | ✅ glm-5.3 走 ark 套餐端点 | 模型名带连字符，`glm5.3` 会被 404 |
| M4 质量门禁 | ⚠️ 已跑(2026-09-14), context_precision 未达标 | faithfulness PASS(+2.14% 提升); context_precision FAIL(-3.95%); 详见 09-14 时间线 |
| 第二~五部分（多模态/前端/RAGAS 重构/通用化） | ❌ 未开发 | 需求定稿在迁移文档 §4-§8.5 |

## 已完成时间线

### 2026-09-09
- **Step 1A（M1+M2）**：`milvus_store.py`（schema 一次建全：dense COSINE + text + BM25 Function + INVERTED 过滤索引）+ `dense_milvus.py` + factory + compose 三容器（v2.6.22）+ 70 文档重入库对账（7541=7541）。
- **R1 RAGAS 修复**：ragas pin PR#2769 commit（`fc0d071`）+ pyarrow==21.0.0 + judge 模型走 ark；生产 `_score_job` 真实出分。
- **Reranker 本地化**：`local_reranker.py`（CrossEncoder，BNB 量化可选，懒加载+预热）+ `rerank.py` 组合工厂（local→bocha 升级链）；切本地 4B（RTX 5080 bf16，暖推理 56ms）。

### 2026-09-10
- **Step 1B**：稀疏检索切 Milvus BM25。
  - `milvus_store.sparse_search()`：原始查询文本 → `anns_field="sparse"`，analyzer（english）服务端分词，复用 `build_filter_expr`；命中含 `sparse_score` + 全文 `text`（PG 回表契约不变）；`SparseQueryPlan` 字段加权无法映射单 BM25 字段 → stage log 记 `query_plan_applied=false`（§3.9 预案）。
  - `retrieval_backends/sparse_milvus.py`：async 包装；`replace_document_nodes` no-op（行由 dense 路径统一写入，text 随 insert 自动生成 BM25 向量）。
  - factory：`SPARSE_BACKEND=milvus` 且 dense≠milvus 时 fail-fast（schema dense 列非空，混配必空结果）。
  - `delete_ingested_document.py` 补 milvus 分支；顺带修 1A 遗留：dense 分支原只支持 qdrant，现网默认下必 raise。
  - `.env`/`env.example` 切 `SPARSE_BACKEND=milvus`。
- **生成 LLM 404 修复**：`DEFAULT_MODEL=openai/glm-5.3`（原 `glm5.3` 少连字符，ark 套餐端点精确匹配模型名 → 404 UnsupportedModel）。
- **E2E 全链验证**（doc 9833，"Why did iPhone net sales increase in 2025?"）：
  - 检索：Milvus dense 43 + Milvus BM25 sparse 43 → RRF 40 → 本地 4B 重排 40→12；
  - 生成：glm-5.3，9.7s，3603 tokens，答案精确引用 Q3 $44,582M vs $39,296M（+13%）等可回溯数字；
  - 证据卡：3 quotes / 12 sources。
  - 对照：opensearch 后端跑同题 `retrieve_done` 逐位一致（43/43/12），证明 1B 无检索回归。




### 2026-09-15
- **查询侧 scope 下推落地**（M4 context_precision FAIL 的架构修复）：
  - `finance/query_scope_resolver.py`：纯规则解析问题的年份/form（含 FY 前缀/中文/复数；ASU 编号噪声排除）；sections 仅信息项不硬过滤。
  - `node_repository.resolve_scoped_document_ids()`：PG DISTINCT + `?|`（兼容数组/标量两种 JSON 形态）。
  - `LlamaIndexRetrievalService.retrieve()` 单点下推（构造 retriever 前收窄 document_ids，dense/sparse 同享，内部零改动）；`QUERY_SCOPE_FILTER_ENABLED` 开关；`query_scope` stage 日志。
  - 验证：单测 70 全过（新增 10）；FY2020 题全 70 候选自动收窄到 1 文档，答案正确引用 FY2020 MD&A。
- **两个关键数据考古发现**：
  - FY2012 10-K（doc 9850）源数据仅封面页（2 节点/1318 字符，PG/Milvus/OS 三方一致）——该题任何后端都不可能答对；scope 后"上下文仅有封面"为诚实行为。
  - OS 轮该题 cp=1.0 系**假阳性**：词法加权把其他年份 MD&A 排前→答案基于错误年份内容→judge 只验"context 支撑 answer"不验 scope 正确性→两错相抵得满分。免参考评测盲区实证（R2 gold set 的价值）。
  - M4 -3.95% 差距因此部分失真：含"假阳性 vs 诚实行为"的对比。重评前应剔除数据缺失题或先做 R2。

### 2026-09-14
- **M4 质量门禁执行完毕**（`m4_benchmark.py` 两轮 100 题 + RAGAS 评分, 零失败样本, 报告 `m4_bench/report.md`）：
  - opensearch(97): faithfulness=0.8954, context_precision=0.7271
  - milvus(95):     faithfulness=0.9145, context_precision=0.6984
  - **判定：faithfulness +2.14% PASS（正向提升）；context_precision -3.95% FAIL（超 ±2% 容差）→ 整体未过门禁**
  - 归因：context_precision(免参考)度量检索上下文的头部排序质量——OS 的 title^2.5/hints^4 字段加权对前排相关性贡献显著, Milvus enrichment(title×2+hints×3 TF 近似)只能部分模拟。
  - 待决策：①调 enrichment 次数(2/3→3/4)后重评 milvus 轮 ②接受差距(faithfulness 反升, 答案质量未受损) ③M6 融合下沉时 WeightedRanker 调权补偿。
- **评测链路三项修复**（benchmark 过程中定位）：
  - ark plan 端点推理模型 thinking 耗尽 max_tokens → content 空 → `RagasSanitizingChatOpenAI` 注入 `thinking:disabled`
  - RAGAS NaN 分数落库 jsonb 报错 → `complete_evaluation_job` 过滤 NaN
  - RAGAS 分数改写入 `rag_evaluation_jobs.metadata`（原只进 Langfuse）; enqueue metadata 增 backend/model 标记
- **benchmark 基建**：`m4_benchmark.py`（三层断点续跑/日志/server 生命周期管理/自动报告+门槛判定）；`run_m4_eval.py` 加 checkpoint/resume。

### 2026-09-11
- **BM25 text enrichment（写入侧拼接，用户决策先于 M4 启用）**：单列 BM25 模拟 OS 字段权重（title^2.5/search_hints^4）。
  - 写路径：`text = title×2 + search_hints×3 + 正文`（`MILVUS_TEXT_TITLE_REPEATS=2 / MILVUS_TEXT_HINTS_REPEATS=3`，0=禁用）；前缀字符数记进行内 `metadata._milvus_text_prefix_chars`。ingest 新文档自动生效。
  - 读路径：dense/sparse 命中按前缀剥离——**BM25 吃拼接文本，reranker/下游吃纯正文**（避免 cross-encoder 输入被污染）。
  - 存量重建：`scripts/milvus_rebuild_text.py`（Milvus 读回→剥旧前缀→重拼→upsert；dense 原样带回、sparse 服务端重算、PG 零操作、幂等）。7541 行 26s 完成，PG 对账零偏差、PK 无重复；`get_collection_stats` row_count 显示滞后（upsert=delete+insert 中间态，query 实测 7541 正确）。
  - 冒烟：`liquidity and capital resources` → top5 全中目标章节（17.5 并列，title 加权直接生效）；`iPhone net sales increase` top1-2=MD&A；词法噪声同样被放大（`fiscal` 类查询 "Fiscal Period" 仍 top1）→ **净效果归 M4 评测定夺**。E2E（9833 iPhone 题）retrieve_done 43/43→12、glm-5.3 答案+3 证据卡正常。
  - 单测 60 项（新增 6：拼接重复/禁用/截断保前缀、行映射记前缀、命中剥离、rebuild 往返幂等）。


## 单测与冒烟证据

- 单测 60 项全绿（1B 新增 7 项 + enrichment 新增 6 项）。新改文件 ruff 全绿；仓库存量 lint 告警 12 项未动。
- 真机 BM25 冒烟：68 文档（9801–9870）；enrichment 后章节导向查询显著改善（见 09-11 时间线）。
- 与 OpenSearch 节点零重叠属预期（打分体系不同）——质量结论留给 M4。

## 当前生效配置（src/agent/.env 摘要）

```
DENSE_BACKEND=milvus          # qdrant 仅回滚备份（M5 删）
SPARSE_BACKEND=milvus         # postgres | opensearch 回滚可用
RERANKER_BACKEND=local        # LOCAL_RERANKER_MODEL=本地 Qwen3-Reranker-4B 缓存路径
DEFAULT_MODEL=openai/glm-5.3  # 走 OPENAI_BASE_URL(ark /api/plan/v3) + ark key
MILVUS_TEXT_ANALYZER=english  # 中文语料可切 jieba
MILVUS_TEXT_TITLE_REPEATS=2   # 缺省即 2/3；改值后跑 scripts/milvus_rebuild_text.py
MILVUS_TEXT_HINTS_REPEATS=3
```

## 关键结论：reranker 不下沉 Milvus

Milvus 原生"重排"只有 `hybrid_search` 的 **RRFRanker/WeightedRanker**——排名融合，输入是两路排名序列，不看内容、无语义交互，属于 1C/M6 的"融合下沉"范畴。真正的 reranker 是 cross-encoder（query+doc 联合编码），Milvus 无此内置能力。实测反例：BM25 第一名 "Fiscal Period"（词法强匹配但意图偏移）在 4B 重排后被淘汰——任何 RRF 加权都救不了这个，因为融合输入里没有语义交互信号。两阶段架构（召回快而粗 → 精排慢而准）保持不变。

## 下一步（按迁移文档 §6.8）

1. **M4 处置决策**：context_precision -3.95% 未达标——选调参重评 / 接受 / M6 补偿（见 09-14 时间线）。

3. 1C（可选）：融合下沉 Milvus，先过 RRFRanker 与应用层 RRF 排序一致性单测。
4. R2 数据集增强 → R3 全量 RAGAS 指标 → R5/R6/R7 选型与回归。
5. 第二部分（多模态 PDF 入库）起，进入通用文档平台开发。

## 未提交提醒

1A/1B 全部代码在工作区未提交 git（milvus_store/dense_milvus/sparse_milvus/test_milvus_store 等为 untracked）。M4 验收通过后建议一并提交。
