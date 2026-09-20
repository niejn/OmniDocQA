# 开发进度（Milvus 迁移 + 评测体系 + 多模态文档库）

> 详细设计/决策/验收标准见 `docs/MULTIMODAL_MILVUS_MIGRATION.md`（v1.37）。本文件只记状态与证据。
> 更新日期：2026-09-19（多模态文档库 Part 2 全量交付；总结报告见文末"2026-09-19 开发总结报告"节）

## 当前状态总览

| 模块 | 状态 | 说明 |
|---|---|---|
| 稠密检索 | ✅ Milvus only（1A + M5 移除 qdrant，2026-09-15） | dense_qdrant/vector_store/qdrant-client/compose service+卷 全部删除；factory milvus-only |
| 稀疏检索 | ✅ Milvus BM25 + text 加权 + 查询侧 scope（M5' 后 milvus/postgres/none 三选, 默认 milvus） | opensearch 已于 2026-09-15 全量移除；09-19 顺带修复 M5' 遗留 sparse_query_profiles 断链（见总结报告"注意事项"） |
| 融合 | 应用层 RRF（k=60） | 1C/M6 计划下沉 Milvus `hybrid_search`，评测门禁未过不切 |
| Reranker | ✅ 本地 Qwen3-Reranker-4B | `RERANKER_BACKEND=local`（Bocha 回退已于 2026-09-16 移除，仅 local/none）；不下沉 Milvus（见"关键结论"） |
| RAGAS 评测 | ⚠️ R1 修复 + R2 链路/补标脚本就绪 + R3/R4 flag 已接（09-20） | ragas 0.4.4.dev9 可用；reference 列已迁移，补标 84/100（16 题待 ark 配额 09-23 重置后续跑）；R3 embeddings 系/R4 多模态系默认关待实战 |
| 生成 LLM | ✅ glm-5.3 走 ark 套餐端点 | 模型名带连字符，`glm5.3` 会被 404 |
| M4 质量门禁 | ✅ 定论(2026-09-15, 用户决策): 接受误差 | faithfulness +2.14% PASS; context_precision -3.95% 已归因(OS 假阳性+源数据缺失), 接受不再对照; opensearch 已移除 |
| **多模态文档库（第二部分 + 第三部分前端）** | ✅ **2026-09-19 全量交付** | MM-1/2/3/4 + T2.5 评测门禁全部完成并真机验收，含资产 GC 与二期 MinIO；总结报告见文末 |
| 第五部分（通用化：用户上传 PDF→入库→问答→评测） | ✅ **2026-09-20 v1 交付** | 上传/文档管理/动态 collection v1/testset 生成+评测运行 9 端点 + 前端全套；v1 简化项见 09-20 节待办表 |

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





### 2026-09-15（下午）
- **M4 定论（用户决策）**：接受 context_precision -3.95% 误差（faithfulness +2.14% PASS 已证答案质量无损；差距已归因于 OS 轮假阳性与封面页源数据缺失），不再重评对照。
- **M5' opensearch 全量移除**：删 `sparse_opensearch.py`/`inspect_opensearch.py`/config 12 字段/factory 分支(sparse 默认 milvus)/delete 脚本分支/`opensearch-py` 依赖/compose service+卷+注释/容器+卷。70 单测全绿（混配校验测试随 dense milvus-only 而废弃，改为断言 opensearch removed）。检索栈收敛：**dense=Milvus 唯一, sparse=milvus(默认)|postgres**。

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

## 下一步（2026-09-19 更新）

1. ~~M4 处置决策~~ ✅ 已定论接受误差（09-15）。
2. ~~第二部分多模态 + 第三部分前端~~ ✅ 全量交付（09-19，见文末总结报告）。
3. 1C（可选）：融合下沉 Milvus，先过 RRFRanker 与应用层 RRF 排序一致性单测。
4. R2 数据集增强：**补标脚本与 100 题 reference 生成待排期**（需跑 LLM，成本另计；链路改造已完成）→ R3 全量 RAGAS 指标 → R5/R6/R7 选型与回归。
5. R4 多模态指标（MultiModalFaithfulness/Relevance）：需先以 dots.ocr/vLLM 真机入库含图片块的书（当前库为 fitz 降级、全 text 块）。
6. 第五部分通用化（用户上传 PDF→入库→问答→评测），多模态底座已就绪。

## 2026-09-18/19 多模态文档库（Part 2 全量落地）

**MM-1 + MM-2 + MM-4 后端 + MM-3 前端 + T2.5 评测基建 全部完成**（施工图 docs/MULTIMODAL_DEV_PLAN.md，详设 docs/MULTIMODAL_RAG_PART2_DESIGN.md）：

- **模块**：`multimodal_asset_store.py`(协议+Local+防穿越) / `dots_ocr_client.py`(vLLM layout 协议+layout_to_md 自实现+bbox 裁剪+fitz 降级) / `multimodal_chunker.py`(标题/插图/语义分块+跨页标题继承) / `multimodal_vlm.py`(双层描述) / `multimodal_vectorizer.py`(ark /embeddings/multimodal 直调,data 单对象兼容,RPM+429 退避,dim 探测) / `dense_milvus_multimodal.py`(rag_multimodal 2048 维,jieba BM25 预留,book_id/chapter_label 冗余,单表达式 filter 下推) / `documents/{document_repository,document_service,document_api}.py`(sets 表 CRUD+filters 聚合+归一单点+六端点+MM-4 章节分页)。
- **接缝**：factory `milvus_multimodal` 分支+混配矩阵；`NoneSparseBackend`；rag_service 入口 ask 守卫；delete 脚本多模态级联（Milvus→PG→资产）。
- **CLI**：`ingest_multimodal_pdf.py`（六步+--book 自然排序聚合+dry-run+幂等守卫+ingest_report.json）。
- **E2E（Flink 第一章 14 页 + PEFT 论文）**：fitz 降级入库 20 块+210 块, dim=2048(ark 实测)；幂等重跑点数不变；`有界流和无界流的定义` top1 正中定义块(0.50)；五端点 curl 验收（422 拼写守卫/filters+set_id 互斥/重名 422/删除集合后 404）；书籍聚合 1 书 3 章（peft-ch1/ch2/ch10 自然排序）；删除级联手工验收；text 路与 /ask 冒烟零改动证明。
- **T2.5**：`gen_multimodal_evalset.py`（分层采样+三题型出题，跳数=生成时设计；gold_chunk_ids+scope 硬断言；n-gram 泄漏自检）+ `run_multimodal_eval.py`（检索→硬断言 HitRate/GoldRecall@k/MRR→可选 RAGAS→±2% 门禁）。
- **前端**：`/documents` 页（库切换/FilterBar 书章独立勾选/集合下拉+存集/证据卡勾选/SaveSetDialog/MM-4 章节抽屉/localStorage 失效清洗）+ 7 个同源代理路由。
- **顺带修复 M5' 遗留**：`sparse_query_profiles.py` 引用已删 `OPENSEARCH_SPARSE_SEARCH_SCOPE` config 字段直接 AttributeError——自 09-15 起稀疏查询计划构造全断（存量单测用 FakeClient 覆盖不到），改 getattr 防御默认 finance。该 bug 由 /documents text 路首跑暴露。
- **测试**：新增 7 个单测文件 73 项（asset/ocr client/chunker/vectorizer/multimodal backend/factory 矩阵/repository+service），全套 142 绿；新改文件 ruff 全绿。

## 2026-09-19 待办续作（主计划完成后继续开发）

- **T2.5 基线出分**：12 题校准后 gold set（glm-5.3-flash 出题，泄漏检测升级为最大连续公共段规则——首版 8-gram 计数把专名误报 6/12；3 道真泄漏题人工重写）。基线（top_k=8）：**HitRate=0.917 / GoldRecall=0.708 / MRR=0.861 / scope_violation=0**，报告 `tools/data/multimodal_eval/reports/`。聚合题 GoldRecall 0.5 即排序质量待改进项（正是 MRR 度量点），跨书 filter 零违规。
- **§10-7 书籍聚合验收**：PEFT 论文 fitz 拆 3 章（ch1/ch2/ch10 验证自然排序）→ `--book "PEFT 论文"` 入库 9911-9913 → `/filters` 1 书 3 章；`chapters:[9912]` 过滤命中仅该章。**T2.4 删除级联**先删旧 9902 手工验收（Milvus 点+PG 行+资产目录全清）。
- **页图资产补齐（设计偏差修正）**：§6 要求 `page_{n}.jpg` 始终落盘（证据缩略图+fitz 降级唯一视觉产物），CLI 原实现只存插图裁剪——已补（两种解析后端都存）。
- **资产 GC**：`scripts/multimodal_assets_gc.py`（Milvus image_ref 对账，dry-run 默认/--apply；`ingest_report.json` 白名单；`page_*.jpg` 约定寻址规则——文档仍有点时保留）。真数据 dry-run 0 孤儿（反向印证删除级联无残留）。
- **二期 MinIO 全链落地**：compose 独立 `rag-minio`（宿主 9002/9003，不复用 milvus 内嵌/langfuse 的 minio）+ config MINIO_* 5 字段 + `MinioAssetStore`（key 同构 `{doc_id}/{name}`）+ `multimodal_assets_migrate.py`（local→minio，size 全量比对+5 抽样 md5 字节校验，--reverse 回滚）。真机迁移 14 文件 verified；minio 模式 `open()` 与本地字节全等。`MULTIMODAL_ASSET_STORE` 默认仍 local（切换不强制）。requirements.txt 加 `minio>=7.2`。
- **MinIO 升级 + minio 模式全链路测试（09-19 续）**：① 版本升级——GitHub 最新 release 2025-10-15（含 STS 提权 CVE 修复）**只发源码不发镜像**（release 正文"clone and build"），docker.io 实际最新镜像 tag=**RELEASE.2025-07-23T15-54-02Z**；本机 13 个公共加速器/镜像源对 minio 全部 403/404/502，**华为云 DDN 公共源可正常 plain docker pull**：`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/minio/minio:<tag>`（compose 已注释说明）。② 升级后桶内 14 文件 md5 与本地逐字节全等。③ minio 模式（`MULTIMODAL_ASSET_STORE=minio`）全链路补测全过：demo_pdf1 入库 doc 9921 → **资产只落桶、本地零副本**（2 张页图 2.6MB）；8001 临时实例 `/page-image` 从桶取图 200/1.29MB jpeg + 穿越 400 + 缺失 404 + ask 检索正常；删除级联后桶内 9921/ key 清空、Milvus 点归零、PG 行删除。~~主服务 8000 保持 local 模式~~ → **2026-09-19 已正式切换**：.env 加 `MULTIMODAL_ASSET_STORE=minio` 并重启 API，验证 `/page-image`（9901/page_0.jpg，617KB jpeg）从桶读取 200、multimodal ask 与 /ask 零改动冒烟均 200，`config` 解析确认为 `MinioAssetStore`。此后入库资产只落桶（本地盘不再新增副本）。

- **MinIO 源码自建 2025-10-15 成功并切换（09-20）**：`scripts/build_minio_from_source.sh` 一键自动化——①shallow clone 官方 tag（clone 时禁 autocrlf，Windows git 否则会把 docker-entrypoint.sh 转成 CRLF 导致容器 exec 失败）②在 golang:1.24-alpine 容器内编译 linux 二进制（host 无需 Go；GOPROXY 走 goproxy.cn；ldflags 按 buildscripts/gen-ldflags.go 公式在 bash 复刻注入 Version/ReleaseTag/CommitID）③`FROM minio/minio:latest` 基础镜像离线用本地 07-23 retag 顶替（该层仅 chmod+COPY）④docker build + `--version` 验证。产物：`minio version RELEASE.2025-10-15T17-29-55Z (commit-id=9e49d5e7, go1.24.2 linux/amd64)`——比官方二进制仅编译器 patch 版不同。rag-minio 已切到自建镜像（compose 注释含回滚 tag），healthy，桶内 14 文件 md5 全等，/page-image 200。注意：MSYS Git Bash 调 docker 需 `MSYS_NO_PATHCONV=1` 防路径改写（脚本已内置）。默认 build 不切换；`--switch` 参数一键切。
- **Docker 加速固化（09-19）**：实测 daemon.json `registry-mirrors` **无法使用**华为 DDN 源——它是"仓库名内嵌代理"（`ddn-k8s/docker.io` 属 SWR 仓库空间，URL 拼接在 `/v2/` 之后），而 registry-mirrors 只会拼 `<mirror>/v2/minio/minio`（实测 404/401）。等效方案落地为 `src/agent/scripts/ddn_pull.sh`（显式前缀 pull + retag 回官方名，实测拉回 RELEASE.2025-06-13 验证通过）；现网 minio 镜像已就地升级完成，无需改 daemon 配置、无需重启 Docker。
- **R2 链路改造（§6.2 第三行）**：`rag_evaluation_jobs` 加 `reference` 列（CREATE+ALTER IF NOT EXISTS 迁移已对真库生效）；`enqueue_evaluation_job(reference=)`；`_score_job` 按有/无参考路由——有参考=faithfulness+context_precision_with_reference+context_recall+factual_correctness（全 LLM-only，AnswerCorrectness/AnswerSimilarity 等 embeddings 系留 R3），无参考=原两指标。指标覆盖 2→5，R2 的 LLM 补标脚本与 100 题参考答案生成待排期（需跑 LLM，成本另计）。

## 2026-09-19 开发总结报告：多模态文档库（Part 2 全量）

按 `docs/MULTIMODAL_DEV_PLAN.md` 施工图完成 **MM-1 至 MM-4 全部里程碑 + T2.5 评测门禁**，以 `第一章 Apache Flink 概述.pdf`（参考项目自带）为真实测试输入完成端到端验收；随后按待办清单续作完成**资产 GC、二期 MinIO、R2 评测链路改造**。

### 交付内容

**MM-1 解析+入库**（6 个新模块 + CLI）

- `dots_ocr_client.py`：vLLM layout 协议复刻（`<|img|><|imgpad|><|endofimg|>` 前缀、11 类 category、filtered 降级）、`layout_to_md` 自实现、bbox→页图坐标映射裁剪（首页 log 因子供 §14.1-1 校验）、fitz 文本降级（`## 第 N 页` 前缀 + 整页图）
- `multimodal_chunker.py`：标题边界分块（H1-H3）、插图占位抽取、>1000 字语义分块（切句→相邻余弦→percentile 断点）、跨页标题继承（§14.1-3）
- `multimodal_vlm.py`：双层图片描述（基础层=上下文拼接零成本 / 增强层=VLM 读图内内容 ≤300 字；doubao 系禁 thinking、GLM 系不禁的模型分叉）
- `multimodal_vectorizer.py`：ark `/embeddings/multimodal` httpx 直调（data 单对象兼容、RPM 固定窗口 + 429 指数退避、dim 自动探测）；dashscope 备选 provider
- `dense_milvus_multimodal.py`：`rag_multimodal` collection（2048 维 COSINE、jieba BM25 预留、book_id/chapter_label 冗余展示字段、单表达式 filter 下推 `document_id in [...] and kind in [...]`）
- `scripts/ingest_multimodal_pdf.py`：六步编排 + `--book` 自然排序章节聚合 + `--dry-run` 成本预估 + `--replace` 幂等守卫 + `ingest_report.json`

**MM-2/MM-4 检索 API**

- `documents/document_repository.py`：`document_sets` 新表（filter/enumerated 两型，重名 422 / chunk_ids≤500 / 失效计数）+ rag_documents.metadata 书籍聚合
- `documents/document_service.py`：**filter 归一单点**（books→PG 展开 ∪ chapters→去重并集，消除书章交集陷阱；显式传入但展开为空→422 拼写守卫；filters 与 set_id 互斥 422）+ 双路检索（multimodal dense / text 全库 RRF+rerank）
- `documents/document_api.py`：七端点——`POST /ask`、`GET /collections`、`GET /page-image`（防穿越内聚 asset store）、`GET /filters`、`GET/POST/DELETE /sets`、`GET /chapters/{id}/chunks`（MM-4 分页）
- 接缝：factory `milvus_multimodal` 分支 + 混配矩阵（mm+sparse≠none 即 raise）+ `NoneSparseBackend` + `rag_service.answer_question` 入口 3 行守卫；`delete_ingested_document` 多模态级联分支（Milvus→PG→资产，§20.1 顺序）

**MM-3 前端**：`/documents` 页（库切换、书/章**独立勾选** FilterBar + 搜索、集合下拉 + 存为集合、证据卡勾选策展 + SaveSetDialog 浮条、MM-4 章节内容抽屉、localStorage 持久化 + 失效清洗）+ 7 个同源代理路由；首页加导航入口

**T2.5 评测**：`scripts/gen_multimodal_evalset.py`（分层采样 书×kind → 三题型 LLM 起草——**跳数=生成时设计**（喂 N chunk 即 N 跳），gold_chunk_ids + scope 硬断言不经 LLM，n-gram 泄漏自检）+ `scripts/run_multimodal_eval.py`（检索→硬断言（HitRate/GoldRecall@k/MRR/scope 违规率，纯计算）→ 可选 RAGAS 软评分 → ±2% 门禁对比 baseline）

### 真实验收结果（Flink 第一章 14 页 + PEFT 论文 3 章聚合）

| 验收项 | 结果 |
|---|---|
| 入库 | Flink 14 页 20 块 + PEFT 3 章 210 块（9911-9913），dim=2048（ark 实测与设计记录一致） |
| 幂等 | 同 id `--replace` 重跑点数不变 ✅ |
| 检索 | "有界流和无界流的定义" top1 正中 Flink 第 2 页定义块（score 0.50）✅ |
| 书籍聚合（§10-7） | `/filters` 1 书 3 章（ch1/ch2/ch10 验证自然排序）；`chapters:[9912]` 过滤命中仅该章 ✅ |
| 删除级联（§20.1） | 先删旧 9902：Milvus 点 + PG 行 + 资产目录全清 ✅ |
| 零改动（§10-1/9） | `/ask` 冒烟 200 + 答案正确（Q3 2025 净销售额 $94,036M）；69 存量单测全绿；text 路传 filters 忽略+告警 ✅ |
| 错误契约（§15） | 422 拼写守卫（含缺失清单）/ filters+set_id 互斥 / 集合重名 422 / 双传 422 / 删除集合后 set_id 检索 404 全部按设计 ✅ |
| **T2.5 基线** | **HitRate@8=0.917 · GoldRecall@8=0.708 · MRR=0.861 · scope_violation=0**（报告在 `tools/data/multimodal_eval/reports/`；聚合题 GoldRecall 0.5 为排序质量待改进项，正是 MRR 度量点） |
| 单测 | **142 项全绿**（存量 69 + 新增 73，覆盖 §17 表全部行）；新改文件 ruff 全绿 |

**测试集**：12 题校准版 gold set 在 `src/agent/tools/data/multimodal_evalset.json`。泄漏检测升级为**最大连续公共段规则**（首版 8-gram 计数把专名误报 6/12——Data Artisans/Structured Streaming 等出题必需名词；升级后 3 道真泄漏题人工重写，最终 0 标记）。

### 续作交付（读待办清单后继续开发）

- **资产 GC**（`scripts/multimodal_assets_gc.py`）：Milvus image_ref 对账，dry-run 默认 / `--apply`；`ingest_report.json` 白名单；`page_*.jpg` 约定寻址规则（文档仍有点时保留）。真数据 dry-run 0 孤儿（反向印证删除级联无残留）
- **二期 MinIO 全链**：compose 独立 `rag-minio`（宿主 9002/9003，不复用 milvus 内嵌/langfuse 的 minio）+ config MINIO_* 5 字段 + `MinioAssetStore`（key 同构 `{doc_id}/{name}`，一期→二期零迁移契约）+ `scripts/multimodal_assets_migrate.py`（size 全量比对 + 5 抽样 md5 字节校验 + `--reverse` 回滚）。真机迁移 14 文件 verified；`MULTIMODAL_ASSET_STORE` 默认仍 local
- **页图资产补齐（设计偏差修正）**：§6 要求 `page_{n}.jpg` 始终落盘（证据缩略图 + fitz 降级唯一视觉产物），原实现只存插图裁剪——已补齐（两种解析后端统一落盘）
- **R2 链路改造（迁移文档 §6.2 第三行）**：`rag_evaluation_jobs` 加 `reference` 列（ALTER IF NOT EXISTS 已对真库生效）；`enqueue_evaluation_job(reference=)`；`_score_job` 按有/无参考路由——有参考 = faithfulness + context_precision_with_reference + context_recall + factual_correctness（全 LLM-only；AnswerCorrectness/AnswerSimilarity 等 embeddings 系留 R3），无参考 = 原两指标。**指标覆盖 2→5**

### 注意事项（重要）

1. **顺带修复 M5' 遗留 bug**：`sparse_query_profiles.py` 直接引用已删除的 `OPENSEARCH_SPARSE_SEARCH_SCOPE` config 字段 → AttributeError。即 **2026-09-15 M5' 之后整个稀疏查询计划构造路径都是断的**（存量单测用 FakeClient 覆盖不到真实路径），由 /documents text 路首跑暴露；已改 `getattr(config, ..., None)` 防御读取，默认 finance profile。`/ask` 与 text 路同时受益。
2. **工作区状态**：本次全部产出 + 会话前既有改动均未提交（58 个文件），按仓库惯例验收后应尽快提交。开发过程中一次 `git stash` 误操作已完整恢复并逐字节核验（`git diff stash` 为空后 drop），无丢失。
3. **当前库形态**：fitz 降级入库（本机无 dots.ocr/vLLM 服务）→ 全部为 text 块、无图片块。**R4 多模态指标与图文混合验收需先起 vLLM 解析服务重入库含图 PDF**。前端/后端服务为会话内临时启动，重启命令：`uvicorn api.server:app --port 8000`（src/agent 下）+ `npm run dev`（src/frontend 下）。

### 剩余待办（不阻塞本计划）

| 项 | 依赖 | 说明 |
|---|---|---|
| R2 补标脚本 + 100 题 reference 生成 | LLM 跑批成本 | 链路改造已完成，补标后即可解锁有参考指标实战 |
| R3 全量指标接入 | R2 补标 | embeddings 系指标（AnswerCorrectness/AnswerSimilarity/ResponseRelevancy）需给评测 worker 接 embeddings |
| R4 多模态指标 | vLLM 真机入库含图 PDF | MultiModalFaithfulness/Relevance 接入 T2.5 评测 |
| R5/R6/R7 | R2/R3 | LLM 选型排行榜 / 组件级 A/B / 持续评测回归 |
| 第五部分通用化（用户上传 PDF Web 化） | 多模态底座（已就绪） | 迁移文档 §8.5 |
| dots.ocr vLLM 部署位置决策 | — | 详设 §12 开放问题；决定后 Flink/课件可重入库获得图片块 |

## 代码审查与提交记录（2026-09-19）

工作区 58 文件改动已按逻辑分 5 个提交入库（均未推送远端）：`6e1c931` 检索栈收尾+R2 链路 → `e6bb290` 多模态模块 → `0649e23` documents API+脚本+MinIO+数据 → `054e200` 前端 /documents → `61b3d03` docs+.gitignore（排除 `.tmp_minio_image/`、`*.bundle`）。提交前完成四路并行代码审查（检索核心 / 多模态模块 / API+脚本 / 前端+测试），142 单测全绿、待提交文件无密钥泄漏。

### 审查待修复清单（后续续作队列）

**P0（数据安全，优先修）**

- `scripts/multimodal_assets_gc.py`：collection 不存在时静默返回空对账，`--apply` 下会把全部磁盘资产判孤儿删除——需加硬互锁（空对账+磁盘非空 → 拒绝 `--apply`，除非显式 `--force`）；同源风险：Milvus 对账查询 16384 上限静默截断，超限后活跃文档资产会被误删（需行数与 count 校验不符即 abort）。

**P1**

- `dense_milvus_multimodal.py:185`：先删后插顺序错误——`--replace` 重跑时 embed 失败/dim 不匹配会把既有向量清空（应先向量化+校验，再 delete+insert）。
- `dots_ocr_client.py`：bbox 重映射空转（input 尺寸被设为原图，scale 恒 1.0，§14.1-1 首验日志自证失效，待真机验证裁剪偏移）；`layout_to_md` 不产出 `#` 前缀，dots.ocr 主路径标题分块契约待真实模型输出验证（fitz 路不受影响）。
- `multimodal_vectorizer.py:64`：FixedWindowRateLimiter 无互斥，并发唤醒互相重置窗口、计数蒸发（全程持 `asyncio.Lock`）。
- documents 端点：async 路由内同步 pymilvus/文件 IO/MinIO 调用阻塞事件循环（改 `def` 或 `to_thread`）；`delete_set`/`page_image` 缺 502 兜底（§15 契约）。
- `run_multimodal_eval.py:142`：存在 embed 失败行时 p95 计算 IndexError（索引基于过滤后列表）；门禁 FAIL 退出码仍为 0（应 `sys.exit(1)`）。
- 前端 `page.tsx`：localStorage 先清空后恢复的竞态（facet 请求失败丢持久化勾选，需 hydrated 标志）；ChapterDrawer 无 stale guard（慢响应覆盖新响应+切章闪旧数据）；删除集合静默失败（无 res.ok 检查）。
- `test_multimodal_vectorizer.py`：依赖本机 `.env` 的 `OPENAI_BASE_URL`，无 `.env` 环境必红（monkeypatch base url）。
- `ingestion_service.py`：DENSE_BACKEND=milvus_multimodal 时文本 ingest 无守卫，向量静默丢失（入口 fail-fast）。

**P2 摘要**：注释/文档残留 Qdrant/OpenSearch 表述、`requirements.txt` torch `file:///C:/` 本机 wheel 路径（其它机器装不上）、delete 脚本 `--skip-pg` 分支漏多模态检测、create_set 并发重名 TOCTOU 应 422、chunker `embed_fn` 抛异常不降级、dots.ocr 每页新建 OpenAI 客户端、`rag_service` 多模态守卫返回 500 而非 4xx 等。

## 2026-09-20 全量续作：审查修复闭环 + R2/R3/R4/1C/第五部分通用化

上一节清单**全部修完**（1 P0 + 8 P1 + P2 批量），并完成主计划剩余需求开发。四波推进：Wave1 三线并行修复（后端/评测线/前端）→ Wave2 documents 后端+前端对接 → Wave2.5 R4+1C → Review 轮（三路并行审查）→ 修复轮（1 P0 + 4 P1 后端 + 2 P1 前端 + P2 批量全修）。**单测 142 → 280 全绿；ruff 改动文件全清；tsc/eslint 双零。**

### 交付内容

- **审查修复闭环**：GC 三道互锁（collection 缺失/空对账+磁盘非空/行数截断 → `--apply` abort + `--force` 逃生门）；dense_milvus_multimodal 先全量向量化+dim 校验再 delete+insert；FixedWindowRateLimiter 全程持锁；dots.ocr smart_resize 预缩放（bbox 重映射真实生效）+ Title/Section-header→`#`/`##` 前缀（主路径标题分块契约修复，chunker `_HEADER_RE` 兼容）+ 客户端复用 + bbox 容错；ingestion 三入口多模态守卫；/ask 守卫 500→409（含 stream 响应头前失败）；删除级联顺序改为 **向量→资产→PG 最后**（部分失败可重试）；rerank 非法 backend 告警；测试密封性修复。
- **R2 补标**：`scripts/generate_reference_answers.py`（检索增强出参考答案 + checkpoint 断点续跑 + `--enqueue` 入队）。真机跑批 **84/100 成功**（`tools/data/narrative_reference_answers.json`），16 题因 **ark 账户月度配额耗尽**（429 AccountQuotaExceeded，2026-09-23 23:59 重置）失败——重置后重跑同命令即自动续跑补齐。
- **R3 embeddings 指标**：`evaluation_pipeline` 有参考路由追加 AnswerCorrectness/AnswerSimilarity/ResponseRelevancy（`EVAL_EMBEDDINGS_METRICS_ENABLED` 默认关，构造失败自动降级回 4 指标）。
- **R4 多模态指标**：`run_multimodal_eval.py --ragas-multimodal` flag（默认关），MultiModalFaithfulness/Relevance 双路径导入验证可用，逐行降级不影响硬断言与门禁；真机出分待 vLLM 重入库含图 PDF。
- **1C 融合下沉**：`rrf_scores/rrf_fuse` 纯函数抽取 + `milvus_store.hybrid_search`（RRFRanker k=60，dense/sparse 等深池）+ `FUSION_BACKEND=app|milvus`（**默认 app 行为不变**）+ 一致性单测 11 项。真机对账（rag_nodes 7541 点）：**overlap@10=10/10、分数逐位一致（±1e-6）、耗时 64.7ms→10.7ms**。
- **第五部分通用化（§8.5）**：9 端点——`POST /upload`（multipart `file`+可选 `collection`，同步六步入库，UploadRejectedError→422 矩阵）、`GET /documents`（裸数组）、`DELETE /documents/{id}`（共享级联）、`POST/GET /collections`（动态 collection v1：注册表+Milvus 建库+kind 字段；**v1 简化：embedding_provider 仅登记，未做按库切模型**）、ask `collection` 路由、`generate-testset`/`testset/{job_id}`/`evaluate`（复用 T2.5 出题/评测核心，进程内 jobs v1）。前端：上传面板（拖拽/校验/集合选择/新建集合）、文档列表管理、动态库下拉、EvalPanel（生成→轮询→运行评测→指标摘要，链式 setTimeout+代际 token 防竞态）。
- **上传可靠性**：document_id 分配竞态修复（**占位行状态机 ingesting→completed/failed**，UniqueViolation 重试，列表过滤非 completed）；同步段全部 `to_thread` 出事件循环（上传期间不再冻结 /ask）；页数守卫前置光栅化（防 OOM）；PG 注册失败 best-effort 回滚 Milvus+资产。
- **附带修复**：`delete_ingested_document.py` argparse dest 不匹配必崩（review 轮新发现 P0）；评测门禁纳入 scope_violation（lower=better）；apply_ragas_multimodal never-raise 收紧；JOBS 上限 50 淘汰；collections 探测并发化。

### 剩余待办（更新）

| 项 | 依赖 | 说明 |
|---|---|---|
| R2 补齐 16 题 | ark 配额 2026-09-23 重置 | 重跑 `generate_reference_answers.py`（checkpoint 自动续） |
| R3 实战出分 | R2 补齐 | 开 `EVAL_EMBEDDINGS_METRICS_ENABLED=true` 后跑评测 |
| R4 真机出分 | dots.ocr vLLM 部署决策 | flag 已就绪；含图 PDF 重入库后验收 |
| 1C 切换 milvus 融合 | 评测门禁 | 代码+一致性已就绪（默认 app），M6 门禁过了再切 |
| 动态 collection 按库切 embedding 模型 | — | v1 仅登记未实现 |
| jobs 持久化 | — | 进程内 dict，重启丢失（v1 声明） |
| R5/R6/R7 | R2/R3 实战数据 | 选型排行榜/组件 A/B/持续回归 |
| dots.ocr vLLM 部署决策 | — | 详设 §12；bbox 裁剪偏移真机验证同批做 |
