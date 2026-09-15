"""答案生成后的"证据引文抽取"模块：LLM 挑选 + 逐字校验，为前端证据卡片（evidence cards）提供数据。

【设计背景 / 模块定位】
在 Ask 管线中，答案由 LLM 基于检索上下文生成后，本模块在同一请求内运行（调用方见
rag_service.py，受 ASK_EVIDENCE_LLM_EXTRACT_ENABLED 开关控制，默认开启）。

核心设计原则——"模糊判断交给 LLM，精确验证交给代码"：
  1. LLM 负责"哪句话最能支撑答案"这类语义判断（无唯一正确答案，适合模型）；
  2. 代码负责"这句话是否逐字存在于源文本"这类二值判断（用 str.find() 子串查找，
     确定性、零成本、不可能出错），防止 LLM 编造/改写证据（幻觉）。

约束方向说明：本模块只保证【引文卡片】与 chunk 原文逐字一致；答案本身允许
（也必须允许）跨 chunk 综合、翻译、计算，不受逐字约束。答案与引文之间的语义
对齐靠抽取 prompt 约束 + 用户对照卡片核查来兜底。

产出结构（每个元素是一张卡片的原始数据，由 rag_service 转成 narrative_cards）：
  {body, document_id, node_id, relevance_score, relevance_level, accn?}

上游依赖：
  - tools/llm.py            LLM 调用（temperature=0 保证输出稳定）
  - core/config.py          ask_evidence_llm_* 系列配置（来源数/引文数/长度上下限等）
  - tools/rag_stage_log.py  管线阶段日志（失败只记 warning，不阻断主流程）
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from tools.llm import get_llm
from tools.rag_stage_log import log_rag


def _norm_accn(value: Any) -> str | None:
    """把任意值规范化为 accession number 字符串；空值统一返回 None。

    accession number 是 SEC filing 的唯一编号（形如 0000320193-24-000123），
    用于证据卡片的溯源展示。入参可能是 str / None / 其他类型，这里统一
    str() + strip()，并把空串折叠成 None，方便上层用 `if accn:` 判空。
    """
    text = str(value or "").strip()
    return text or None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    """安全取出节点的 metadata 字典；缺失或类型不对时返回空 dict。

    检索管线返回的 node 结构可能来自不同后端（Qdrant/OpenSearch/Postgres），
    metadata 不保证存在，也不保证是 dict。这里做防御性读取，避免上层每个
    使用点都写 isinstance 判断。
    """
    metadata = node.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _node_accessions(node: dict[str, Any]) -> tuple[str, ...]:
    """收集节点上所有可能的 accession number，去重后返回。

    由于入库来源不同（companyfacts 对齐 vs EDGAR 解析），accession 可能
    挂在 node 顶层，也可能藏在 metadata 里；字段名有 finance_accns（SQL
    事实对齐写入的列表）和 sec_accession（ filing 元数据）两种。这里把
    四个位置都查一遍，按出现顺序去重。

    返回 tuple 而非 list：表达"只读快照"语义，防止调用方意外修改。
    """
    meta = _node_metadata(node)
    raw_values = [
        node.get("finance_accns"),
        meta.get("finance_accns"),
        node.get("sec_accession"),
        meta.get("sec_accession"),
    ]
    out: list[str] = []
    for raw in raw_values:
        # 单个字段既可能是列表（多个 filing 对齐到同一节点）也可能是单值，统一展开
        values = raw if isinstance(raw, list) else [raw]
        for item in values:
            accn = _norm_accn(item)
            if accn and accn not in out:
                out.append(accn)
    return tuple(out)


def _first_accession_per_document(nodes: list[dict[str, Any]]) -> dict[int, str]:
    """构建 document_id → accession 的兜底映射（每个文档取第一个遇到的 accession）。

    用途：某些节点自身没带 accession（例如纯 RAG 命中的叶子节点），
    但同一文档下通常有别的节点带。此映射作为"节点缺 accession 时按
    文档回填"的后备数据源（见 _accn_for_node）。

    因为入参已按相关度降序排列，"第一个遇到的"即"最相关节点上的 accession"。
    """
    doc_to_accn: dict[int, str] = {}
    for node in nodes:
        doc_id = node.get("document_id")
        if doc_id is None:
            continue
        did = int(doc_id)
        if did in doc_to_accn:
            continue  # 已有该文档的 accession，保留最相关那个
        accns = _node_accessions(node)
        if accns:
            doc_to_accn[did] = accns[0]
    return doc_to_accn


def _node_relevance(item: dict[str, Any]) -> float:
    """提取节点相关度分数：优先 rerank 分，退化用检索原始分。

    经过 reranker 的节点带 rerank_score（更可靠）；如果管线没开
    rerank（或被跳过），节点只有混合检索的 score。取不到时按 0 分
    处理，保证排序逻辑不会因缺字段而崩。
    """
    r = item.get("rerank_score")
    if r is not None:
        return float(r)
    return float(item.get("score") or 0.0)


def _strip_json_fence(text: str) -> str:
    """剥掉 LLM 输出可能裹的 markdown 代码围栏（```json ... ```）。

    尽管 system prompt 明确要求"只输出 JSON、不要围栏"，模型仍偶发违反。
    与其解析失败后重试，不如宽松清洗后再 json.loads——这里只处理
    开头的 ```（可带语言标记）和结尾的 ```，不碰正文。
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _is_mostly_md_table(quote: str) -> bool:
    """判断一段文本是否"大部分是 markdown 表格行"（以竖线分隔为主的行）。

    设计动机：markdown 表格截断成证据卡片后完全没法读（竖线满天飞），
    且 LLM 复制表格行时极易在空白上出错导致逐字校验失败。system prompt
    已要求 LLM 跳过表格，这里是代码侧的第二道闸门——即使 LLM 违规
    返回了表格引文，也在这里拦下。

    判定标准（前 16 个非空行内）：含 >= 2 个 '|' 的行数过半即视为表格。
    """
    lines = [ln for ln in quote.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False  # 单行不可能是表格
    head = lines[: min(16, len(lines))]  # 只看开头，长引文无需全文扫描
    pipe_lines = sum(1 for ln in head if ln.count("|") >= 2)
    return pipe_lines >= max(2, (len(head) + 1) // 2)


def _verify_verbatim(quote: str, excerpt: str) -> str | None:
    """逐字校验：quote 是否是 excerpt 中连续出现的原样子串；命中则返回原文切片。

    这是整个模块"防伪造证据"的核心——用确定性的 str.find() 取代 LLM 判断
    "是否一致"，因为 LLM 做逐字比对会漏检改写（语义相同≠逐字相同）且结果
    不稳定，而子串查找不可能出错。

    返回值设计：命中时返回从 excerpt 切出的片段（而非原样返回 quote），
    保证展示给用户的一定是源文本的原生字符（包括内部空白）；未命中返回
    None，调用方据此丢弃该候选引文。

    两轮查找：
      第 1 轮：原文直接 find（覆盖绝大多数情况）；
      第 2 轮：把 \r\n 统一成 \n 再 find——Windows 行尾差异不应导致
              "真实存在的引用"被误杀（唯一允许的宽容度）。
    注意：除换行归一外不做任何其他模糊匹配（不做大小写折叠、不做
    空白折叠），改写/纠错/翻译的引文一律判为不合格。
    """
    if not quote or not excerpt:
        return None
    q_strip = quote.strip()  # 只宽容首尾空白（LLM 输出常带前后换行）
    if not q_strip:
        return None
    pos = excerpt.find(q_strip)
    if pos >= 0:
        return excerpt[pos : pos + len(q_strip)]
    # 第 2 轮：CRLF -> LF 归一后重试
    en = excerpt.replace("\r\n", "\n")
    qn = q_strip.replace("\r\n", "\n")
    pos = en.find(qn)
    if pos < 0:
        return None
    return en[pos : pos + len(qn)]


def _accn_for_node(node: dict[str, Any], doc_accn: dict[int, str]) -> str | None:
    """为节点解析 accession：节点自带优先，否则回退到同文档兜底映射。

    两级查找：
      1. 节点自身携带（finance_accns / sec_accession，见 _node_accessions）
         ——最精确，因为节点可能被多个 filing 的内容对齐过；
      2. _first_accession_per_document 建立的文档级映射——覆盖
         "节点本身没带 accession，但同文档更相关的节点带了"的情况。
    """
    accns = _node_accessions(node)
    if accns:
        return accns[0]
    doc_id = node.get("document_id")
    if doc_id is None:
        return None
    return doc_accn.get(int(doc_id))


async def extract_answer_aligned_verified_quotes(
    *,
    question: str,
    answer: str,
    nodes: list[dict[str, Any]],
    locale: str,
) -> list[dict[str, Any]]:
    """入口函数：从已定稿答案 + 检索节点中，产出经过逐字校验的证据引文列表。

    整体流程（三段式）：
      阶段 1  准备源文本——按相关度截取 top-N 节点，拼成带编号的 SOURCE 块；
      阶段 2  LLM 抽取——temperature=0，让它从 SOURCE 里挑至多 max_quotes 条
              "直接支撑该答案"的原文连续子串（中英双语 prompt，按 locale 切换）；
      阶段 3  代码校验——对每条候选做长度/表格/逐字/去重四道过滤，只保留
              真实存在于源文本的引文，组装成卡片数据。

    失败语义：任何一步出错（LLM 异常、JSON 解析失败、无合格引文）都返回
    空 list 或截断结果——本模块是"锦上添花"的可信度增强，绝不阻断主问答流程
    （调用方 rag_service 还会再套一层 try/except 兜底）。

    参数：
      question: 用户原始问题（仅供 LLM 理解语境，不参与校验）
      answer:   已定稿的答案（只用于判断相关性，prompt 明确禁止改写它）
      nodes:    检索管线命中的节点（含 text/metadata/rerank_score 等）
      locale:   语言（"en"/"zh"），决定使用哪个版本的抽取 prompt
    """
    if not nodes or not (answer or "").strip():
        # 没有源文本或答案为空，无证据可抽——直接短路返回
        return []

    # 读取可调参数（均来自 config 的 ask_evidence_llm_* 环境变量），
    # 并对每个值做下限钳制，防止配置成 0 或负数导致 prompt 自相矛盾
    max_sources = max(1, int(config.ask_evidence_llm_max_sources))  # 送入 prompt 的最大节点数
    cap = max(500, int(config.ask_evidence_llm_chars_per_source))   # 每个节点文本的截断长度（控 prompt 体积）
    max_quotes = max(1, min(5, int(config.ask_evidence_llm_max_quotes)))  # 最终产出的最大引文数
    max_q_chars = max(200, int(config.ask_evidence_llm_max_quote_chars))  # 单条引文长度上限
    min_q_chars = max(20, int(config.ask_evidence_llm_min_quote_chars))  # 单条引文长度下限（防碎片化短引用）

    # ---- 阶段 1：准备源文本 ----
    # 按相关度降序取前 max_sources 个节点——只把最相关的候选交给 LLM，
    # 既控制 token 成本，也降低 LLM 从弱相关节点里硬凑证据的概率
    ranked = sorted(nodes, key=_node_relevance, reverse=True)[:max_sources]
    doc_accn = _first_accession_per_document(ranked)

    sources: list[dict[str, Any]] = []  # index -> {index, node, excerpt}，用于阶段 3 回查
    blocks: list[str] = []              # 拼进 prompt 的 SOURCE 文本块
    for i, node in enumerate(ranked, start=1):
        raw = str(node.get("text") or "")
        truncated = len(raw) > cap
        excerpt = raw[:cap] if truncated else raw
        # 截断时显式告知 LLM"后面还有内容，只能引用上方文本"，
        # 避免它引用被截掉的后半段导致逐字校验必然失败
        tail = "\n[Text truncated for this prompt — quote only from the text above.]" if truncated else ""
        node_id = str(node.get("node_id") or "")
        doc_id = node.get("document_id")
        title = str(node.get("title") or "").strip()
        # 每个 SOURCE 块带编号 + 元数据头，LLM 通过 source_index 引用回对应节点
        header = f"--- SOURCE {i} ---\nnode_id: {node_id}\ndocument_id: {doc_id}\ntitle: {title or 'n/a'}"
        blocks.append(f"{header}\n\n{excerpt}{tail}")
        sources.append(
            {
                "index": i,
                "node": node,
                "excerpt": excerpt,  # 保存截断后的文本——校验必须对着"LLM 实际看到的"文本做
            }
        )

    bundle = "\n\n".join(blocks)
    loc = "en" if str(locale).lower() == "en" else "zh"

    # ---- 阶段 2：LLM 抽取（中英双语 prompt）----
    # system prompt 的关键约束（与阶段 3 的代码校验一一对应）：
    #   * 只输出 JSON（对应 _strip_json_fence 的宽松清洗）
    #   * text 必须是 SOURCE 正文的"连续子串"原样复制（对应 _verify_verbatim）
    #   * 不得改写/翻译/纠错/拼接不相邻片段（防幻觉，find() 兜底）
    #   * 跳过 markdown 表格（对应 _is_mostly_md_table 第二道闸）
    #   * 长度区间限制（对应阶段 3 的 min/max_q_chars 过滤）
    if loc == "en":
        system = (
            "You select supporting evidence for a finance disclosure Q&A product. "
            "You MUST output a single JSON object only, no markdown fences. "
            "Schema: {\"quotes\": [{\"source_index\": <int 1-based>, \"text\": <string>}, ...]}. "
            f"Include at most {max_quotes} quotes. Each \"text\" MUST be copied exactly as a contiguous "
            "substring from the corresponding SOURCE block's body (after the header lines). "
            "Do not paraphrase, translate, fix typos, or merge non-adjacent spans. "
            "Prefer narrative sentences that directly support the given answer; skip markdown tables "
            "(lines dominated by pipe characters), boilerplate indexes, and generic section intros unless they alone support the answer. "
            f"Each quote must be at least {min_q_chars} characters and at most {max_q_chars} characters. "
            "If nothing qualifies, return {\"quotes\": []}."
        )
        user = (
            f"Question:\n{question}\n\n"
            # 强调"答案已定稿、仅用于判断相关性"——本模块不回写答案，
            # 答案与引文是"支撑"关系而非"逐字相等"关系
            "Answer (already finalized — use only to judge relevance; do not rewrite it):\n"
            f"{answer}\n\n"
            "Source excerpts:\n"
            f"{bundle}\n\n"
            "Return JSON only."
        )
    else:
        system = (
            "你是披露问答产品的证据抽取助手。只输出一个 JSON 对象，不要用 markdown 代码围栏。"
            "格式：{\"quotes\": [{\"source_index\": <从1开始的整数>, \"text\": <字符串>}, ...]}。"
            f"最多 {max_quotes} 条。每条 \"text\" 必须从对应 SOURCE 正文（标题行之后）原样复制连续子串，"
            "不得改写、翻译、纠错或拼接不相邻片段。"
            "优先选择与所给答案直接相关的叙述句；跳过以竖线表格为主的 markdown 表、附件索引、以及与答案无关的套话开篇。"
            f"每条长度须在 {min_q_chars}–{max_q_chars} 字符之间。若无合格片段，返回 {{\"quotes\": []}}。"
        )
        user = (
            f"问题：\n{question}\n\n"
            "答案（已定稿——仅用于判断相关性，不要改写）：\n"
            f"{answer}\n\n"
            "来源摘录：\n"
            f"{bundle}\n\n"
            "只返回 JSON。"
        )

    # temperature=0：证据抽取要的是稳定可复现，不要创造性
    llm = get_llm(model_name=config.default_model, temperature=0.0)
    try:
        resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        raw_out = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        # LLM 挂了：记 warning 返回空——证据缺失可以接受，主答案不受影响
        log_rag("answer_evidence_quotes_llm_error", level="warning", error=str(exc)[:400])
        return []

    try:
        payload = json.loads(_strip_json_fence(str(raw_out)))
    except json.JSONDecodeError as exc:
        # 模型输出不是合法 JSON（含清洗后仍失败的情况）：同样降级为空结果
        log_rag("answer_evidence_quotes_json_error", level="warning", error=str(exc)[:200])
        return []

    raw_quotes = payload.get("quotes") if isinstance(payload, dict) else None
    if not isinstance(raw_quotes, list):
        return []  # 结构不符（缺 quotes 字段或类型不对）——视为无合格引文

    # ---- 阶段 3：代码校验（每条候选引文过四道闸）----
    # 闸 1 长度区间  闸 2 非表格  闸 3 逐字存在  闸 4 去重
    out: list[dict[str, Any]] = []
    seen_norm: set[str] = set()  # 归一化后的引文指纹，用于跨节点去重

    for item in raw_quotes:
        if len(out) >= max_quotes:
            break  # 已收满，多余的直接丢弃
        if not isinstance(item, dict):
            continue
        try:
            src_i = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue  # source_index 缺失/非数字——无法定位来源节点
        text = str(item.get("text") or "").strip()
        # 闸 1：长度过滤——太短没有展示价值，太长不像"一句话引用"
        if len(text) < min_q_chars or len(text) > max_q_chars:
            continue
        # 闸 2：表格过滤——markdown 表格卡片不可读（prompt 拦一次，代码再兜一次）
        if _is_mostly_md_table(text):
            continue
        # source_index 必须落在实际提供的 SOURCE 编号范围内
        if src_i < 1 or src_i > len(sources):
            continue
        entry = sources[src_i - 1]
        excerpt = str(entry.get("excerpt") or "")
        # 闸 3（核心）：逐字校验——LLM 说"这是原文"不算数，find() 说才算
        verified = _verify_verbatim(text, excerpt)
        if not verified:
            continue  # 编造/改写的引文，丢弃
        # 闸 4：去重——按"空白折叠 + 小写 + 前 400 字符"生成指纹，
        # 不同节点里的同一段话（或换行差异的同一段）只展示一次
        norm_key = re.sub(r"\s+", " ", verified.lower())[:400]
        if norm_key in seen_norm:
            continue
        seen_norm.add(norm_key)

        node: dict[str, Any] = entry["node"]
        doc_id = node.get("document_id")
        if doc_id is None:
            continue  # 无法溯源到文档的引文没有卡片价值
        sc = _node_relevance(node)
        # 组装卡片数据：verified 是从源文本切出的原生片段，body 一定是原文
        card = {
            "body": verified,
            "document_id": int(doc_id),
            "node_id": str(node.get("node_id") or ""),
            "relevance_score": round(float(sc), 4),
            # 分级标签供前端着色：rerank 分数 >=0.55 高 / >=0.35 中 / 其余低
            "relevance_level": "high" if sc >= 0.55 else ("medium" if sc >= 0.35 else "low"),
        }
        accn = _accn_for_node(node, doc_accn)
        if accn:
            card["accn"] = accn  # accession 是可选字段——能溯源到 filing 编号才带上
        out.append(card)

    # 记录本阶段的产出统计（引文数/送入的源数/语言），供管线观测
    log_rag(
        "answer_evidence_quotes",
        quote_count=len(out),
        sources_in_prompt=len(sources),
        locale=loc,
    )
    return out
