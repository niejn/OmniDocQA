"""Qwen3 reranker utilities: CrossEncoder loading/scoring + standalone CLI.

``load_cross_encoder`` is the production loader used by ``tools/local_reranker.py``;
``main()`` remains a standalone runner for quick manual checks::

    python src/agent/tools/qwen_reranker.py --top-k 3


The model returns a relevance probability in the ``[0, 1]`` range.  The
underlying CrossEncoder score is a yes/no logit difference; applying a
sigmoid makes the printed score easier to interpret and compare.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_DATA = Path(__file__).with_name("data") / "qwen_reranker_test_set.json"
logger = logging.getLogger(__name__)


def _format_memory(num_bytes: int) -> str:
    """Format bytes as decimal GB for the runtime log."""
    return f"{num_bytes / 10**9:.2f} GB"


def _log_peak_gpu_memory(torch_module: Any, stage: str) -> None:
    """Log PyTorch's peak allocated and reserved CUDA memory, if available."""
    if not torch_module.cuda.is_available():
        logger.info("[%s] CUDA 不可用，跳过 GPU 显存统计", stage)
        return

    # Synchronize because CUDA kernels are asynchronous by default.
    torch_module.cuda.synchronize()
    device_index = torch_module.cuda.current_device()
    allocated = torch_module.cuda.max_memory_allocated(device_index)
    reserved = torch_module.cuda.max_memory_reserved(device_index)
    free, total = torch_module.cuda.mem_get_info(device_index)
    device_name = torch_module.cuda.get_device_name(device_index)
    logger.info(
        "[%s] GPU=%s (%s), total=%s, free=%s, driver used=%s, "
        "peak allocated=%s, peak reserved=%s (累计峰值)",
        stage,
        device_index,
        device_name,
        _format_memory(total),
        _format_memory(free),
        _format_memory(total - free),
        _format_memory(allocated),
        _format_memory(reserved),
    )


def load_test_set(path: Path) -> list[dict[str, Any]]:
    """Load and validate the small question/document test set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"测试集必须是 JSON 数组: {path}")
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"第 {index + 1} 条缺少 question: {path}")
        documents = item.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"第 {index + 1} 条缺少 documents: {path}")
        if any(not isinstance(doc, dict) or not doc.get("text") for doc in documents):
            raise ValueError(f"第 {index + 1} 条包含空文档: {path}")
    return payload


def load_cross_encoder(
    model_name: str,
    *,
    max_length: int,
    quantization: str,
    torch_module: Any,
    cross_encoder_class: Any,
) -> Any:
    """Load the CrossEncoder, optionally with BitsAndBytes quantization."""
    if quantization == "none":
        return cross_encoder_class(model_name, max_length=max_length)

    if not torch_module.cuda.is_available():
        raise RuntimeError("4-bit/8-bit 量化需要 CUDA 版 PyTorch")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "量化模式需要 transformers 和 bitsandbytes，请先安装: "
            "python -m pip install -U transformers bitsandbytes"
        ) from exc

    compute_dtype = (
        torch_module.bfloat16
        if torch_module.cuda.is_bf16_supported()
        else torch_module.float16
    )
    config_kwargs: dict[str, Any] = {
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    if quantization == "4bit":
        config_kwargs["load_in_4bit"] = True
    else:
        config_kwargs["load_in_8bit"] = True

    quantization_config = BitsAndBytesConfig(**config_kwargs)
    return cross_encoder_class(
        model_name,
        max_length=max_length,
        device="cuda",
        automodel_args={
            "quantization_config": quantization_config,
            "device_map": "auto",
        },
    )


def rerank_questions(
    model: Any,
    test_set: list[dict[str, Any]],
    *,
    batch_size: int,
    top_k: int | None,
    torch_module: Any,
) -> list[dict[str, Any]]:
    """Score every question/document pair and return documents in rank order."""
    results: list[dict[str, Any]] = []
    for item in test_set:
        question = str(item["question"])
        documents = item["documents"]
        pairs = [(question, str(document["text"])) for document in documents]
        scores = model.predict(
            pairs,
            batch_size=batch_size,
            activation_fn=torch_module.nn.Sigmoid(),
            show_progress_bar=False,
        )
        _log_peak_gpu_memory(torch_module, f"question={item.get('id', 'unknown')}")
        ranked_documents = [
            {
                **document,
                "relevance_score": round(float(score), 6),
            }
            for document, score in zip(documents, scores, strict=True)
        ]
        ranked_documents.sort(key=lambda document: document["relevance_score"], reverse=True)
        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "reference_answer": item.get("reference_answer"),
                "ranked_documents": ranked_documents[:top_k] if top_k else ranked_documents,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--quantization",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="使用 BitsAndBytes 量化；8B 模型在 16GB GPU 上建议使用 4bit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于 0")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k 必须大于 0")

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖，请安装: pip install sentence-transformers "
            "'transformers>=4.51.0' bitsandbytes"
        ) from exc

    test_set = load_test_set(args.data)
    model = load_cross_encoder(
        args.model,
        max_length=args.max_length,
        quantization=args.quantization,
        torch_module=torch,
        cross_encoder_class=CrossEncoder,
    )
    _log_peak_gpu_memory(torch, "after model loading")
    results = rerank_questions(
        model,
        test_set,
        batch_size=args.batch_size,
        top_k=args.top_k,
        torch_module=torch,
    )
    _log_peak_gpu_memory(torch, "total run")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
