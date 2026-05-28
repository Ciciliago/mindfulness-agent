"""Hybrid retrieval utilities for Weaviate (BM25 + vector + filter + rerank)."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_terms(text: str) -> List[str]:
    normalized = _normalize_text(text)
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", normalized)
    en_terms = re.findall(r"[a-z0-9_]{3,24}", normalized)
    terms = zh_terms + en_terms
    # keep order while removing duplicates
    deduped = list(dict.fromkeys(terms))
    return deduped[:40]


def expand_query_aliases(query: str) -> str:
    text = query or ""
    expanded = text
    alias_pairs = [
        ("MBSR", "mindfulness-based stress reduction 正念减压"),
        ("MBCT", "mindfulness-based cognitive therapy 正念认知疗法"),
    ]
    for short, full in alias_pairs:
        if short.lower() in text.lower() and full.lower() not in text.lower():
            expanded = f"{expanded} {full}"
    return expanded


def _text_overlap_score(query: str, text: str) -> float:
    q_terms = set(_extract_terms(query))
    if not q_terms:
        return 0.0
    d_terms = set(_extract_terms(text))
    if not d_terms:
        return 0.0
    inter = len(q_terms.intersection(d_terms))
    return inter / math.sqrt(len(q_terms) * len(d_terms))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_where_clause(filters: Dict[str, str]) -> Optional[Dict[str, Any]]:
    operands: List[Dict[str, Any]] = []
    for key, value in filters.items():
        if not value:
            continue
        operands.append(
            {
                "path": [key],
                "operator": "Equal",
                "valueText": value,
            }
        )
    if not operands:
        return None
    if len(operands) == 1:
        return operands[0]
    return {"operator": "And", "operands": operands}


def _doc_from_weaviate_obj(obj: Dict[str, Any]) -> Document:
    metadata = {
        "source": obj.get("source", ""),
        "title": obj.get("title", ""),
        "track": obj.get("track", ""),
        "topic": obj.get("topic", ""),
        "scenario": obj.get("scenario", ""),
        "doc_type": obj.get("doc_type", ""),
        "chunk_id": obj.get("chunk_id", ""),
        "source_url": obj.get("source_url", ""),
        "score": _safe_float((obj.get("_additional") or {}).get("score"), 0.0),
        "explain_score": (obj.get("_additional") or {}).get("explainScore", ""),
    }
    return Document(page_content=obj.get("text", ""), metadata=metadata)


def infer_retrieval_filters(
    query: str,
    skill1: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    text = _normalize_text(query)
    filters: Dict[str, str] = {}

    skill1 = skill1 or {}
    route = skill1.get("route", "")
    if route == "script_gen":
        filters["track"] = "skill_script"

    # Explicit intent hints
    if any(key in text for key in ["脚本", "引导语", "冥想练习"]):
        filters["track"] = "skill_script"
    elif any(key in text for key in ["什么是", "原理", "理论", "mbsr", "mbct"]):
        filters.setdefault("track", "rag_theory")

    scenario_map: Sequence[Tuple[List[str], str]] = [
        (["失眠", "睡不着", "睡前", "入睡", "助眠"], "睡眠"),
        (["焦虑", "紧张", "不安", "惊慌"], "焦虑"),
        (["压力", "压抑", "喘不过气", "工作量大"], "压力"),
        (["感恩", "感谢", "欣赏"], "感恩"),
    ]
    for keys, scenario in scenario_map:
        if any(k in text for k in keys):
            filters["scenario"] = scenario
            break

    return filters


class HybridWeaviateRetriever(BaseRetriever):
    """Weaviate hybrid retriever with optional metadata filters and rerank."""

    client: Any = Field(exclude=True)
    embedding_model: Embeddings = Field(exclude=True)
    class_name: str
    k: int = 6
    fetch_k: int = 18
    alpha: float = 0.55
    query_properties: List[str] = Field(default_factory=lambda: ["text", "title", "topic", "scenario"])
    return_properties: List[str] = Field(
        default_factory=lambda: [
            "text",
            "source",
            "title",
            "track",
            "topic",
            "scenario",
            "doc_type",
            "chunk_id",
            "source_url",
            "chunk_summary_zh",
            "chunk_title_zh",
        ]
    )

    class Config:
        arbitrary_types_allowed = True

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        if not query.strip():
            return []
        query = expand_query_aliases(query)

        top_k = limit or self.k
        where_clause = _build_where_clause(filters or {})
        try:
            vector = self.embedding_model.embed_query(query)
        except Exception as exc:
            logger.warning("Embedding query failed, fallback to keyword-only hybrid: %s", exc)
            vector = None

        effective_alpha = self.alpha if vector is not None else 0.0

        gql = (
            self.client.query.get(self.class_name, self.return_properties)
            .with_additional(["score", "explainScore"])
            .with_hybrid(
                query=query,
                properties=self.query_properties,
                vector=vector,
                alpha=effective_alpha,
            )
            .with_limit(max(self.fetch_k, top_k))
        )

        if where_clause is not None:
            gql = gql.with_where(where_clause)

        result = gql.do()
        objects = ((result or {}).get("data", {}).get("Get", {}) or {}).get(self.class_name, []) or []
        if not objects:
            for fallback_query in self._fallback_queries(query):
                fallback_gql = (
                    self.client.query.get(self.class_name, self.return_properties)
                    .with_additional(["score", "explainScore"])
                    .with_hybrid(
                        query=fallback_query,
                        properties=self.query_properties,
                        alpha=0.0,  # fallback is lexical-first for exact term recovery
                    )
                    .with_limit(max(self.fetch_k, top_k))
                )
                if where_clause is not None:
                    fallback_gql = fallback_gql.with_where(where_clause)
                fallback_result = fallback_gql.do()
                objects = (
                    ((fallback_result or {}).get("data", {}).get("Get", {}) or {}).get(self.class_name, []) or []
                )
                if objects:
                    break

        docs = [_doc_from_weaviate_obj(obj) for obj in objects]
        reranked = self._rerank(query=query, docs=docs, filters=filters or {})
        return reranked[:top_k]

    def _fallback_queries(self, query: str) -> List[str]:
        text = query or ""
        candidates: List[str] = []
        acronyms = re.findall(r"[A-Za-z]{3,8}", text)
        zh_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
        if acronyms:
            candidates.append(" ".join(dict.fromkeys([term.upper() for term in acronyms])))
        if zh_terms:
            candidates.append(" ".join(dict.fromkeys(zh_terms[:4])))
        mix = []
        if acronyms:
            mix.extend([term.upper() for term in acronyms])
        if zh_terms:
            mix.extend(zh_terms[:4])
        if mix:
            candidates.append(" ".join(dict.fromkeys(mix)))
        deduped = [c.strip() for c in dict.fromkeys(candidates) if c.strip() and c.strip() != text.strip()]
        return deduped[:3]

    def _rerank(self, query: str, docs: List[Document], filters: Dict[str, str]) -> List[Document]:
        if not docs:
            return []

        scored: List[Tuple[float, Document]] = []
        for doc in docs:
            base_score = _safe_float(doc.metadata.get("score"), 0.0)
            overlap = _text_overlap_score(query, doc.page_content)
            filter_boost = 0.0
            for key, value in filters.items():
                if str(doc.metadata.get(key, "")).strip() == value:
                    filter_boost += 0.15

            # hybrid score from weaviate + lexical overlap + filter consistency
            final_score = base_score + 0.45 * overlap + filter_boost
            doc.metadata["rerank_score"] = final_score
            scored.append((final_score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.search(query=query)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.search(query=query)
