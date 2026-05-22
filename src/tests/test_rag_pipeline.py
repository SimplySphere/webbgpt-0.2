from __future__ import annotations

import json

from rag.simple_index import LocalRagRetriever, build_index_payload, query_index
from serve.quality import analyze_generation
from serve.orchestrator import AssistantOrchestrator
from serve.types import ChatMessage


class _CaptureBackend:
    def __init__(self):
        self.prompt = None
        self.kwargs = None

    def generate(self, prompt: str, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return "The context says the community includes students sharing routines and support."


class _FakeRetriever:
    def __init__(self, hits: list[dict]):
        self.hits = hits
        self.query_text = None

    def query(self, query: str):
        self.query_text = query
        return {"query": query, "no_hit": not self.hits, "hits": self.hits}


def test_query_index_returns_relevant_chunk_and_no_hit_for_unrelated_query():
    chunks = [
        {
            "chunk_id": "chunk-catalog",
            "source_file": "catalog.txt",
            "source_category": "rag_candidates",
            "title": "Course Planning",
            "text": "A prerequisite is required before taking a course. A recommendation is suggested preparation.",
            "word_count": 13,
            "sha256": "catalog-sha",
            "risk_level": "medium",
            "allowed_use": "RAG",
        },
        {
            "chunk_id": "chunk-community",
            "source_file": "community.txt",
            "source_category": "rag_candidates",
            "title": "Community",
            "text": "A boarding school community includes residential life, shared routines, and student support.",
            "word_count": 12,
            "sha256": "community-sha",
            "risk_level": "medium",
            "allowed_use": "RAG",
        },
    ]
    index = build_index_payload(chunks, chunks_path="chunks.jsonl")

    result = query_index(
        "What is the difference between a prerequisite and a recommendation?",
        index=index,
        chunks=chunks,
        top_k=1,
        min_score=0.0001,
    )

    assert result["no_hit"] is False
    assert result["hits"][0]["chunk_id"] == "chunk-catalog"

    no_hit = query_index(
        "Who won the championship?",
        index=index,
        chunks=chunks,
        top_k=1,
        min_score=0.0001,
        min_matched_terms=1,
    )

    assert no_hit["no_hit"] is True
    assert no_hit["hits"] == []


def test_query_index_rejects_missing_named_entity_even_with_topic_overlap():
    chunks = [
        {
            "chunk_id": "chunk-policy",
            "source_file": "policy.txt",
            "source_category": "rag_candidates",
            "title": "Dining Policy",
            "text": "The Webb context mentions dining expectations and student policy language.",
            "word_count": 10,
            "sha256": "policy-sha",
            "risk_level": "medium",
            "allowed_use": "RAG",
        }
    ]
    index = build_index_payload(chunks, chunks_path="chunks.jsonl")

    result = query_index(
        "What is the Hogwarts dining policy?",
        index=index,
        chunks=chunks,
        top_k=1,
        min_score=0.0001,
        min_lexical_overlap=0.4,
        min_matched_terms=2,
        require_named_terms=True,
    )

    assert result["no_hit"] is True
    assert result["diagnostics"]["rejected_reasons"]["missing_named_query_terms"] == 1


def test_local_rag_retriever_loads_json_files(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "index.json"
    chunks = [
        {
            "chunk_id": "chunk-science",
            "source_file": "science.txt",
            "source_category": "rag_candidates",
            "title": "Science Project",
            "text": "During a science project, the first step is to ask a focused question.",
            "word_count": 13,
            "sha256": "science-sha",
            "risk_level": "medium",
            "allowed_use": "RAG",
        }
    ]
    chunks_path.write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n", encoding="utf-8")
    index = build_index_payload(chunks, chunks_path=str(chunks_path))
    index_path.write_text(json.dumps(index), encoding="utf-8")

    retriever = LocalRagRetriever(index_path=index_path, top_k=1, min_score=0.0001, min_matched_terms=1)
    result = retriever.query("science project first step")

    assert result["no_hit"] is False
    assert result["hits"][0]["source_file"] == "science.txt"


def test_orchestrator_rag_no_hit_still_generates_for_main_chat():
    backend = _CaptureBackend()
    retriever = _FakeRetriever([])
    orchestrator = AssistantOrchestrator(backend, rag_retriever=retriever)

    reply = orchestrator.respond([ChatMessage(role="user", content="Who is the dean?")])

    assert reply.text == "The context says the community includes students sharing routines and support."
    assert backend.prompt is not None
    assert retriever.query_text == "Who is the dean?"
    assert reply.metadata["rag"]["queried"] is True
    assert reply.metadata["rag"]["no_hit"] is True
    assert reply.metadata["abstained_due_to_no_hits"] is False
    assert reply.metadata["status"]["final_label"] == "Generated"


def test_orchestrator_rag_hit_injects_context_and_reports_metadata():
    hit = {
        "chunk_id": "chunk-community",
        "score": 0.12,
        "source_file": "community.txt",
        "source_category": "rag_candidates",
        "title": "Community",
        "text": "A boarding school community includes residential life, shared routines, and student support.",
        "word_count": 12,
        "sha256": "community-sha",
        "risk_level": "medium",
        "allowed_use": "RAG",
    }
    backend = _CaptureBackend()
    orchestrator = AssistantOrchestrator(backend, rag_retriever=_FakeRetriever([hit]))

    reply = orchestrator.respond([ChatMessage(role="user", content="What does the context say about community?")])

    assert reply.used_tools is True
    assert "community includes students" in reply.text
    assert backend.prompt is not None
    assert "Use the retrieved context below when it is relevant" in backend.prompt
    assert "chunk-community" in backend.prompt
    assert reply.metadata["rag"]["retrieved_hits"] == 1
    assert reply.metadata["rag"]["hits"][0]["chunk_id"] == "chunk-community"
    assert "text_preview" in reply.metadata["rag"]["hits"][0]
    assert reply.metadata["status"]["final_label"] == "Generated with sources"


def test_orchestrator_rag_bad_generation_is_shown_with_weak_generation_label():
    class _BadBackend(_CaptureBackend):
        def generate(self, prompt: str, **kwargs):
            self.prompt = prompt
            self.kwargs = kwargs
            return "."

    hit = {
        "chunk_id": "chunk-prereq",
        "score": 0.2,
        "source_file": "catalog.txt",
        "source_category": "rag_candidates",
        "title": "Course Planning",
        "text_preview": "A prerequisite is required before taking a course. A recommendation is suggested preparation but not required.",
        "text": "A prerequisite is required before taking a course. A recommendation is suggested preparation but not required.",
        "word_count": 15,
        "sha256": "catalog-sha",
        "risk_level": "medium",
        "allowed_use": "RAG",
    }
    orchestrator = AssistantOrchestrator(_BadBackend(), rag_retriever=_FakeRetriever([hit]))

    reply = orchestrator.respond([ChatMessage(role="user", content="What is a prerequisite?")])

    assert reply.text == "."
    assert reply.metadata["status"]["final_label"] == "Weak generation"
    assert reply.metadata["status"]["answered"] is True
    assert reply.metadata["status"]["degenerate_output"] is True
    assert reply.metadata["status"]["retrieved_context_fallback"] is False
    assert reply.metadata["rag"]["hits"][0]["chunk_id"] == "chunk-prereq"


def test_manual_context_bad_generation_is_model_context_failure_not_retrieval_failure():
    class _BadBackend(_CaptureBackend):
        def generate(self, prompt: str, **kwargs):
            self.prompt = prompt
            self.kwargs = kwargs
            return "?"

    retriever = _FakeRetriever(
        [
            {
                "chunk_id": "should-not-query",
                "score": 0.5,
                "source_file": "x.txt",
                "source_category": "rag_candidates",
                "title": "X",
                "text": "Unused.",
                "word_count": 1,
                "sha256": "x",
                "risk_level": "medium",
                "allowed_use": "RAG",
            }
        ]
    )
    orchestrator = AssistantOrchestrator(_BadBackend(), rag_retriever=retriever)
    prompt = (
        "Context: A prerequisite is required before taking a course. "
        "A recommendation is suggested preparation but not required. "
        "Question: What is the difference?"
    )

    reply = orchestrator.respond([ChatMessage(role="user", content=prompt)])

    assert retriever.query_text is None
    assert reply.text == "?"
    assert reply.metadata["status"]["final_label"] == "Weak generation"
    assert reply.metadata["status"]["degenerate_output"] is True
    assert reply.metadata["manual_context"]["detected"] is True


def test_quality_gate_flags_punctuation_hyphen_garbage_and_copyright_residue():
    assert analyze_generation(".", grounded=True)["degenerate"] is True
    assert "punctuation_only" in analyze_generation("?", grounded=True)["reasons"]
    hyphen = analyze_generation("n-d-n-ch-n", grounded=True)
    assert hyphen["degenerate"] is True
    assert "hyphen_letter_garbage" in hyphen["reasons"]
    copyright_residue = analyze_generation("© 2012", grounded=True)
    assert copyright_residue["degenerate"] is True
    assert "copyright_date_residue" in copyright_residue["reasons"]
