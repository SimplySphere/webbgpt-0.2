from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import GroundingConfig, ServeConfig
from grounding.ingest import webb_sync
from grounding.provider import WebbGroundingProvider
from grounding.store import WebbKnowledgeStore
from provenance import (
    checkpoint_manifest,
    export_manifest,
    grounding_snapshot_manifest,
    tokenizer_manifest,
)
from repro import seed_everything
from rag.simple_index import LocalRagRetriever
from serve.backends.native_backend import NativeCheckpointChatBackend
from serve.backends.transformers_backend import TransformersChatBackend
from serve.backends.vllm_backend import VLLMChatBackend
from serve.orchestrator import AssistantOrchestrator
from serve.playground import render_playground_html


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageModel]
    tools: bool = True
    citations: bool = True
    safe_decode: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    tools: bool = True
    citations: bool = True
    safe_decode: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class ChatResponse(BaseModel):
    text: str
    used_tools: bool
    citations: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


def _serialize_citation(citation: object) -> dict:
    if isinstance(citation, dict):
        return dict(citation)
    if is_dataclass(citation):
        return asdict(citation)
    if hasattr(citation, "model_dump"):
        return citation.model_dump()
    payload = {}
    for field in ("source_type", "source_id", "label", "snippet", "metadata"):
        if hasattr(citation, field):
            payload[field] = getattr(citation, field)
    return payload or {"value": str(citation)}


def _append_transcript(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _build_repro_capsule(provenance: dict, response_metadata: dict) -> dict:
    checkpoint = provenance.get("checkpoint") or {}
    tokenizer = provenance.get("tokenizer") or {}
    catalog_snapshot = provenance.get("catalog_snapshot") or {}
    generation = response_metadata.get("generation") or {}
    return {
        "checkpoint_artifact_id": checkpoint.get("artifact_id"),
        "tokenizer_artifact_id": tokenizer.get("artifact_id"),
        "snapshot_id": catalog_snapshot.get("snapshot_id"),
        "decode_preset": generation.get("decode_preset") or provenance.get("decode", {}).get("preset"),
        "backend": generation.get("backend"),
        "seed_bundle": provenance.get("seed_bundle"),
    }


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _progressive_text_chunks(text: str) -> list[str]:
    parts = re.findall(r"\S+\s*", text)
    if not parts:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for part in parts:
        current += part
        if len(current) >= 14 or part.endswith(("\n", ". ", "? ", "! ")):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def apply_environment_overrides(config: ServeConfig) -> ServeConfig:
    updated = ServeConfig.from_dict(config.to_dict())
    checkpoint = os.environ.get("WEBBGPT_CHECKPOINT")
    if checkpoint:
        updated.checkpoint_path = checkpoint
    mode = os.environ.get("WEBBGPT_MODEL_MODE")
    if mode in {"pretrained", "sft", "dpo"}:
        updated.model_mode = mode
    elif checkpoint:
        lowered = checkpoint.lower()
        if "dpo" in lowered:
            updated.model_mode = "dpo"
        elif "sft" in lowered:
            updated.model_mode = "sft"
    use_rag = os.environ.get("WEBBGPT_USE_RAG")
    if use_rag is not None:
        updated.use_rag = use_rag.strip().lower() in {"1", "true", "yes", "on"}
    rag_index = os.environ.get("WEBBGPT_RAG_INDEX")
    if rag_index:
        updated.rag_index_path = rag_index
    rag_chunks = os.environ.get("WEBBGPT_RAG_CHUNKS")
    if rag_chunks:
        updated.rag_chunks_path = rag_chunks
    rag_top_k = os.environ.get("WEBBGPT_RAG_TOP_K")
    if rag_top_k:
        updated.rag_top_k = max(1, int(rag_top_k))
    rag_min_score = os.environ.get("WEBBGPT_RAG_MIN_SCORE")
    if rag_min_score:
        updated.rag_min_score = max(0.0, float(rag_min_score))
    rag_min_lexical_overlap = os.environ.get("WEBBGPT_RAG_MIN_LEXICAL_OVERLAP")
    if rag_min_lexical_overlap:
        updated.rag_min_lexical_overlap = min(1.0, max(0.0, float(rag_min_lexical_overlap)))
    rag_min_matched_terms = os.environ.get("WEBBGPT_RAG_MIN_MATCHED_TERMS")
    if rag_min_matched_terms:
        updated.rag_min_matched_terms = max(1, int(rag_min_matched_terms))
    rag_require_named_terms = os.environ.get("WEBBGPT_RAG_REQUIRE_NAMED_TERMS")
    if rag_require_named_terms is not None:
        updated.rag_require_named_terms = rag_require_named_terms.strip().lower() in {"1", "true", "yes", "on"}
    rag_min_top_score_margin = os.environ.get("WEBBGPT_RAG_MIN_TOP_SCORE_MARGIN")
    if rag_min_top_score_margin:
        updated.rag_min_top_score_margin = max(0.0, float(rag_min_top_score_margin))
    return updated


def _build_chat_backend(config: ServeConfig):
    if (Path(config.checkpoint_path) / "checkpoint.pt").exists():
        return NativeCheckpointChatBackend(config)
    try:
        return VLLMChatBackend(config)
    except RuntimeError:
        return TransformersChatBackend(config)


def build_app(config: ServeConfig) -> FastAPI:
    config = apply_environment_overrides(config)
    seed_bundle = seed_everything(config.seed)
    backend = _build_chat_backend(config)
    grounding_config = (
        GroundingConfig.from_dict(config.grounding.to_dict()) if config.enable_grounding and config.grounding else None
    )
    if config.enable_grounding and grounding_config is None:
        grounding_config = GroundingConfig()
    grounding_store = WebbKnowledgeStore(grounding_config.dsn) if grounding_config is not None else None
    if grounding_store is not None:
        grounding_store.create_schema()
    resolved_snapshot_id = None
    sync_status: dict[str, object] | None = None
    needs_sync = False
    if grounding_config is not None and grounding_store is not None:
        needs_sync = grounding_config.sync_on_start or grounding_store.get_snapshot(grounding_config.snapshot_id) is None
    if grounding_config and needs_sync:
        sync_mode = "sync_on_start" if grounding_config.sync_on_start else "bootstrap_on_missing_snapshot"
        try:
            sync_result = webb_sync(
                grounding_config.dsn,
                seed_url_pack=grounding_config.seed_url_pack,
                offline_seed_url_pack=grounding_config.offline_seed_url_pack,
                source_policy_path=grounding_config.source_policy_path,
                handbook_url=grounding_config.handbook_url,
                allow_ocr_fallback=grounding_config.allow_ocr_fallback,
                label=f"{config.model_name}-serve-sync",
                families=grounding_config.sync_families or None,
            )
            resolved_snapshot_id = sync_result.get("snapshot_id")
            sync_status = {
                "mode": sync_mode,
                "status": "completed",
                "snapshot_id": resolved_snapshot_id,
            }
        except Exception as exc:
            fallback_store = grounding_store
            fallback_snapshot = fallback_store.latest_completed_snapshot()
            if fallback_snapshot is None:
                raise RuntimeError(
                    "Webb grounding sync failed and no completed trusted grounding snapshot was available."
                ) from exc
            resolved_snapshot_id = fallback_snapshot.id
            sync_status = {
                "mode": sync_mode,
                "status": "failed_using_latest_completed_snapshot",
                "snapshot_id": resolved_snapshot_id,
                "error": str(exc),
            }
    if grounding_config and grounding_store is not None:
        if resolved_snapshot_id is None:
            resolved_snapshot_id = grounding_store.resolve_snapshot_id(grounding_config.snapshot_id)
        catalog_snapshot = grounding_snapshot_manifest(
            grounding_config.dsn,
            snapshot_id=resolved_snapshot_id,
            seed_url_pack=grounding_config.seed_url_pack,
            offline_seed_url_pack=grounding_config.offline_seed_url_pack,
            handbook_url=grounding_config.handbook_url,
            source_policy_path=grounding_config.source_policy_path,
            catalog_input_path=grounding_config.legacy_catalog_input_path,
        )
    else:
        catalog_snapshot = {}
    checkpoint_provenance = (
        checkpoint_manifest(config.checkpoint_path)
        if (Path(config.checkpoint_path) / "checkpoint.pt").exists()
        else export_manifest(config.checkpoint_path) or {"path": config.checkpoint_path}
    )
    provenance = {
        "checkpoint": checkpoint_provenance,
        "tokenizer": tokenizer_manifest(config.tokenizer_path),
        "catalog_snapshot": catalog_snapshot,
        "grounding_snapshot": catalog_snapshot,
        "decode": {
            "preset": config.decode_preset,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "stop_strings": config.stop_strings,
        },
        "seed_bundle": seed_bundle,
        "model_mode": config.model_mode,
        "device": str(getattr(backend, "device", "unknown")),
        "rag": {
            "enabled": bool(config.use_rag),
            "index_path": config.rag_index_path,
            "chunks_path": config.rag_chunks_path,
            "top_k": config.rag_top_k,
            "min_score": config.rag_min_score,
            "min_lexical_overlap": config.rag_min_lexical_overlap,
            "min_matched_terms": config.rag_min_matched_terms,
            "require_named_terms": config.rag_require_named_terms,
            "min_top_score_margin": config.rag_min_top_score_margin,
        },
    }
    if sync_status is not None:
        provenance["sync_on_start"] = sync_status
    grounding_provider = None
    if config.enable_grounding and grounding_config and grounding_store is not None:
        grounding_provider = WebbGroundingProvider(
            grounding_store,
            snapshot_id=resolved_snapshot_id,
            route_fanout_limit=grounding_config.route_fanout_limit,
            planner_beta_enabled=grounding_config.planner_beta_enabled,
        )
    rag_retriever = None
    if config.use_rag:
        rag_retriever = LocalRagRetriever(
            index_path=config.rag_index_path,
            chunks_path=config.rag_chunks_path,
            top_k=config.rag_top_k,
            min_score=config.rag_min_score,
            min_lexical_overlap=config.rag_min_lexical_overlap,
            min_matched_terms=config.rag_min_matched_terms,
            require_named_terms=config.rag_require_named_terms,
            min_top_score_margin=config.rag_min_top_score_margin,
        )
    orchestrator = AssistantOrchestrator(
        backend,
        grounding_provider=grounding_provider,
        rag_retriever=rag_retriever,
        default_max_tokens=config.max_new_tokens,
        default_temperature=config.temperature,
        default_top_k=config.top_k,
        default_top_p=config.top_p,
        default_repetition_penalty=config.repetition_penalty,
        default_no_repeat_ngram_size=config.no_repeat_ngram_size,
        default_stop_strings=config.stop_strings,
        decode_preset=config.decode_preset,
        backend_name=getattr(backend, "backend_name", backend.__class__.__name__),
        catalog_snapshot=catalog_snapshot,
    )

    app = FastAPI(title="WebbGPT 0.2")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return render_playground_html(config)

    @app.get("/status")
    async def status() -> dict[str, object]:
        return {
            "name": "WebbGPT 0.2",
            "status": "ok",
            "endpoints": [
                "/",
                "/status",
                "/healthz",
                "/v1/chat/completions",
                "/generate",
                "/generate_stream",
                "/docs",
            ],
            "checkpoint_path": config.checkpoint_path,
            "model_mode": config.model_mode,
            "device": str(getattr(backend, "device", "unknown")),
            "rag": {
                "enabled": bool(config.use_rag),
                "index_path": config.rag_index_path,
                "chunks_path": config.rag_chunks_path,
                "top_k": config.rag_top_k,
                "min_score": config.rag_min_score,
                "min_lexical_overlap": config.rag_min_lexical_overlap,
                "min_matched_terms": config.rag_min_matched_terms,
                "require_named_terms": config.rag_require_named_terms,
                "min_top_score_margin": config.rag_min_top_score_margin,
            },
            "provenance": provenance,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    def _complete_chat(request: ChatRequest, *, api_route: str) -> ChatResponse:
        response = orchestrator.respond(
            [message.model_dump() for message in request.messages],
            tools=request.tools,
            citations=request.citations,
            safe_decode=request.safe_decode,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        serialized_citations = [_serialize_citation(citation) for citation in response.citations]
        response.metadata.setdefault("api", {"route": api_route, "canonical": "/v1/chat/completions"})
        response.metadata.setdefault("provenance", provenance)
        response.metadata.setdefault("repro_capsule", _build_repro_capsule(provenance, response.metadata))
        if config.transcript_path:
            _append_transcript(
                config.transcript_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "api_route": api_route,
                    "request": request.model_dump(),
                    "response": {
                        "text": response.text,
                        "used_tools": response.used_tools,
                        "citations": serialized_citations,
                        "metadata": response.metadata,
                    },
                },
            )
        return ChatResponse(
            text=response.text,
            used_tools=response.used_tools,
            citations=serialized_citations,
            metadata=response.metadata,
        )

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> ChatResponse:
        return _complete_chat(request, api_route="/v1/chat/completions")

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> ChatResponse:
        chat_request = ChatRequest(
            messages=[ChatMessageModel(role="user", content=request.prompt)],
            tools=request.tools,
            citations=request.citations,
            safe_decode=request.safe_decode,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        return _complete_chat(chat_request, api_route="/generate")

    @app.post("/generate_stream")
    async def generate_stream(request: GenerateRequest) -> StreamingResponse:
        async def event_stream():
            yield _sse_event(
                "start",
                {
                    "streaming": "ui_progressive_rendering",
                    "message": "Generation request accepted. Final text will be progressively rendered after backend completion.",
                },
            )
            try:
                chat_request = ChatRequest(
                    messages=[ChatMessageModel(role="user", content=request.prompt)],
                    tools=request.tools,
                    citations=request.citations,
                    safe_decode=request.safe_decode,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
                response = await asyncio.to_thread(_complete_chat, chat_request, api_route="/generate_stream")
                response.metadata.setdefault("streaming", {})
                response.metadata["streaming"].update(
                    {
                        "route": "/generate_stream",
                        "mode": "ui_progressive_rendering",
                        "true_model_token_streaming": False,
                        "note": (
                            "The native backend does not expose safe token callbacks here; "
                            "the server preserves normal generation and progressively reveals the final text."
                        ),
                    }
                )
                for chunk in _progressive_text_chunks(response.text):
                    yield _sse_event("delta", {"text": chunk})
                    await asyncio.sleep(0.018)
                yield _sse_event(
                    "metadata",
                    {
                        "metadata": response.metadata,
                        "used_tools": response.used_tools,
                        "citations": response.citations,
                    },
                )
                yield _sse_event("done", {"ok": True})
            except Exception as exc:
                yield _sse_event("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_server(config: ServeConfig) -> None:
    app = build_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
