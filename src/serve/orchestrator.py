from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
from typing import TYPE_CHECKING

from generation import default_stop_strings as default_generation_stop_strings
from rag.simple_index import format_rag_context
from serve.quality import analyze_generation
from serve.types import AssistantResponse, ChatMessage
from tokenizer import format_chat

if TYPE_CHECKING:
    from grounding.provider import GroundingProvider, WebbGroundingProvider


COURSE_CODE_RE = re.compile(r"\b[a-z]{2,6}\s?-?\d{2,4}[a-z]?\b", re.IGNORECASE)
MANUAL_CONTEXT_RE = re.compile(
    r"context\s*:\s*(?P<context>.+?)\s*(?:question\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


class AssistantOrchestrator:
    def __init__(
        self,
        backend,
        grounding_provider: GroundingProvider | WebbGroundingProvider | None = None,
        rag_retriever=None,
        default_max_tokens: int = 512,
        default_temperature: float = 0.7,
        default_top_k: int | None = 50,
        default_top_p: float = 0.95,
        default_repetition_penalty: float = 1.05,
        default_no_repeat_ngram_size: int = 4,
        default_stop_strings: list[str] | None = None,
        decode_preset: str = "serve",
        backend_name: str = "unknown",
        catalog_snapshot: dict | None = None,
    ):
        self.backend = backend
        self.grounding_provider = grounding_provider
        self.rag_retriever = rag_retriever
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_top_k = default_top_k
        self.default_top_p = default_top_p
        self.default_repetition_penalty = default_repetition_penalty
        self.default_no_repeat_ngram_size = default_no_repeat_ngram_size
        self.default_stop_strings = list(default_stop_strings or default_generation_stop_strings())
        self.decode_preset = decode_preset
        self.backend_name = backend_name
        self.catalog_snapshot = catalog_snapshot or {}

    def _route_decision(self, messages: list[ChatMessage]) -> tuple[str, bool, list[str]]:
        details = self._route_decision_details(messages)
        return details["route"], details["grounded"], details["school_years"]

    def _route_decision_details(self, messages: list[ChatMessage]) -> dict:
        if self.grounding_provider is None:
            return {
                "route": "chat",
                "grounded": False,
                "school_years": [],
                "candidate_routes": [],
                "fanout_routes": [],
                "low_confidence": False,
                "planner_beta_disabled": False,
            }
        if hasattr(self.grounding_provider, "route_messages"):
            decision = self.grounding_provider.route_messages(messages)
            metadata = dict(decision.metadata or {})
            return {
                "route": decision.route,
                "grounded": decision.grounded,
                "school_years": list(decision.school_years),
                "candidate_routes": list(metadata.get("candidate_routes") or [decision.route]),
                "fanout_routes": list(metadata.get("fanout_routes") or ([decision.route] if decision.grounded else [])),
                "low_confidence": bool(metadata.get("low_confidence", False)),
                "planner_beta_disabled": bool(metadata.get("planner_beta_disabled", False)),
            }
        latest_user = next((message.content.lower() for message in reversed(messages) if message.role == "user"), "")
        if COURSE_CODE_RE.search(latest_user):
            return {
                "route": "course_catalog",
                "grounded": True,
                "school_years": [],
                "candidate_routes": ["course_catalog"],
                "fanout_routes": ["course_catalog"],
                "low_confidence": False,
                "planner_beta_disabled": False,
            }
        keywords = [
            "course",
            "catalog",
            "prereq",
            "prerequisite",
            "program",
            "major",
            "minor",
            "credits",
            "section",
            "schedule",
            "offered",
            "requirement",
            "advisor",
            "advising",
            "semester",
        ]
        if any(keyword in latest_user for keyword in keywords):
            return {
                "route": "course_catalog",
                "grounded": True,
                "school_years": [],
                "candidate_routes": ["course_catalog"],
                "fanout_routes": ["course_catalog"],
                "low_confidence": False,
                "planner_beta_disabled": False,
            }
        return {
            "route": "chat",
            "grounded": False,
            "school_years": [],
            "candidate_routes": [],
            "fanout_routes": [],
            "low_confidence": False,
            "planner_beta_disabled": False,
        }

    def _should_ground(self, messages: list[ChatMessage]) -> bool:
        # Backward-compatible shim for older eval code paths that only need
        # the boolean routing decision.
        _, should_ground, _ = self._route_decision(messages)
        return should_ground

    def _ambiguous_timeframe(self, query: str, school_years: list[str]) -> bool:
        lowered = query.lower()
        current_markers = ("current", "currently", "now", "this year", "today")
        return bool(school_years) and any(marker in lowered for marker in current_markers)

    def _grounded_instruction(self, routes: list[str]) -> str:
        if "planner_advice" in routes:
            return (
                "Use grounded Webb facts when available. Clearly separate Webb-sourced facts from general advice. "
                "Do not present advice as a school rule."
            )
        if "handbook_policy" in routes:
            return (
                "Answer from the handbook support only. Distinguish handbook rules from any general advice and prefer "
                "page/section-aware citations when possible."
            )
        return (
            "Use the grounded Webb facts when relevant and do not invent current policies, course offerings, "
            "faculty roles, schedules, or admissions details."
        )

    def _grounding_context(
        self,
        query: str,
        *,
        routes: list[str],
    ) -> tuple[str, list, bool, int, str | None, list[str], list[str]]:
        citations = []
        routes_with_hits: list[str] = []
        routes_checked: list[str] = []
        retrieved_hits = 0
        snapshot_id = None
        context_blocks: list[str] = []
        seen_citations: set[tuple[str, str]] = set()
        for route in routes:
            if hasattr(self.grounding_provider, "route_messages"):
                result = self.grounding_provider.query(query, route=route)
            else:
                result = self.grounding_provider.query(query)
            routes_checked.append(route)
            if snapshot_id is None:
                snapshot_id = getattr(result, "snapshot_id", None)
            if not result.hits:
                continue
            routes_with_hits.append(route)
            retrieved_hits += len(result.hits)
            if hasattr(self.grounding_provider, "render_context"):
                rendered = self.grounding_provider.render_context(result)
            else:
                snippets = []
                for hit in result.hits:
                    source_list = ", ".join(citation.label for citation in hit.citations)
                    snippets.append(f"{hit.title}\n{hit.content}\n[source: {source_list}]")
                rendered = "\n\n".join(snippets)
            context_blocks.append(f"[{route.replace('_', ' ')}]\n{rendered}".strip())
            for citation in [citation for hit in result.hits for citation in hit.citations]:
                key = (citation.source_type, citation.source_id)
                if key in seen_citations:
                    continue
                seen_citations.add(key)
                citations.append(citation)
        if not context_blocks:
            return (
                "No matching grounded Webb sources were found. If the answer requires current Webb facts, state the uncertainty.",
                citations,
                False,
                0,
                snapshot_id,
                routes_checked,
                routes_with_hits,
            )
        return (
            "\n\n".join(context_blocks),
            citations,
            True,
            retrieved_hits,
            snapshot_id,
            routes_checked,
            routes_with_hits,
        )

    def _catalog_snapshot_label(self) -> str | None:
        sqlite_path = self.catalog_snapshot.get("sqlite_path")
        if sqlite_path:
            return Path(sqlite_path).name
        seed_url_pack = self.catalog_snapshot.get("seed_url_pack")
        if seed_url_pack:
            return Path(seed_url_pack).name
        input_path = self.catalog_snapshot.get("catalog_input_path")
        if input_path:
            return Path(input_path).name
        return None

    def _no_hit_abstention(self, query: str, *, route: str) -> str:
        if self.grounding_provider is not None and hasattr(self.grounding_provider, "no_hit_message"):
            return self.grounding_provider.no_hit_message(route, query)
        return (
            "I could not find a matching catalog entry in the current catalog snapshot, "
            f"so I cannot verify the answer to: {query}"
        )

    def _manual_context(self, query: str) -> str | None:
        match = MANUAL_CONTEXT_RE.search(query)
        if not match:
            return None
        context = re.sub(r"\s+", " ", match.group("context")).strip()
        return context or None

    def _rag_hit_metadata(self, hit: dict) -> dict:
        preview = str(hit.get("text_preview") or hit.get("text") or "").strip()
        if len(preview) > 420:
            preview = preview[:420].rsplit(" ", 1)[0].strip()
        return {
            "chunk_id": hit.get("chunk_id"),
            "score": hit.get("score"),
            "source_file": hit.get("source_file"),
            "source_category": hit.get("source_category"),
            "title": hit.get("title"),
            "text_preview": preview,
            "risk_level": hit.get("risk_level"),
            "allowed_use": hit.get("allowed_use"),
            "word_count": hit.get("word_count"),
            "matched_terms": list(hit.get("matched_terms") or []),
            "missing_terms": list(hit.get("missing_terms") or []),
            "lexical_overlap": hit.get("lexical_overlap"),
        }

    def _summary_line(
        self,
        *,
        used_tools: bool,
        route: str,
        retrieved_hits: int,
        abstained_due_to_no_hits: bool,
        degenerate_output: bool,
        safe_decode: bool,
        final_label: str = "Generated",
        retrieved_context_fallback: bool = False,
    ) -> str:
        if final_label == "Weak generation":
            if retrieved_hits:
                return "Local-MVP text was generated with retrieved context, but the quality check flagged low confidence."
            return "Local-MVP text was generated, but the quality check flagged low confidence."
        if final_label == "Generated with sources":
            return f"Local-MVP text was generated with {retrieved_hits} retrieved source chunk(s)."
        if final_label == "Generated":
            return "Local-MVP text was generated from the current model."
        if retrieved_context_fallback:
            return "Retrieved context was found, but generation failed; showing retrieved passages instead."
        if degenerate_output:
            suffix = "Safe decode was enabled." if safe_decode else "Safe decode was not enabled."
            return f"Local-MVP text was generated, but the quality check flagged malformed output. {suffix}"
        if abstained_due_to_no_hits:
            return f"Grounded {route.replace('_', ' ')} lookup found no hits, so the assistant abstained deterministically."
        if used_tools:
            return f"Grounded {route.replace('_', ' ')} path ran with {retrieved_hits} retrieved hit(s)."
        if safe_decode:
            return "Not grounded; answered from model only with the safe decode preset."
        return "Not grounded; answered from model only."

    def _base_metadata(
        self,
        *,
        used_tools: bool,
        route: str,
        candidate_routes: list[str] | None = None,
        routes_checked: list[str] | None = None,
        routes_with_hits: list[str] | None = None,
        low_confidence: bool = False,
        retrieved_hits: int,
        abstained_due_to_no_hits: bool,
        degenerate_output: bool,
        safe_decode: bool,
        effective_decode_preset: str,
        stop_reason: str,
        citation_labels: list[str],
        snapshot_id: str | None = None,
        quality: dict | None = None,
        raw_output: str | None = None,
        rag: dict | None = None,
        manual_context: dict | None = None,
        final_label: str = "Generated",
        retrieved_context_fallback: bool = False,
    ) -> dict:
        summary = self._summary_line(
            used_tools=used_tools,
            route=route,
            retrieved_hits=retrieved_hits,
            abstained_due_to_no_hits=abstained_due_to_no_hits,
            degenerate_output=degenerate_output,
            safe_decode=safe_decode,
            final_label=final_label,
            retrieved_context_fallback=retrieved_context_fallback,
        )
        return {
            "grounded": used_tools,
            "abstained_due_to_no_hits": abstained_due_to_no_hits,
            "decode_preset": effective_decode_preset,
            "backend": self.backend_name,
            "catalog_snapshot": self.catalog_snapshot,
            "grounding_snapshot": self.catalog_snapshot,
            "status": {
                "grounded": used_tools,
                "catalog": used_tools and route == "course_catalog",
                "cited": bool(citation_labels) and not degenerate_output,
                "abstained": abstained_due_to_no_hits,
                "degenerate_output": degenerate_output,
                "generation_failed": final_label == "Generation failed",
                "retrieved_context_fallback": retrieved_context_fallback,
                "answered": final_label in {"Generated", "Generated with sources", "Weak generation"},
                "final_label": final_label,
            },
            "summary": summary,
            "routing": {
                "mode": route,
                "route": route,
                "catalog_queried": used_tools and route == "course_catalog",
                "retrieved_hits": retrieved_hits,
                "candidate_routes": list(candidate_routes or []),
                "routes_checked": list(routes_checked or []),
                "routes_with_hits": list(routes_with_hits or []),
                "fanout_used": len(list(routes_checked or [])) > 1,
                "low_confidence": low_confidence,
            },
            "generation": {
                "backend": self.backend_name,
                "decode_preset": effective_decode_preset,
                "safe_decode": safe_decode,
                "stop_reason": stop_reason,
            },
            "grounding": {
                "queried": used_tools,
                "route": route,
                "retrieved_hits": retrieved_hits,
                "abstained_due_to_no_hits": abstained_due_to_no_hits,
                "citation_labels": citation_labels,
                "catalog_snapshot_label": self._catalog_snapshot_label(),
                "snapshot_id": snapshot_id or self.catalog_snapshot.get("snapshot_id"),
                "routes_checked": list(routes_checked or []),
                "routes_with_hits": list(routes_with_hits or []),
            },
            "rag": rag
            or {
                "enabled": self.rag_retriever is not None,
                "queried": False,
                "no_hit": False,
                "retrieved_hits": 0,
                "hits": [],
                "diagnostics": {},
            },
            "manual_context": manual_context or {"detected": False},
            "quality": quality or {"degenerate": False, "reasons": [], "metrics": {}},
            "debug": {
                "raw_output": raw_output,
            },
            "timeline": [
                {"label": "user input", "value": "received"},
                {"label": "routed as", "value": route},
                {"label": "retrieved hits", "value": str(retrieved_hits) if used_tools else "not queried"},
                {"label": "generation backend", "value": self.backend_name},
                {"label": "decode preset", "value": effective_decode_preset},
                {"label": "stop reason", "value": stop_reason},
            ],
        }

    def respond(
        self,
        messages,
        tools: bool = True,
        citations: bool = True,
        safe_decode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> AssistantResponse:
        parsed_messages = [
            message if isinstance(message, ChatMessage) else ChatMessage(**message) for message in messages
        ]
        query = next(message.content for message in reversed(parsed_messages) if message.role == "user")
        manual_context_text = self._manual_context(query)
        manual_context_info = {
            "detected": manual_context_text is not None,
            "context_preview": (manual_context_text or "")[:420],
            "context_characters": len(manual_context_text or ""),
        }
        rag_info = {
            "enabled": self.rag_retriever is not None,
            "queried": False,
            "no_hit": False,
            "retrieved_hits": 0,
            "hits": [],
            "instruction": None,
            "diagnostics": {},
            "skipped_reason": None,
        }
        route_details = self._route_decision_details(parsed_messages)
        route = route_details["route"]
        should_ground = route_details["grounded"]
        school_years = list(route_details["school_years"])
        candidate_routes = list(route_details["candidate_routes"])
        routes_to_try = list(route_details["fanout_routes"] or ([route] if should_ground else []))
        low_confidence = bool(route_details["low_confidence"])
        used_tools = bool(tools and should_ground)
        grounding_citations = []
        effective_messages = list(parsed_messages)
        no_hit_fallback = False
        retrieved_hits = 0
        snapshot_id = self.catalog_snapshot.get("snapshot_id")
        routes_checked: list[str] = []
        routes_with_hits: list[str] = []
        effective_max_tokens = self.default_max_tokens if max_tokens is None else max(1, int(max_tokens))
        effective_temperature = self.default_temperature if temperature is None else max(0.0, float(temperature))
        effective_top_k = self.default_top_k if top_k is None else max(0, int(top_k))
        effective_top_p = self.default_top_p if top_p is None else min(1.0, max(1e-5, float(top_p)))
        effective_repetition_penalty = self.default_repetition_penalty
        effective_no_repeat_ngram_size = self.default_no_repeat_ngram_size
        effective_decode_preset = self.decode_preset
        stop_reason = "not_reported_by_backend"
        if tools and self.rag_retriever is not None and manual_context_text is None:
            rag_result = self.rag_retriever.query(query)
            rag_hits = list(rag_result.get("hits") or [])
            rag_info.update(
                {
                    "queried": True,
                    "no_hit": bool(rag_result.get("no_hit")),
                    "retrieved_hits": len(rag_hits),
                    "hits": [self._rag_hit_metadata(hit) for hit in rag_hits],
                    "diagnostics": dict(rag_result.get("diagnostics") or {}),
                }
            )
            if rag_hits:
                rag_instruction = (
                    "Use the retrieved context below when it is relevant, but still generate a local-MVP answer. "
                    "If the context does not answer the question, say what the model can infer and avoid inventing "
                    "names, dates, policies, courses, or sources."
                )
                rag_info["instruction"] = rag_instruction
                effective_messages.insert(
                    0,
                    ChatMessage(
                        role="system",
                        content=(
                            "You are WebbGPT 0.2, a local-MVP research model.\n"
                            f"{rag_instruction}\n\n"
                            "Retrieved context:\n"
                            f"{format_rag_context(rag_hits)}"
                        ),
                    ),
                )
                used_tools = False
        elif tools and self.rag_retriever is not None and manual_context_text is not None:
            rag_info["skipped_reason"] = "manual_context_prompt"
            effective_messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=(
                        "Use only the manual context in the user prompt. If it does not answer the question, say that "
                        "the provided context does not contain enough information."
                    ),
                ),
            )
        if used_tools and self._ambiguous_timeframe(query, school_years):
            clarification = (
                "Your question mixes a current timeframe with an explicit historical year. "
                "Tell me whether you want the latest Webb snapshot or the historical snapshot "
                f"for {', '.join(school_years)}."
            )
            return AssistantResponse(
                text=clarification,
                used_tools=False,
                citations=[],
                metadata=self._base_metadata(
                    used_tools=False,
                    route=route,
                    candidate_routes=candidate_routes,
                    routes_checked=[],
                    routes_with_hits=[],
                    low_confidence=low_confidence,
                    retrieved_hits=0,
                    abstained_due_to_no_hits=False,
                    degenerate_output=False,
                    safe_decode=safe_decode,
                    effective_decode_preset=effective_decode_preset,
                    stop_reason="timeframe_clarification_required",
                    citation_labels=[],
                    snapshot_id=snapshot_id,
                    rag=rag_info,
                    manual_context=manual_context_info,
                    final_label="Abstained",
                ),
            )
        if used_tools:
            grounding_context, grounding_citations, found_hits, retrieved_hits, snapshot_id, routes_checked, routes_with_hits = self._grounding_context(
                query,
                routes=routes_to_try,
            )
            if not found_hits:
                no_hit_fallback = True
                text = self._no_hit_abstention(query, route=route)
                citation_labels: list[str] = []
                return AssistantResponse(
                    text=text,
                    used_tools=used_tools,
                    citations=[],
                    metadata=self._base_metadata(
                        used_tools=used_tools,
                        route=route,
                        candidate_routes=candidate_routes,
                        routes_checked=routes_checked,
                        routes_with_hits=routes_with_hits,
                        low_confidence=low_confidence,
                        retrieved_hits=0,
                        abstained_due_to_no_hits=True,
                        degenerate_output=False,
                        safe_decode=safe_decode,
                        effective_decode_preset=effective_decode_preset,
                        stop_reason="deterministic_no_hit_fallback",
                        citation_labels=citation_labels,
                        snapshot_id=snapshot_id,
                        rag=rag_info,
                        manual_context=manual_context_info,
                        final_label="Abstained",
                    ),
                )
            effective_messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=(
                        "You are WebbGPT 0.2, a local-MVP research model. "
                        f"{self._grounded_instruction(routes_with_hits or routes_to_try)}\n\n"
                        f"{grounding_context}"
                    ),
                ),
            )
        if safe_decode:
            effective_decode_preset = f"{self.decode_preset}-safe"
            effective_temperature = 0.0
            effective_top_k = None
            effective_top_p = 1.0
            effective_repetition_penalty = max(self.default_repetition_penalty, 1.10)
            effective_no_repeat_ngram_size = max(self.default_no_repeat_ngram_size, 4)
            effective_max_tokens = min(effective_max_tokens, 128 if used_tools else 96)

        prompt = format_chat([asdict(message) for message in effective_messages], add_generation_prompt=True)
        raw_text = self.backend.generate(
            prompt,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            top_k=effective_top_k,
            top_p=effective_top_p,
            repetition_penalty=effective_repetition_penalty,
            no_repeat_ngram_size=effective_no_repeat_ngram_size,
            stop_strings=self.default_stop_strings,
        ).strip()
        rag_context_text = " ".join(str(hit.get("text_preview") or "") for hit in rag_info.get("hits") or [])
        quality_context = manual_context_text or rag_context_text
        quality = analyze_generation(
            raw_text,
            prompt=query,
            context=quality_context,
            grounded=bool(manual_context_text or rag_info.get("retrieved_hits")),
        )
        if quality["degenerate"]:
            citation_labels = [citation.label for citation in grounding_citations[:3]]
            response_used_tools = used_tools or bool(rag_info.get("retrieved_hits"))
            metadata = self._base_metadata(
                used_tools=response_used_tools,
                route="rag" if rag_info.get("retrieved_hits") else ("manual_context" if manual_context_text else route),
                candidate_routes=candidate_routes,
                routes_checked=routes_checked,
                routes_with_hits=routes_with_hits,
                low_confidence=low_confidence,
                retrieved_hits=retrieved_hits + int(rag_info.get("retrieved_hits") or 0),
                abstained_due_to_no_hits=False,
                degenerate_output=True,
                safe_decode=safe_decode,
                effective_decode_preset=effective_decode_preset,
                stop_reason="intercepted_degenerate_output",
                citation_labels=citation_labels,
                snapshot_id=snapshot_id,
                quality=quality,
                raw_output=raw_text,
                rag=rag_info,
                manual_context=manual_context_info,
                final_label="Weak generation",
                retrieved_context_fallback=False,
            )
            return AssistantResponse(
                text=raw_text,
                used_tools=response_used_tools,
                citations=grounding_citations if citations else [],
                metadata=metadata,
            )
        text = raw_text
        if citations and grounding_citations and "[source:" not in text.lower():
            citation_labels = ", ".join(citation.label for citation in grounding_citations[:3])
            text = f"{text}\n\n[source: {citation_labels}]"
        citation_labels = [citation.label for citation in grounding_citations[:3]]
        response_used_tools = used_tools or bool(rag_info.get("retrieved_hits"))
        response_route = "rag" if rag_info.get("retrieved_hits") else ("manual_context" if manual_context_text else route)
        response_candidate_routes = ["rag"] if rag_info.get("retrieved_hits") else candidate_routes
        final_label = "Generated with sources" if rag_info.get("retrieved_hits") else "Generated"
        return AssistantResponse(
            text=text,
            used_tools=response_used_tools,
            citations=grounding_citations if citations else [],
            metadata=self._base_metadata(
                used_tools=response_used_tools,
                route=response_route,
                candidate_routes=response_candidate_routes,
                routes_checked=routes_checked,
                routes_with_hits=routes_with_hits,
                low_confidence=low_confidence,
                retrieved_hits=retrieved_hits + int(rag_info.get("retrieved_hits") or 0),
                abstained_due_to_no_hits=no_hit_fallback,
                degenerate_output=False,
                safe_decode=safe_decode,
                effective_decode_preset=effective_decode_preset,
                stop_reason=stop_reason,
                citation_labels=citation_labels,
                snapshot_id=snapshot_id,
                quality=quality,
                rag=rag_info,
                manual_context=manual_context_info,
                final_label=final_label,
            ),
        )
