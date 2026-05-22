from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest
from fastapi.testclient import TestClient

from config import DataConfig, DataSourceConfig, GroundingConfig, ServeConfig
from data.dataset import DatasetBuilder
from eval.assistant import _score_response
from grounding.ingest import webb_sync
from grounding.types import Citation
from posttrain.eval import update_topk_candidates
from provenance import benchmark_manifest, reliability_payload
from serve.app import build_app
from serve.backends.transformers_backend import TransformersChatBackend
from serve.quality import analyze_generation
from serve.orchestrator import AssistantOrchestrator
from serve.playground import render_playground_html
from serve.types import ChatMessage


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return str(path)


class _DummyBackend:
    def generate(self, *args, **kwargs):
        return "stubbed answer"


class _DegenerateBackend:
    def generate(self, *args, **kwargs):
        return ",,,, a,,, and,,, not,,, the,,, of,,, is,,, or,,,-,,, you,,, an,,,s,,,."


class _NoHitProvider:
    def query(self, text):
        class Result:
            hits = []

        return Result()


def test_orchestrator_uses_deterministic_no_hit_fallback():
    orchestrator = AssistantOrchestrator(
        _DummyBackend(),
        grounding_provider=_NoHitProvider(),
        catalog_snapshot={"snapshot_id": "demo"},
    )
    reply = orchestrator.respond([ChatMessage(role="user", content="What are the prerequisites for BIO 404?")])
    assert "could not find a matching catalog entry" in reply.text.lower()
    assert reply.metadata["abstained_due_to_no_hits"] is True


def test_orchestrator_retains_should_ground_compatibility():
    orchestrator = AssistantOrchestrator(
        _DummyBackend(),
        grounding_provider=_NoHitProvider(),
        catalog_snapshot={"snapshot_id": "demo"},
    )
    should_ground = orchestrator._should_ground(
        [ChatMessage(role="user", content="What are the prerequisites for BIO 404?")]
    )
    assert should_ground is True


def test_orchestrator_intercepts_degenerate_output():
    orchestrator = AssistantOrchestrator(_DegenerateBackend(), catalog_snapshot={"snapshot_id": "demo"})
    reply = orchestrator.respond([ChatMessage(role="user", content="Hi there")])
    assert reply.text == ",,,, a,,, and,,, not,,, the,,, of,,, is,,, or,,,-,,, you,,, an,,,s,,,."
    assert reply.metadata["status"]["final_label"] == "Weak generation"
    assert reply.metadata["status"]["answered"] is True
    assert reply.metadata["status"]["degenerate_output"] is True
    assert reply.metadata["debug"]["raw_output"]


def test_chat_endpoint_serializes_slotted_citations(monkeypatch: pytest.MonkeyPatch):
    class _Backend:
        backend_name = "dummy"

        def __init__(self, *_args, **_kwargs):
            pass

    class _Orchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def respond(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="Grounded answer",
                used_tools=True,
                citations=[
                    Citation(
                        source_type="mock",
                        source_id="src-1",
                        label="Demo Source",
                        snippet="Snippet text",
                        metadata={"page": 1},
                    )
                ],
                metadata={},
            )

    monkeypatch.setattr("serve.app.VLLMChatBackend", _Backend)
    monkeypatch.setattr("serve.app.TransformersChatBackend", _Backend)
    monkeypatch.setattr("serve.app.AssistantOrchestrator", _Orchestrator)
    monkeypatch.setattr("serve.app.seed_everything", lambda _seed: {"python": 52, "numpy": 52, "torch": 52})

    client = TestClient(build_app(ServeConfig(enable_grounding=False)))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is Webb?"}],
            "tools": True,
            "citations": True,
            "safe_decode": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["used_tools"] is True
    assert payload["citations"] == [
        {
            "source_type": "mock",
            "source_id": "src-1",
            "label": "Demo Source",
            "snippet": "Snippet text",
            "metadata": {"page": 1},
        }
    ]


def test_generate_alias_wraps_chat_completion(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Backend:
        backend_name = "dummy"
        device = "cpu"

        def __init__(self, *_args, **_kwargs):
            pass

    class _Orchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def respond(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(text="Alias answer", used_tools=False, citations=[], metadata={})

    monkeypatch.setattr("serve.app.VLLMChatBackend", _Backend)
    monkeypatch.setattr("serve.app.TransformersChatBackend", _Backend)
    monkeypatch.setattr("serve.app.AssistantOrchestrator", _Orchestrator)
    monkeypatch.setattr("serve.app.seed_everything", lambda _seed: {"python": 52, "numpy": 52, "torch": 52})

    client = TestClient(build_app(ServeConfig(enable_grounding=False)))
    status = client.get("/status").json()
    response = client.post(
        "/generate",
        json={
            "prompt": "At Webb, students often...",
            "tools": False,
            "citations": False,
            "max_new_tokens": 12,
            "temperature": 0.4,
            "top_k": 40,
            "top_p": 0.95,
        },
    )

    assert "/generate" in status["endpoints"]
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Alias answer"
    assert payload["metadata"]["api"] == {"route": "/generate", "canonical": "/v1/chat/completions"}
    assert captured["messages"] == [{"role": "user", "content": "At Webb, students often..."}]
    assert captured["kwargs"]["tools"] is False
    assert captured["kwargs"]["citations"] is False
    assert captured["kwargs"]["max_tokens"] == 12
    assert captured["kwargs"]["temperature"] == pytest.approx(0.4)
    assert captured["kwargs"]["top_k"] == 40
    assert captured["kwargs"]["top_p"] == pytest.approx(0.95)


def test_playground_uses_webbgpt_02_labels_and_final_prompt_chips():
    html = render_playground_html(ServeConfig(use_rag=True))
    expected_order = [
        "hi WebbGPT 0.2, how are you?",
        "What is the difference between a prerequisite and a recommendation?",
        "What does a course catalog help students understand?",
        "What is the phone policy in the dining hall?",
        "A course catalog helps students",
        "During a science project, the first step is",
    ]

    assert "<title>WebbGPT 0.2 Chat</title>" in html
    assert "<strong>WebbGPT 0.2</strong>" in html
    assert "Use RAG" in html
    assert "Show sources" in html
    assert "Sources available" in html
    assert "What is the Hogwarts dining policy?" not in html
    assert "hi im harry potter" not in html
    positions = [html.index(prompt) for prompt in expected_order]
    assert positions == sorted(positions)


def test_generate_stream_progressively_reveals_final_response(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Backend:
        backend_name = "dummy"
        device = "cpu"

        def __init__(self, *_args, **_kwargs):
            pass

    class _Orchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def respond(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                text="Streamed final answer with sources.",
                used_tools=True,
                citations=[],
                metadata={"status": {"final_label": "Generated with sources", "retrieved_context_fallback": False}},
            )

    monkeypatch.setattr("serve.app.VLLMChatBackend", _Backend)
    monkeypatch.setattr("serve.app.TransformersChatBackend", _Backend)
    monkeypatch.setattr("serve.app.AssistantOrchestrator", _Orchestrator)
    monkeypatch.setattr("serve.app.seed_everything", lambda _seed: {"python": 52, "numpy": 52, "torch": 52})

    client = TestClient(build_app(ServeConfig(enable_grounding=False)))
    status = client.get("/status").json()
    response = client.post(
        "/generate_stream",
        json={
            "prompt": "What does the catalog say about prerequisites?",
            "tools": True,
            "citations": True,
            "max_new_tokens": 32,
        },
    )

    assert "/generate_stream" in status["endpoints"]
    assert response.status_code == 200
    body = response.text
    assert "event: start" in body
    assert "event: delta" in body
    assert "Streamed final" in body
    assert "answer with" in body
    assert "event: metadata" in body
    assert "ui_progressive_rendering" in body
    assert "true_model_token_streaming" in body
    assert "event: done" in body
    assert captured["messages"] == [{"role": "user", "content": "What does the catalog say about prerequisites?"}]
    assert captured["kwargs"]["tools"] is True


def test_benchmark_manifest_and_reliability_report_counts(tmp_path: Path):
    benchmark_path = tmp_path / "assistant.jsonl"
    _write_jsonl(benchmark_path, [{"messages": [{"role": "user", "content": "Hello"}]} for _ in range(3)])
    manifest = benchmark_manifest([str(benchmark_path)])
    assert manifest["entries"][0]["examples"] == 3
    reliability = reliability_payload(3)
    assert reliability["per_example_swing"] == pytest.approx(1 / 3)
    assert reliability["warning"] is not None


def test_grouped_autosplit_can_be_fail_closed_when_explicit_validation_is_required(tmp_path: Path):
    sft_path = _write_jsonl(
        tmp_path / "sft.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"Answer {index}."},
                ]
            }
            for index in range(6)
        ],
    )
    config = DataConfig(tokenizer_path="artifacts/tokenizer/webbgpt-local-mvp.model", sequence_length=32)
    config.sft_sources = [
        DataSourceConfig(
            name="sft",
            path=sft_path,
            format="jsonl",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    with pytest.raises(RuntimeError, match="requires explicit sft_validation_sources"):
        builder.build_sft_split(
            seed=52,
            validation_fraction=0.25,
            validation_min_examples=2,
            allow_weak_validation=False,
            require_explicit_validation=True,
        )


def test_generic_filler_is_penalized_in_assistant_scoring():
    score, exact, has_citation = _score_response(
        {"expected_substrings": ["prerequisite"]},
        "I am doing well and ready to help. What would you like to work on today?",
        require_citations=False,
    )
    assert score == 0.0
    assert exact is False
    assert has_citation is False


def test_topk_candidate_metadata_prunes_stale_candidate_dirs(tmp_path: Path):
    output_dir = tmp_path / "checkpoints"
    candidate_paths = []
    for step, value in [(10, 0.8), (20, 0.7), (30, 0.6), (40, 0.5)]:
        candidate_dir = output_dir / f"candidate-step-{step:08d}"
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "checkpoint.pt").write_text("stub")
        candidate_paths.append(candidate_dir)
        update_topk_candidates(
            output_dir,
            candidate_path=candidate_dir,
            candidate_payload={"selection_value": value, "step": step, "metrics": {"loss": value}},
            metric_key="loss",
            limit=3,
            lower_is_better=True,
        )
    assert not candidate_paths[0].exists()
    assert candidate_paths[-1].exists()
    topk_payload = json.loads((output_dir / "topk.json").read_text())
    assert len(topk_payload) == 3


def test_generation_quality_detector_flags_separator_spam():
    analysis = analyze_generation(
        ",,,, a,,, and,,, not,,, the,,, of,,, is,,, or,,,-,,, you,,, an,,,s,,,."
    )
    assert analysis["degenerate"] is True
    assert analysis["reasons"]


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    def no_grad(self):
        return _FakeNoGrad()


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        return _FakeBatch({"input_ids": np.array([[10, 11]])})

    def convert_tokens_to_ids(self, token: str):
        if token == "</s>":
            return self.eos_token_id
        return self.unk_token_id

    def decode(self, generated, skip_special_tokens: bool = True):
        return "hello </s>"


class _FakeModel:
    def __init__(self):
        self.kwargs = None
        self.generation_config = SimpleNamespace(do_sample=False, temperature=0.0, top_p=1.0)

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.array([[10, 11, 20, 21]])


def test_transformers_backend_passes_tokenizer_when_generating_with_stop_strings():
    backend = TransformersChatBackend.__new__(TransformersChatBackend)
    backend._torch = _FakeTorch()
    backend.tokenizer = _FakeTokenizer()
    backend.model = _FakeModel()
    backend.device = "cpu"
    backend.backend_name = "transformers"
    backend.seed = 52

    text = backend.generate(
        "hello",
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
        stop_strings=["</s>"],
    )

    assert backend.model.kwargs is not None
    assert backend.model.kwargs["tokenizer"] is backend.tokenizer
    assert "temperature" not in backend.model.kwargs
    assert text == "hello"


def test_build_app_falls_back_to_latest_completed_snapshot_when_sync_on_start_fails(
    monkeypatch,
    tmp_path: Path,
):
    dsn = f"sqlite:///{tmp_path / 'webb.db'}"
    webb_sync(
        dsn,
        seed_url_pack="data/webb/seed_urls_demo.json",
        source_policy_path="data/webb/source_policies.json",
        handbook_url="data/webb/mock/handbook.txt",
        label="baseline",
    )

    class _Backend:
        backend_name = "dummy"

        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, *args, **kwargs):
            return "stubbed answer"

    monkeypatch.setattr("serve.app.VLLMChatBackend", _Backend)
    monkeypatch.setattr("serve.app.TransformersChatBackend", _Backend)
    monkeypatch.setattr("serve.app.webb_sync", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")))
    monkeypatch.setattr("serve.app.seed_everything", lambda _seed: {"python": 52, "numpy": 52, "torch": 52})

    config = ServeConfig(
        grounding=GroundingConfig(
            dsn=dsn,
            seed_url_pack="data/webb/seed_urls_demo.json",
            source_policy_path="data/webb/source_policies.json",
            handbook_url="data/webb/mock/handbook.txt",
            sync_on_start=True,
        )
    )
    app = build_app(config)
    assert app is not None
