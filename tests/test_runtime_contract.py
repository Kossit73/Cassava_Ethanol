from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from bioethanol_model.inputs import default_input_page

import streamlit_app as cassava_app


def _install_fake_streamlit(monkeypatch, session_state: dict | None = None) -> dict:
    fake_state = session_state or {}

    @contextmanager
    def _spinner(_message: str):
        yield

    monkeypatch.setattr(
        cassava_app,
        "st",
        SimpleNamespace(session_state=fake_state, spinner=_spinner),
    )
    return fake_state


def test_input_page_signature_changes_when_inputs_change() -> None:
    page = default_input_page()

    before = cassava_app._input_page_signature(page)
    page.projection.start_year += 1
    after = cassava_app._input_page_signature(page)

    assert before is not None
    assert after is not None
    assert before != after


def test_ensure_scenario_payload_reuses_cache_for_same_run_signature(monkeypatch) -> None:
    _install_fake_streamlit(monkeypatch, {})
    build_calls: list[str | None] = []

    class DummyModel:
        def __init__(self, input_page):
            self.input_page = input_page

        def build(self, scenario=None):
            build_calls.append(scenario)
            return {
                "input_page_snapshot": self.input_page,
                "metrics": {"Scenario": scenario or "BASE"},
            }

    monkeypatch.setattr(cassava_app, "CassavaBioethanolModel", DummyModel)
    page = default_input_page()

    cassava_app._ensure_scenario_payload("FARM_ONLY", page, run_signature="run-a")
    cassava_app._ensure_scenario_payload("FARM_ONLY", page, run_signature="run-a")
    cassava_app._ensure_scenario_payload("FARM_ONLY", page, run_signature="run-b")

    assert build_calls == ["FARM_ONLY", "FARM_ONLY"]


def test_export_cache_requires_matching_run_signature(monkeypatch) -> None:
    _install_fake_streamlit(monkeypatch, {})

    cassava_app._store_cached_export_bytes("FARM_ONLY", "sig-a", b"abc")

    assert cassava_app._get_cached_export_bytes("FARM_ONLY", "sig-a") == b"abc"
    assert cassava_app._get_cached_export_bytes("FARM_ONLY", "sig-b") is None


def test_set_state_clears_computed_runtime_state(monkeypatch) -> None:
    page = default_input_page()
    fake_state = _install_fake_streamlit(
        monkeypatch,
        {
            cassava_app.CORE_MODEL_CACHE_KEY: {"FARM_ONLY": {"results": {"npv": 1}}},
            cassava_app.EXPORT_CACHE_KEY: {"FARM_ONLY": {"run_signature": "sig-a", "bytes": b"xyz"}},
            cassava_app.MC_CACHE_KEY: {"FARM_ONLY": {"results": []}},
            cassava_app.SENSITIVITY_CACHE_KEY: {"FARM_ONLY": {"results": []}},
            cassava_app.SCENARIO_CACHE_KEY: {"FARM_ONLY": {"comparison": []}},
            "input_snapshot": page,
            cassava_app.LAST_RUN_SIGNATURE_KEY: "sig-a",
            cassava_app.MODEL_VERSION_KEY: 7,
        },
    )

    cassava_app.set_state(
        {
            "input_page": cassava_app._page_to_dict(page),
            "selected_scenario": "FARM_ONLY",
        }
    )

    assert fake_state["input_page"]["projection"]["start_year"] == page.projection.start_year
    assert fake_state["selected_scenario"] == "FARM_ONLY"
    assert cassava_app.CORE_MODEL_CACHE_KEY not in fake_state
    assert cassava_app.EXPORT_CACHE_KEY not in fake_state
    assert cassava_app.MC_CACHE_KEY not in fake_state
    assert "input_snapshot" not in fake_state
    assert fake_state[cassava_app.MODEL_VERSION_KEY] == 0
    assert fake_state["inputs_dirty"] is False
