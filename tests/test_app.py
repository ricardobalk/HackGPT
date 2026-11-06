"""Streamlit application smoke tests."""

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest


def test_app_shows_environment_configuration_error(monkeypatch: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app_path = Path(__file__).resolve().parents[1] / "src" / "app.py"
    app = AppTest.from_file(app_path, default_timeout=10).run()

    assert not app.exception
    assert app.selectbox[0].label == "Provider"
    assert any("OPENAI_API_KEY" in error.value for error in app.error)
