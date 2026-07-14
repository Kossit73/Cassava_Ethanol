from __future__ import annotations

import base64

from streamlit_app import (
    _HERO_IMAGE_PATH,
    _hero_image_data_uri,
    _render_model_hero,
)


def test_hero_background_image_is_bundled_as_png_data_uri() -> None:
    assert _HERO_IMAGE_PATH.is_file()

    prefix, encoded = _hero_image_data_uri().split(",", maxsplit=1)

    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_hero_renders_supplied_png_as_a_background_layer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe_allow_html"] = unsafe_allow_html

    monkeypatch.setattr("streamlit_app.st.markdown", fake_markdown)

    _render_model_hero("FARM_ONLY")

    body = str(captured["body"])
    assert 'class="cassava-hero-image"' in body
    assert f'src="{_hero_image_data_uri()}"' in body
    assert body.index('class="cassava-hero-image"') < body.index(
        'class="cassava-hero-content"'
    )
    assert captured["unsafe_allow_html"] is True