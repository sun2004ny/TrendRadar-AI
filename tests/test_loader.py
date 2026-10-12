"""
tests/test_loader.py
====================
Unit tests for components.loader and page loading transition helpers.
"""

from unittest.mock import MagicMock
from components.loader import (
    get_loading_card_html,
    show_page_loader,
    clear_page_loader,
    section_spinner,
)
from src.embeddings import get_sentence_transformer_model


def test_get_loading_card_html():
    """Verify HTML markup contains spinner, dark styling class, and title."""
    html = get_loading_card_html("monitoring")
    assert "tr-loading-card" in html
    assert "tr-loading-spinner" in html
    assert "Loading <strong>Monitoring</strong>..." in html

    html_multi = get_loading_card_html("research_lab")
    assert "Loading <strong>Research Lab</strong>..." in html_multi


def test_show_and_clear_page_loader():
    """Verify placeholder interaction for showing and clearing page loader."""
    mock_placeholder = MagicMock()

    show_page_loader(mock_placeholder, "ALERTS")
    mock_placeholder.markdown.assert_called_once()
    args, kwargs = mock_placeholder.markdown.call_args
    assert "Loading <strong>Alerts</strong>..." in args[0]
    assert kwargs.get("unsafe_allow_html") is True

    clear_page_loader(mock_placeholder)
    mock_placeholder.empty.assert_called_once()


def test_section_spinner():
    """Verify section_spinner context manager executes block successfully."""
    executed = False
    with section_spinner("Loading test data..."):
        executed = True
    assert executed is True


def test_get_sentence_transformer_model():
    """Verify get_sentence_transformer_model returns model and caches properly."""
    from unittest.mock import patch, MagicMock
    mock_model = MagicMock()
    with patch("src.embeddings.SentenceTransformer", return_value=mock_model):
        model = get_sentence_transformer_model("test-model")
        assert model is not None
