"""
components/loader.py
====================
Unified dark-glassmorphic page transition and section loader component for TrendRadar AI.
Provides immediate visual feedback when navigating between intelligence views.
"""

import streamlit as st
from contextlib import contextmanager


def get_loading_card_html(page_name: str) -> str:
    """
    Returns HTML markup for a small full-width dark-glass loading card
    matching the TrendRadar AI theme.
    """
    clean_name = page_name.replace("_", " ").title() if "_" in page_name else page_name.title()
    return f"""
    <div class="tr-loading-card" id="tr-page-loading-indicator">
        <div class="tr-loading-spinner"></div>
        <div class="tr-loading-text">
            <span>Loading <strong>{clean_name}</strong>...</span>
        </div>
    </div>
    """


def show_page_loader(placeholder, page_name: str) -> None:
    """
    Fills a Streamlit placeholder with a sleek, full-width loading card.
    """
    if placeholder:
        placeholder.markdown(get_loading_card_html(page_name), unsafe_allow_html=True)


def clear_page_loader(placeholder) -> None:
    """
    Clears the loading card from the placeholder once page rendering completes.
    """
    if placeholder:
        placeholder.empty()


@contextmanager
def section_spinner(message: str = "Loading section..."):
    """
    Context manager wrapping heavy computational or data loading parts
    in a dark-themed spinner block.
    """
    with st.spinner(message):
        yield
