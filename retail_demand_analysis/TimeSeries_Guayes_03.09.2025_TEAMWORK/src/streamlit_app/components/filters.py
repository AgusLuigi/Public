"""
filters.py

Wiederverwendbare Filter-Widgets für Streamlit-Pages.
"""

from __future__ import annotations

import streamlit as st

from Favorita_TSA.utils.config import cfg


def render_store_item_selector(
    default_store: int | None = None,
    default_item: int | None = None,
    max_store: int = 54,
) -> tuple[int, int]:
    """
    Rendert Store- und Item-Nummern-Eingabefelder.
    Gibt (store, item) zurück.
    """
    col_store, col_item = st.columns(2)
    with col_store:
        store = int(
            st.number_input(
                "Store Nr.",
                min_value=1,
                max_value=max_store,
                value=default_store or cfg.ui.default_store,
                step=1,
            )
        )
    with col_item:
        item = int(
            st.number_input(
                "Item Nr.",
                min_value=1,
                value=default_item or 1000,
                step=1,
            )
        )
    return store, item


def render_pattern_filter(
    df,
    key_prefix: str,
) -> dict:
    """
    Rendert Family-, Pattern-, Perishable- und Density-Filter.
    Gibt die gewählten Filterwerte als dict zurück.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fam_filter = st.multiselect(
            "Family",
            sorted(df["family"].dropna().unique()),
            key=f"{key_prefix}_family",
        )
    with col2:
        pattern_filter = st.multiselect(
            "Demand Type",
            ["Smooth", "Erratic", "Intermittent", "Lumpy"],
            key=f"{key_prefix}_pattern",
        )
    with col3:
        perishable_filter = st.selectbox(
            "Perishable",
            ["All", True, False],
            key=f"{key_prefix}_perishable",
        )
    with col4:
        min_density = st.slider(
            "Min. sales_density",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=cfg.ui.sales_density_step,
            key=f"{key_prefix}_min_density",
        )

    return {
        "family": fam_filter,
        "pattern": pattern_filter,
        "perishable": perishable_filter,
        "min_density": min_density,
    }
