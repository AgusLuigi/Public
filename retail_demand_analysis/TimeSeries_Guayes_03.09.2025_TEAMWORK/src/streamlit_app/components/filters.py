"""
filters.py

Reusable filter components for Streamlit pages.
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
    Nutzt st.session_state, um die Auswahl beim Seitenwechsel zu merken.
    """
    col_store, col_item = st.columns(2)
    
    with col_store:
        store = int(
            st.number_input(
                "Store Nr.",
                min_value=1,
                max_value=max_store,
                value=default_store or st.session_state.get("selected_store", cfg.ui.default_store),
                step=1,
                key="input_store_nbr"
            )
        )
        st.session_state.selected_store = store # Wert global merken

    with col_item:
        item = int(
            st.number_input(
                "Item Nr.",
                min_value=1,
                value=default_item or st.session_state.get("selected_item", 1000),
                step=1,
                key="input_item_nbr"
            )
        )
        st.session_state.selected_item = item # Wert global merken
        
    return store, item

def render_pattern_filter(
    df,
    key_prefix: str,
) -> dict:
    """
    Rendert Family-, Pattern-, Perishable- und Density-Filter.
    Inklusive Sicherheits-Check für Spalten.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        options = sorted(df["family"].dropna().unique()) if "family" in df.columns else []
        fam_filter = st.multiselect(
            "Family",
            options,
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
            ["All", "Yes", "No"],
            key=f"{key_prefix}_perishable",
        )
    with col4:
        step_val = getattr(cfg.ui, "sales_density_step", 0.05)
        min_density = st.slider(
            "Min. sales_density",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=step_val,
            key=f"{key_prefix}_min_density",
        )

    return {
        "family": fam_filter,
        "pattern": pattern_filter,
        "perishable": perishable_filter,
        "min_density": min_density,
    }