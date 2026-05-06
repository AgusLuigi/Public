"""
test_preprocess_data.py

Tests für src/Favorita_TSA/utils/preprocess_data.py
Fokus: Aggregations-Factory (_make_aggregator) und fix_agg_columns.
"""

import pandas as pd
import pytest

from Favorita_TSA.utils.preprocess_data import (
    agg_daily,
    agg_monthly,
    agg_weekly,
    fix_agg_columns,
    item_daily,
    item_monthly,
    item_weekly,
    store_daily,
    store_item_daily,
    store_item_monthly,
    store_item_weekly,
    store_monthly,
    store_weekly,
)


@pytest.fixture
def fact_df():
    """Minimale Fact-Table für Aggregations-Tests."""
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 2],
            "item_nbr": [100, 100, 200, 200],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]
            ),
            "week": [1, 1, 1, 1],
            "week_start": pd.to_datetime(["2023-01-02"] * 4),
            "year_iso": [2023, 2023, 2023, 2023],
            "month": pd.to_datetime(["2023-01-01"] * 4),
            "unit_sales": [10.0, 20.0, 5.0, 15.0],
        }
    )


class TestAggregationsFactory:
    """Stellt sicher dass alle Factory-Aggregatoren korrekt funktionieren."""

    def test_agg_daily_groups_by_date(self, fact_df):
        result = agg_daily(fact_df)
        assert "date" in result.columns
        assert len(result) == 2  # 2 verschiedene Daten

    def test_agg_weekly_groups_by_week(self, fact_df):
        result = agg_weekly(fact_df)
        assert "week" in result.columns
        assert len(result) == 1  # alle in Woche 1

    def test_agg_monthly_groups_by_month(self, fact_df):
        result = agg_monthly(fact_df)
        assert "month" in result.columns
        assert len(result) == 1

    def test_store_daily_groups_by_store_and_date(self, fact_df):
        result = store_daily(fact_df)
        assert "store_nbr" in result.columns
        assert "date" in result.columns
        assert len(result) == 4  # 2 stores x 2 dates

    def test_item_daily_groups_by_item_and_date(self, fact_df):
        result = item_daily(fact_df)
        assert "item_nbr" in result.columns
        assert "date" in result.columns

    def test_store_item_daily_groups_by_store_item_date(self, fact_df):
        result = store_item_daily(fact_df)
        assert "store_nbr" in result.columns
        assert "item_nbr" in result.columns
        assert "date" in result.columns
        assert len(result) == 4

    def test_store_weekly(self, fact_df):
        result = store_weekly(fact_df)
        assert "store_nbr" in result.columns
        assert "week" in result.columns

    def test_item_weekly(self, fact_df):
        result = item_weekly(fact_df)
        assert "item_nbr" in result.columns
        assert "week" in result.columns

    def test_store_item_weekly_has_all_keys(self, fact_df):
        result = store_item_weekly(fact_df)
        for col in ("store_nbr", "item_nbr", "year_iso", "week", "week_start"):
            assert col in result.columns

    def test_store_monthly(self, fact_df):
        result = store_monthly(fact_df)
        assert "store_nbr" in result.columns
        assert "month" in result.columns

    def test_item_monthly(self, fact_df):
        result = item_monthly(fact_df)
        assert "item_nbr" in result.columns

    def test_store_item_monthly(self, fact_df):
        result = store_item_monthly(fact_df)
        assert "store_nbr" in result.columns
        assert "item_nbr" in result.columns
        assert "month" in result.columns

    def test_all_aggregators_return_dataframe(self, fact_df):
        aggregators = [
            agg_daily,
            agg_weekly,
            agg_monthly,
            store_daily,
            item_daily,
            store_item_daily,
            store_weekly,
            item_weekly,
            store_item_weekly,
            store_monthly,
            item_monthly,
            store_item_monthly,
        ]
        for fn in aggregators:
            result = fn(fact_df)
            assert isinstance(
                result, pd.DataFrame
            ), f"{fn.__name__} returned no DataFrame"

    def test_unit_sales_sum_column_present(self, fact_df):
        """Nach fix_agg_columns muss unit_sales_sum vorhanden sein."""
        from Favorita_TSA.utils.preprocess_data import ALL_METRICS_UNIT_SALES, aggregate

        result = aggregate(fact_df, ["date"], ALL_METRICS_UNIT_SALES)
        fixed = fix_agg_columns(result)
        assert "unit_sales_sum" in fixed.columns


class TestFixAggColumns:
    def test_tuple_columns_flattened(self):
        df = pd.DataFrame([[1, 2]], columns=[("unit_sales", "sum"), ("store_nbr", "")])
        result = fix_agg_columns(df)
        assert "unit_sales_sum" in result.columns
        assert "store_nbr" in result.columns

    def test_string_columns_unchanged(self):
        df = pd.DataFrame([[1]], columns=["date"])
        result = fix_agg_columns(df)
        assert "date" in result.columns

    def test_returns_copy(self):
        df = pd.DataFrame([[1]], columns=["date"])
        result = fix_agg_columns(df)
        assert id(result) != id(df)
