"""
test_date_utils.py

Tests für src/Favorita_TSA/utils/date_utils.py
"""

import pandas as pd
import pytest

from Favorita_TSA.utils.date_utils import (
    get_date_col,
    normalize_time_col,
    period_to_timestamp,
)


class TestPeriodToTimestamp:
    def test_period_series_converted(self):
        s = pd.Series(pd.period_range("2023-01", periods=3, freq="M"))
        result = period_to_timestamp(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_string_series_converted(self):
        s = pd.Series(["2023-01-01", "2023-01-08", "2023-01-15"])
        result = period_to_timestamp(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_datetime_series_unchanged_type(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=3))
        result = period_to_timestamp(s)
        assert pd.api.types.is_datetime64_any_dtype(result)

    def test_period_start_time_correct(self):
        s = pd.Series(pd.period_range("2023-01-02", periods=1, freq="W"))
        result = period_to_timestamp(s)
        assert result.iloc[0] == pd.Timestamp("2023-01-02")


class TestNormalizeTimeCol:
    def test_string_col_becomes_datetime(self):
        df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
        result = normalize_time_col(df, "date")
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_period_col_becomes_datetime(self):
        df = pd.DataFrame({"week": pd.period_range("2023-01-02", periods=3, freq="W")})
        result = normalize_time_col(df, "week")
        assert pd.api.types.is_datetime64_any_dtype(result["week"])

    def test_invalid_dates_dropped(self):
        df = pd.DataFrame({"date": ["2023-01-01", "not-a-date", "2023-01-03"]})
        result = normalize_time_col(df, "date")
        assert len(result) == 2

    def test_missing_col_returns_original(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = normalize_time_col(df, "date")
        assert list(result.columns) == ["other"]
        assert len(result) == 3

    def test_returns_copy_not_inplace(self):
        df = pd.DataFrame({"date": ["2023-01-01"]})
        result = normalize_time_col(df, "date")
        assert id(result) != id(df)


class TestGetDateCol:
    def test_finds_date(self):
        df = pd.DataFrame({"date": [], "value": []})
        assert get_date_col(df) == "date"

    def test_finds_week_start(self):
        df = pd.DataFrame({"week_start": [], "value": []})
        assert get_date_col(df) == "week_start"

    def test_finds_week(self):
        df = pd.DataFrame({"week": [], "value": []})
        assert get_date_col(df) == "week"

    def test_finds_month(self):
        df = pd.DataFrame({"month": [], "value": []})
        assert get_date_col(df) == "month"

    def test_prefers_date_over_week(self):
        df = pd.DataFrame({"date": [], "week": []})
        assert get_date_col(df) == "date"

    def test_raises_if_none_found(self):
        df = pd.DataFrame({"value": [], "store_nbr": []})
        with pytest.raises(ValueError, match="Keine bekannte Zeit-Spalte"):
            get_date_col(df)
