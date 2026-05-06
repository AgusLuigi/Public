"""
test_pipeline.py

Unit-Tests für pipeline.py.
Testet alle internen Schritte sowie beide öffentlichen Pipeline-Funktionen
inklusive der Filter-Logik (Intermittent/Lumpy aus daily → weekly).

Aufruf:
    pytest test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from Favorita_TSA.utils.forecastability import (
    ADI_THRESHOLD,
    CV2_THRESHOLD,
    _aggregate_base_metrics,
    _calculate_demand_metrics,
    _calculate_lifecycle_metrics,
    _classify_demand_pattern,
    _enrich_item_metadata,
    _filter_sparse_patterns,
    run_daily_pipeline,
    run_weekly_pipeline,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def daily_df():
    """Minimales daily-DataFrame mit zwei store-item-Kombinationen."""
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 1, 1, 2, 2, 2],
            "item_nbr": [100, 100, 100, 100, 200, 200, 200],
            "date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-05",
                "2023-01-10",
                "2023-01-01",
                "2023-01-15",
                "2023-01-29",
            ],
            "unit_sales_sum": [10, 15, 20, 25, 5, 8, 12],
        }
    )


@pytest.fixture
def weekly_df():
    """Minimales weekly-DataFrame - gleiche store-item-Paare wie daily_df."""
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 1, 2, 2],
            "item_nbr": [100, 100, 100, 200, 200],
            "week": [1, 2, 3, 1, 4],
            "week_start": [
                "2023-01-02",
                "2023-01-09",
                "2023-01-16",
                "2023-01-02",
                "2023-01-23",
            ],
            "year_iso": [2023, 2023, 2023, 2023, 2023],
            "unit_sales_sum": [25, 20, 15, 5, 20],
        }
    )


@pytest.fixture
def item_meta():
    return pd.DataFrame(
        {
            "item_nbr": [100, 200],
            "family": ["GROCERY", "BEVERAGES"],
            "class": [1001, 2002],
            "perishable": [1, 0],
            "item_label": ["100", "200"],
            "emoji": ["🥫", "🥤"],
        }
    )


@pytest.fixture
def tmp_output(tmp_path):
    """Temporäres Verzeichnis für Parquet-Output."""
    return tmp_path


# =============================================================================
# _aggregate_base_metrics
# =============================================================================


class TestAggregateBaseMetrics:
    def test_row_count(self, daily_df):
        result = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        assert len(result) == 2  # 2 store-item-Kombinationen

    def test_column_names(self, daily_df):
        result = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        expected = {
            "store_nbr",
            "item_nbr",
            "first_sale",
            "last_sale",
            "periods_sold",
            "total_units",
            "mean_sales",
            "std_sales",
        }
        assert set(result.columns) == expected

    def test_values_store1_item100(self, daily_df):
        result = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        row = result[(result.store_nbr == 1) & (result.item_nbr == 100)].iloc[0]
        assert row["periods_sold"] == 4
        assert row["total_units"] == 70  # 10+15+20+25

    def test_values_store2_item200(self, daily_df):
        result = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        row = result[(result.store_nbr == 2) & (result.item_nbr == 200)].iloc[0]
        assert row["periods_sold"] == 3
        assert row["total_units"] == 25  # 5+8+12


# =============================================================================
# _calculate_lifecycle_metrics
# =============================================================================


class TestLifecycleMetrics:
    def _base(self, daily_df):
        return _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )

    def test_active_span_days(self, daily_df):
        result = _calculate_lifecycle_metrics(self._base(daily_df), "days")
        row = result[(result.store_nbr == 1) & (result.item_nbr == 100)].iloc[0]
        # 2023-01-01 bis 2023-01-10 = 10 Tage
        assert row["active_span"] == 10

    def test_active_span_weeks(self, weekly_df):
        agg = _aggregate_base_metrics(
            weekly_df.assign(week_start=pd.to_datetime(weekly_df["week_start"])),
            ["store_nbr", "item_nbr"],
            "week_start",
            "unit_sales_sum",
        )
        result = _calculate_lifecycle_metrics(agg, "weeks")
        row = result[(result.store_nbr == 1) & (result.item_nbr == 100)].iloc[0]
        # 2023-01-02 bis 2023-01-16 = 14 Tage / 7 + 1 = 3 Wochen
        assert row["active_span"] == 3

    def test_sales_density_and_zero_rate_sum_to_one(self, daily_df):
        result = _calculate_lifecycle_metrics(self._base(daily_df), "days")
        total = result["sales_density"] + result["zero_rate"]
        assert (total.round(10) == 1.0).all()

    def test_sales_density_range(self, daily_df):
        result = _calculate_lifecycle_metrics(self._base(daily_df), "days")
        assert result["sales_density"].between(0, 1).all()

    def test_invalid_span_unit_raises(self, daily_df):
        with pytest.raises(ValueError, match="span_unit"):
            _calculate_lifecycle_metrics(self._base(daily_df), "months")


# =============================================================================
# _calculate_demand_metrics
# =============================================================================


class TestDemandMetrics:
    def _pipeline_up_to_lifecycle(self, daily_df):
        agg = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        return _calculate_lifecycle_metrics(agg, "days")

    def test_adi_formula(self, daily_df):
        lc = self._pipeline_up_to_lifecycle(daily_df)
        dm = _calculate_demand_metrics(lc)
        row = dm[(dm.store_nbr == 1) & (dm.item_nbr == 100)].iloc[0]
        # ADI = active_span / periods_sold = 10 / 4 = 2.5
        assert abs(row["adi"] - 2.5) < 1e-9

    def test_cv2_formula(self, daily_df):
        lc = self._pipeline_up_to_lifecycle(daily_df)
        dm = _calculate_demand_metrics(lc)
        row = dm[(dm.store_nbr == 1) & (dm.item_nbr == 100)].iloc[0]
        expected_cv2 = (row["std_sales"] / row["mean_sales"]) ** 2
        assert abs(row["cv2"] - expected_cv2) < 1e-9

    def test_no_inf_values(self, daily_df):
        lc = self._pipeline_up_to_lifecycle(daily_df)
        dm = _calculate_demand_metrics(lc)
        assert not np.isinf(dm["adi"]).any()
        assert not np.isinf(dm["cv2"]).any()

    def test_single_observation_cv2_is_nan(self):
        """Einzelner Datenpunkt → std = NaN → cv2 = NaN (kein Crash)."""
        single = pd.DataFrame(
            {
                "store_nbr": [1],
                "item_nbr": [999],
                "date": ["2023-01-01"],
                "unit_sales_sum": [10],
            }
        )
        agg = _aggregate_base_metrics(
            single, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        lc = _calculate_lifecycle_metrics(agg, "days")
        dm = _calculate_demand_metrics(lc)
        assert pd.isna(dm.iloc[0]["cv2"])


# =============================================================================
# _classify_demand_pattern
# =============================================================================


class TestClassifyDemandPattern:
    def _make_row(self, adi, cv2):
        """Erstellt einen minimalen DataFrame mit vorgegebenen ADI/CV²-Werten."""
        return pd.DataFrame(
            {
                "store_nbr": [1],
                "item_nbr": [1],
                "active_span": [10],
                "periods_sold": [5],
                "total_units": [50],
                "mean_sales": [10.0],
                "std_sales": [np.sqrt(cv2) * 10],
                "sales_density": [0.5],
                "zero_rate": [0.5],
                "adi": [adi],
                "cv2": [cv2],
            }
        )

    def test_smooth(self):
        result = _classify_demand_pattern(
            self._make_row(ADI_THRESHOLD - 0.1, CV2_THRESHOLD - 0.1)
        )
        assert result.iloc[0]["pattern"] == "Smooth"

    def test_erratic(self):
        result = _classify_demand_pattern(
            self._make_row(ADI_THRESHOLD - 0.1, CV2_THRESHOLD + 0.1)
        )
        assert result.iloc[0]["pattern"] == "Erratic"

    def test_intermittent(self):
        result = _classify_demand_pattern(
            self._make_row(ADI_THRESHOLD + 0.1, CV2_THRESHOLD - 0.1)
        )
        assert result.iloc[0]["pattern"] == "Intermittent"

    def test_lumpy(self):
        result = _classify_demand_pattern(
            self._make_row(ADI_THRESHOLD + 0.1, CV2_THRESHOLD + 0.1)
        )
        assert result.iloc[0]["pattern"] == "Lumpy"

    def test_pattern_label_contains_emoji_and_name(self):
        df = self._make_row(ADI_THRESHOLD - 0.1, CV2_THRESHOLD - 0.1)
        result = _classify_demand_pattern(df)
        row = result.iloc[0]
        assert row["pattern"] in row["pattern_label"]
        assert row["pattern_emoji"] in row["pattern_label"]

    def test_valid_pattern_values(self, daily_df):
        agg = _aggregate_base_metrics(
            daily_df, ["store_nbr", "item_nbr"], "date", "unit_sales_sum"
        )
        lc = _calculate_lifecycle_metrics(agg, "days")
        dm = _calculate_demand_metrics(lc)
        cls = _classify_demand_pattern(dm)
        valid = {"Smooth", "Erratic", "Intermittent", "Lumpy", "Unknown"}
        assert set(cls["pattern"].unique()).issubset(valid)


# =============================================================================
# _filter_sparse_patterns
# =============================================================================


class TestFilterSparsePatterns:
    @pytest.fixture
    def daily_metrics_with_patterns(self):
        return pd.DataFrame(
            {
                "store_nbr": [1, 1, 2, 2],
                "item_nbr": [100, 101, 200, 201],
                "pattern": ["Smooth", "Intermittent", "Lumpy", "Erratic"],
            }
        )

    @pytest.fixture
    def weekly_input(self):
        return pd.DataFrame(
            {
                "store_nbr": [1, 1, 2, 2],
                "item_nbr": [100, 101, 200, 201],
                "week_start": ["2023-01-02"] * 4,
                "unit_sales_sum": [10, 20, 30, 40],
            }
        )

    def test_default_keeps_intermittent_and_lumpy(
        self, weekly_input, daily_metrics_with_patterns
    ):
        result = _filter_sparse_patterns(weekly_input, daily_metrics_with_patterns)
        # store1/item101 (Intermittent) + store2/item200 (Lumpy) bleiben
        assert len(result) == 2
        assert set(result["item_nbr"]) == {101, 200}

    def test_custom_patterns(self, weekly_input, daily_metrics_with_patterns):
        result = _filter_sparse_patterns(
            weekly_input, daily_metrics_with_patterns, patterns=["Erratic"]
        )
        assert len(result) == 1
        assert result.iloc[0]["item_nbr"] == 201

    def test_no_match_returns_empty(self, weekly_input, daily_metrics_with_patterns):
        result = _filter_sparse_patterns(
            weekly_input, daily_metrics_with_patterns, patterns=["Unknown"]
        )
        assert len(result) == 0

    def test_all_patterns_returns_full(self, weekly_input, daily_metrics_with_patterns):
        all_patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
        result = _filter_sparse_patterns(
            weekly_input, daily_metrics_with_patterns, patterns=all_patterns
        )
        assert len(result) == len(weekly_input)


# =============================================================================
# _enrich_item_metadata
# =============================================================================


class TestEnrichItemMetadata:
    def test_columns_added(self, item_meta):
        metrics = pd.DataFrame(
            {
                "store_nbr": [1, 2],
                "item_nbr": [100, 200],
                "pattern": ["Smooth", "Lumpy"],
            }
        )
        result = _enrich_item_metadata(metrics, item_meta)
        assert "family" in result.columns
        assert "perishable" in result.columns

    def test_correct_values(self, item_meta):
        metrics = pd.DataFrame(
            {"store_nbr": [1], "item_nbr": [100], "pattern": ["Smooth"]}
        )
        result = _enrich_item_metadata(metrics, item_meta)
        assert result.iloc[0]["family"] == "GROCERY"

    def test_existing_columns_not_overwritten(self, item_meta):
        metrics = pd.DataFrame(
            {
                "store_nbr": [1],
                "item_nbr": [100],
                "family": ["ALREADY_SET"],  # bereits vorhanden
            }
        )
        result = _enrich_item_metadata(metrics, item_meta)
        assert result.iloc[0]["family"] == "ALREADY_SET"

    def test_left_join_preserves_all_metrics_rows(self, item_meta):
        metrics = pd.DataFrame(
            {
                "store_nbr": [1, 2, 3],  # store 3 hat kein item_meta
                "item_nbr": [100, 200, 999],
                "pattern": ["Smooth", "Lumpy", "Unknown"],
            }
        )
        result = _enrich_item_metadata(metrics, item_meta)
        assert len(result) == 3  # kein Zeilen-Verlust durch Left-Join


# =============================================================================
# run_daily_pipeline (End-to-End)
# =============================================================================


class TestRunDailyPipeline:
    def test_output_shape(self, daily_df, tmp_output):
        result = run_daily_pipeline(
            daily_df,
            output_path=tmp_output / "daily.parquet",
        )
        assert len(result) == 2

    def test_required_columns_present(self, daily_df, tmp_output):
        result = run_daily_pipeline(
            daily_df,
            output_path=tmp_output / "daily.parquet",
        )
        for col in ["adi", "cv2", "pattern", "pattern_label", "sales_density"]:
            assert col in result.columns, f"Spalte fehlt: {col}"

    def test_parquet_written(self, daily_df, tmp_output):
        path = tmp_output / "daily.parquet"
        run_daily_pipeline(daily_df, output_path=path)
        assert path.exists()

    def test_parquet_readable(self, daily_df, tmp_output):
        path = tmp_output / "daily.parquet"
        run_daily_pipeline(daily_df, output_path=path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 2

    def test_with_item_metadata(self, daily_df, item_meta, tmp_output):
        result = run_daily_pipeline(
            daily_df,
            item_metadata=item_meta,
            output_path=tmp_output / "daily.parquet",
        )
        assert "family" in result.columns
        assert result.loc[result.item_nbr == 100, "family"].iloc[0] == "GROCERY"

    def test_no_inf_in_output(self, daily_df, tmp_output):
        result = run_daily_pipeline(
            daily_df,
            output_path=tmp_output / "daily.parquet",
        )
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols].values).any()


# =============================================================================
# run_weekly_pipeline (End-to-End)
# =============================================================================


class TestRunWeeklyPipeline:
    @pytest.fixture
    def daily_metrics_intermittent_lumpy(self):
        """Daily-Metriken wo store1/item100=Intermittent, store2/item200=Lumpy."""
        return pd.DataFrame(
            {
                "store_nbr": [1, 2],
                "item_nbr": [100, 200],
                "pattern": ["Intermittent", "Lumpy"],
                "adi": [2.5, 3.0],
                "cv2": [0.1, 0.8],
            }
        )

    def test_only_sparse_combinations_in_output(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            output_path=tmp_output / "weekly.parquet",
        )
        # Beide store-item-Paare aus daily_metrics sind Intermittent/Lumpy
        assert set(result["item_nbr"]).issubset({100, 200})

    def test_daily_pattern_column_present(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            output_path=tmp_output / "weekly.parquet",
        )
        assert "daily_pattern" in result.columns

    def test_daily_pattern_values_correct(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            output_path=tmp_output / "weekly.parquet",
        )
        row = result[(result.store_nbr == 1) & (result.item_nbr == 100)].iloc[0]
        assert row["daily_pattern"] == "Intermittent"

    def test_weekly_column_names(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            output_path=tmp_output / "weekly.parquet",
        )
        assert "active_span_weeks" in result.columns
        assert "weeks_sold" in result.columns
        assert "first_week_start" in result.columns
        assert "last_week_start" in result.columns
        # daily-Spaltennamen dürfen nicht mehr auftauchen
        assert "active_span" not in result.columns
        assert "periods_sold" not in result.columns

    def test_parquet_written(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        path = tmp_output / "weekly.parquet"
        run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            output_path=path,
        )
        assert path.exists()

    def test_smooth_items_excluded(self, weekly_df, tmp_output):
        """Items die in daily als Smooth klassifiziert sind, dürfen nicht auftauchen."""
        daily_smooth = pd.DataFrame(
            {
                "store_nbr": [1, 2],
                "item_nbr": [100, 200],
                "pattern": ["Smooth", "Smooth"],
            }
        )
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_smooth,
            output_path=tmp_output / "weekly.parquet",
        )
        assert len(result) == 0

    def test_custom_sparse_patterns(
        self, weekly_df, daily_metrics_intermittent_lumpy, tmp_output
    ):
        """Nur Lumpy übernehmen."""
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            sparse_patterns=["Lumpy"],
            output_path=tmp_output / "weekly.parquet",
        )
        assert all(result["item_nbr"] == 200)

    def test_with_item_metadata(
        self, weekly_df, daily_metrics_intermittent_lumpy, item_meta, tmp_output
    ):
        result = run_weekly_pipeline(
            weekly_df,
            daily_metrics=daily_metrics_intermittent_lumpy,
            item_metadata=item_meta,
            output_path=tmp_output / "weekly.parquet",
        )
        assert "family" in result.columns


# =============================================================================
# Ausführen
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
