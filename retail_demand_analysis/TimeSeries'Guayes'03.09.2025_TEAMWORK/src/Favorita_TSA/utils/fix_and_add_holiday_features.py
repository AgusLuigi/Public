"""
fix_and_add_holiday_features.py

Robuste Version die auch mit kaputten Indices umgeht.
"""

import pandas as pd


def fix_and_add_holiday_features(
    df,
    date_col="date",
    holiday_col="is_holiday_or_event",
    pre_days=5,
    post_days=5,
):
    """
    Repariert Index und fügt Pre/Post Holiday Features hinzu.

    Behandelt:
      - Kaputte Indices (1970-01-01 Problem)
      - Date als Spalte vs. Index
      - Multi-Index Probleme

    Parameters
    ----------
    df : pd.DataFrame
        Dein smooth_clean DataFrame
    date_col : str
        Name der Datumsspalte (falls nicht Index)
    holiday_col : str
        Name der Holiday-Spalte
    pre_days : int
        Tage VOR Holiday
    post_days : int
        Tage NACH Holiday

    Returns
    -------
    pd.DataFrame mit korrektem Index und pre_holiday/post_holiday Spalten
    """

    df = df.copy()

    print("🔍 Diagnostiziere DataFrame-Struktur...")
    print(f"   Index type: {type(df.index)}")
    print(f"   Index dtype: {df.index.dtype}")
    print(f"   Index name: {df.index.name}")
    print(f"   Columns: {df.columns.tolist()}")

    # -------------------------------------------------------------------------
    # SCHRITT 1: Finde die Date-Spalte
    # -------------------------------------------------------------------------

    # Fall 1: date_col ist im DataFrame als Spalte
    if date_col in df.columns:
        print(f"✅ Gefunden: '{date_col}' als Spalte")
        date_series = pd.to_datetime(df[date_col])
        df = df.drop(columns=[date_col])  # Entferne alte Spalte
        df.index = date_series
        df.index.name = "date"

    # Fall 2: Index ist schon date, aber kaputt
    elif df.index.name == date_col or df.index.name == "date":
        print(f"⚠️  Index ist '{df.index.name}' aber kaputt, repariere...")
        # Reset und neu parsen
        df = df.reset_index()
        date_series = pd.to_datetime(df[date_col if date_col in df.columns else "date"])
        df = df.drop(columns=[date_col if date_col in df.columns else "date"])
        df.index = date_series
        df.index.name = "date"

    # Fall 3: Index ist bereits korrekt
    elif isinstance(df.index, pd.DatetimeIndex):
        print("✅ Index ist bereits DatetimeIndex")
        df.index.name = "date"

    # Fall 4: Keine Ahnung wo date ist
    else:
        print("❌ Kann date-Spalte nicht finden!")
        print(f"   Verfügbare Spalten: {df.columns.tolist()}")
        raise ValueError(
            f"Spalte '{date_col}' nicht gefunden und Index ist kein DatetimeIndex"
        )

    # -------------------------------------------------------------------------
    # SCHRITT 2: Sortiere nach Datum
    # -------------------------------------------------------------------------

    df = df.sort_index()

    print("\n✅ Index repariert:")
    print(f"   Von: {df.index.min()}")
    print(f"   Bis: {df.index.max()}")
    print(f"   Länge: {len(df)}")

    # -------------------------------------------------------------------------
    # SCHRITT 3: Finde Holidays
    # -------------------------------------------------------------------------

    if holiday_col not in df.columns:
        raise ValueError(f"Holiday-Spalte '{holiday_col}' nicht gefunden!")

    # Konvertiere zu Boolean falls nötig
    df[holiday_col] = df[holiday_col].astype(bool)

    holiday_dates = df[df[holiday_col]].index

    print(f"\n🎉 Gefunden: {len(holiday_dates)} Holidays")
    if len(holiday_dates) > 0:
        print(f"   Erste 5: {holiday_dates[:5].tolist()}")

    # -------------------------------------------------------------------------
    # SCHRITT 4: Erstelle Pre/Post Features
    # -------------------------------------------------------------------------

    print(
        f"\n⚙️  Erstelle Pre-Holiday ({pre_days} Tage) und Post-Holiday ({post_days} Tage)..."
    )

    df["pre_holiday"] = False
    df["post_holiday"] = False

    for holiday in holiday_dates:
        # Pre-Holiday: 1-N Tage VOR Holiday
        pre_start = holiday - pd.Timedelta(days=pre_days)
        pre_end = holiday - pd.Timedelta(days=1)

        # Post-Holiday: 1-N Tage NACH Holiday
        post_start = holiday + pd.Timedelta(days=1)
        post_end = holiday + pd.Timedelta(days=post_days)

        # Markiere
        df.loc[pre_start:pre_end, "pre_holiday"] = True
        df.loc[post_start:post_end, "post_holiday"] = True

    # -------------------------------------------------------------------------
    # SCHRITT 5: Statistik
    # -------------------------------------------------------------------------

    n_holidays = df[holiday_col].sum()
    n_pre = df["pre_holiday"].sum()
    n_post = df["post_holiday"].sum()
    n_regular = len(df) - n_holidays - n_pre - n_post

    print("\n📊 Ergebnis:")
    print(f"   Holidays:     {n_holidays:,} Tage ({n_holidays / len(df) * 100:.1f}%)")
    print(f"   Pre-Holiday:  {n_pre:,} Tage ({n_pre / len(df) * 100:.1f}%)")
    print(f"   Post-Holiday: {n_post:,} Tage ({n_post / len(df) * 100:.1f}%)")
    print(f"   Regular:      {n_regular:,} Tage ({n_regular / len(df) * 100:.1f}%)")

    return df
