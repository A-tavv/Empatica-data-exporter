from pathlib import Path
from typing import List

import pandas as pd


def _date_strs_between(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """List YYYY-MM-DD strings covering calendar days intersecting [start, end).
    End is exclusive so a full-day window (00:00 to next 00:00) includes only the start date.
    """
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
    except Exception:
        return []
    if s.tzinfo is None:
        s = s.tz_localize('UTC')
    else:
        s = s.tz_convert('UTC')
    if e.tzinfo is None:
        e = e.tz_localize('UTC')
    else:
        e = e.tz_convert('UTC')
    if e <= s:
        return []
    days = []
    cur = s.normalize()
    last = (e - pd.Timedelta(1, unit='ns')).normalize()
    while cur <= last:
        days.append(cur.strftime('%Y-%m-%d'))
        cur = cur + pd.Timedelta(days=1)
    return days


def _filter_df_by_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows within the interval [start, end] (end inclusive) using the best available time column."""
    if df is None or df.empty:
        return df
    try:
        s = pd.Timestamp(start).tz_convert('UTC') if pd.Timestamp(start).tzinfo is not None else pd.Timestamp(start).tz_localize('UTC')
    except Exception:
        s = pd.Timestamp(start)
    try:
        e = pd.Timestamp(end).tz_convert('UTC') if pd.Timestamp(end).tzinfo is not None else pd.Timestamp(end).tz_localize('UTC')
    except Exception:
        e = pd.Timestamp(end)
    col = None
    for c in ("timestamp_iso", "timestamp", "time_local", "timestamp_unix"):
        if c in df.columns:
            col = c; break
    if col is None:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df = df.copy()
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        except Exception:
            return df
    mask = (df[col] >= s) & (df[col] <= e)
    return df.loc[mask]


def _filter_df_by_time_daily_band(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filter rows so that on each day covered by [start, end], only the time band
    between start's time-of-day and end's time-of-day is kept. End boundary is inclusive.

    Special-case: when start and end have the same hour:minute and the overall duration is ~one day
    (24h, or 23/25h around DST), use a single continuous window [start, end] to avoid off-by-one-minute gaps.
    """
    if df is None or df.empty:
        return df
    col = None
    for c in ("timestamp_iso", "timestamp", "time_local", "timestamp_unix"):
        if c in df.columns:
            col = c
            break
    if col is None:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df = df.copy()
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        except Exception:
            return df
    sdt = pd.Timestamp(start)
    edt = pd.Timestamp(end)
    sdt = sdt.tz_localize('UTC') if sdt.tzinfo is None else sdt.tz_convert('UTC')
    edt = edt.tz_localize('UTC') if edt.tzinfo is None else edt.tz_convert('UTC')
    band_start_hm = (int(sdt.hour), int(sdt.minute))
    band_end_hm = (int(edt.hour), int(edt.minute))
    try:
        duration = edt - sdt
        if band_start_hm == band_end_hm and pd.Timedelta(0) < duration <= pd.Timedelta(hours=27):
            ser2 = df[col]
            mask = (ser2 >= sdt) & (ser2 <= edt)
            return df.loc[mask]
    except Exception:
        pass
    dates = _date_strs_between(sdt, edt)
    if not dates:
        return df.iloc[0:0]
    ser = df[col]
    mask_total = pd.Series(False, index=df.index)
    for dstr in dates:
        try:
            y, m, d = map(int, dstr.split('-'))
            s_day = pd.Timestamp(year=y, month=m, day=d, hour=band_start_hm[0], minute=band_start_hm[1], second=0, tz='UTC')
            e_day = pd.Timestamp(year=y, month=m, day=d, hour=band_end_hm[0], minute=band_end_hm[1], second=0, tz='UTC')
            if (band_end_hm == (0, 0)) or (e_day <= s_day):
                e_day = e_day + pd.Timedelta(days=1)
            mask_day = (ser >= s_day) & (ser <= e_day)
            mask_total = mask_total | mask_day
        except Exception:
            continue
    return df.loc[mask_total]
