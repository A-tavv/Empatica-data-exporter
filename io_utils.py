import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_unique_folder(path: Path) -> Path:
    """Return a unique folder path based on 'path'. Ensure directory exists.
    If the folder exists, append one or more '_' until a free name is found.
    """
    try:
        base = Path(path)
        p = base
        while p.exists():
            p = Path(str(p) + "_")
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)


def load_csv_files(root: Path, date_str: str, device_id: str) -> List[Path]:
    """List aggregated-per-minute CSV files for a device on a given date."""
    base = root / date_str / device_id / "digital_biomarkers" / "aggregated_per_minute"
    return sorted(base.glob("*.csv")) if base.exists() else []


def find_avro_files(root: Path, date_str: str, device_id: str) -> List[Path]:
    """Return all .avro files for a device on a date.
    Prefer <root>/<date>/<device>/raw_data/**/*.avro; otherwise search the device subtree.
    """
    base_dev = root / date_str / device_id
    raw_base = base_dev / "raw_data"
    if raw_base.exists():
        return sorted(raw_base.rglob("*.avro"))
    return sorted(base_dev.rglob("*.avro")) if base_dev.exists() else []


# ---------------- Aggregated naming helpers ----------------

def _aggregated_measurement_key(stem: str) -> str:
    """Extract a stable measurement key from an aggregated CSV stem.
    Expected pattern: <device>_<YYYY-MM-DD>_<measurement>.
    Returns <measurement>. Falls back to the substring after the last underscore.
    """
    try:
        m = re.match(r"^.+?_(\d{4}-\d{2}-\d{2})_(.+)$", stem)
        if m:
            return m.group(2)
        if "_" in stem:
            return stem.rsplit("_", 1)[-1]
        return stem
    except Exception:
        return stem


def _device_code_from_dir(name: str) -> str:
    """Extract the trailing device code from a device directory name, e.g., 'HTI01-3YK...' -> '3YK...'"""
    try:
        return name.split('-')[-1] if name else name
    except Exception:
        return name


# ---------------- Save helpers (NPZ/MAT/CSV) ----------------

def _df_to_arrays_with_meta(merged: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Return arrays dict (excluding timestamp_iso) and datetime64 array for timestamp_iso when present."""
    ts_arr: Optional[np.ndarray] = None
    df = merged.copy()
    if "timestamp_iso" in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp_iso"]):
                df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], utc=True, errors='coerce')
            df = df.sort_values(by="timestamp_iso")
            ts_arr = np.asarray(df["timestamp_iso"].astype('datetime64[ns]').values)
        except Exception:
            ts_arr = None
    arrays: Dict[str, np.ndarray] = {}
    for c in df.columns:
        if c == "timestamp_iso":
            continue
        try:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy()
        except Exception:
            arr = df[c].to_numpy()
        arrays[c] = arr
    return arrays, ts_arr


def _write_df(df: pd.DataFrame, out_path: Path, fmt: str = 'csv') -> Path:
    fmt_l = (fmt or 'csv').strip().lower()
    out_path = Path(out_path)
    if fmt_l in ('parquet', 'pq'):
        outf = out_path.with_suffix('.parquet')
        df.to_parquet(outf, index=False)
        return outf
    else:
        outf = out_path.with_suffix('.csv')
        to_write = df.copy()
        try:
            if 'timestamp_iso' in to_write.columns:
                ser = to_write['timestamp_iso']
                if not pd.api.types.is_datetime64_any_dtype(ser):
                    ser = pd.to_datetime(ser, errors='coerce')
                iso_str = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
                iso_str = iso_str.where(ser.notna(), to_write['timestamp_iso'].astype(str))
                to_write['timestamp_iso'] = "'" + iso_str.fillna("")
        except Exception:
            pass
        to_write.to_csv(outf, index=False)
        return outf


def _save_sensor_npz(raw_dir: Path, sensor: str, merged: pd.DataFrame, meta: Dict[str, Any]):
    """Save a per-sensor NPZ with arrays and JSON meta."""
    arrays, ts_arr = _df_to_arrays_with_meta(merged)
    payload: Dict[str, Any] = {k: v for k, v in arrays.items()}
    if ts_arr is not None:
        try:
            ts_str = np.array([pd.Timestamp(x).isoformat() for x in ts_arr], dtype=object)
        except Exception:
            ts_str = np.array([], dtype=object)
        payload["timestamp_iso"] = ts_str
    payload["__meta"] = json.dumps(meta)
    np.savez_compressed(raw_dir / f"{sensor}.npz", **payload)


def _save_sensor_mat(raw_dir: Path, sensor: str, merged: pd.DataFrame, meta: Dict[str, Any]):
    """Save a per-sensor MATLAB .mat file containing arrays and JSON meta."""
    try:
        from scipy.io import savemat  # lazy import
    except Exception as e:
        raise RuntimeError("scipy is required for MAT output; install it via pip/conda") from e
    arrays, ts_arr = _df_to_arrays_with_meta(merged)
    mdict: Dict[str, Any] = {k: v for k, v in arrays.items()}
    if ts_arr is not None:
        try:
            ts_str = np.array([pd.Timestamp(x).isoformat() for x in ts_arr], dtype=object)
        except Exception:
            ts_str = np.array([], dtype=object)
        mdict["timestamp_iso"] = ts_str
    mdict["__meta"] = np.array(json.dumps(meta), dtype=object)
    outp = raw_dir / f"{sensor}.mat"
    from scipy.io import savemat
    savemat(outp, {sensor: mdict}, do_compression=True)
    return outp


def _save_sensor_csv(raw_dir: Path, sensor: str, merged: pd.DataFrame, meta: Dict[str, Any]):
    """Save a per-sensor CSV with timestamp and values. Metadata is not embedded in CSV."""
    cols = [c for c in ["timestamp_iso", "unix_timestamp", "timestamp_unix"] if c in merged.columns]
    cols += [c for c in merged.columns if c not in cols]
    df = merged[cols].copy()
    try:
        if 'timestamp_iso' in df.columns:
            ser = df['timestamp_iso']
            if not pd.api.types.is_datetime64_any_dtype(ser):
                ser = pd.to_datetime(ser, errors='coerce')
            iso_str = ser.dt.strftime('%Y-%m-%d %H:%M:%S')
            iso_str = iso_str.where(ser.notna(), df['timestamp_iso'].astype(str))
            df['timestamp_iso'] = "'" + iso_str.fillna("")
    except Exception:
        pass
    outp = raw_dir / f"{sensor}.csv"
    try:
        df.to_csv(outp, index=False)
    except Exception:
        df2 = pd.DataFrame({k: (pd.to_numeric(v, errors="coerce") if k != "timestamp_iso" else pd.to_datetime(v, errors="coerce")) for k, v in df.items()})
        df2.to_csv(outp, index=False)
    return outp
