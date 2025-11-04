from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PySide6.QtCore import QThread, Signal
from dateutil import tz

from .time_utils import _date_strs_between, _filter_df_by_time_daily_band
from .io_utils import (
    ensure_unique_folder,
    load_csv_files,
    find_avro_files,
    _write_df,
    _save_sensor_npz,
    _save_sensor_mat,
    _save_sensor_csv,
    _aggregated_measurement_key,
    _device_code_from_dir,
)
from .avro_utils import (
    read_single_v6_record,
    v6_sensor_frames,
    generic_sensor_frames,
    _record_time_range_micros,
)


class SelectionExportWorker(QThread):
    """Export devices for a selected window.
    - Aggregated: one CSV per measurement type (concatenated across days)
    - RAW: one file per sensor in chosen format (NPZ/MAT/CSV)
    """
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(int)
    finished_err = Signal(str)

    def __init__(
        self,
        data_root: Path,
        devices: List[str],
        start_agg: pd.Timestamp,
        end_agg: pd.Timestamp,
        start_raw: Optional[pd.Timestamp],
        end_raw: Optional[pd.Timestamp],
        out_root: Path,
        export_name: Optional[str] = None,
        export_aggregated: bool = True,
        export_raw: bool = True,
        fmt: str = "npz",
        selected_start: Optional[pd.Timestamp] = None,
        selected_end: Optional[pd.Timestamp] = None,
        output_as_local: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.data_root = data_root
        self.devices = devices
        self.start_agg = start_agg
        self.end_agg = end_agg
        self.start_raw = start_raw if start_raw is not None else start_agg
        self.end_raw = end_raw if end_raw is not None else end_agg
        self.selected_start = selected_start if selected_start is not None else self.start_raw
        self.selected_end = selected_end if selected_end is not None else self.end_raw
        self.export_name = export_name
        self.export_aggregated = export_aggregated
        self.export_raw = export_raw
        self.out_root = out_root
        fmt_l = (fmt or "npz").strip().lower()
        self.fmt = "npz" if fmt_l in ("npz", "npx") else fmt_l
        self.output_as_local = bool(output_as_local)
        # Cache tz once for performance
        try:
            self._tz_local = tz.tzlocal()
        except Exception:
            self._tz_local = None

    def _gather_avro_files_for_device(self, device_id: str) -> List[Path]:
        files: List[Path] = []
        for dstr in _date_strs_between(self.start_raw, self.end_raw):
            files.extend(find_avro_files(self.data_root, dstr, device_id))
        return sorted(files)

    def run(self):
        try:
            total = len(self.devices)
            done = 0
            exported = 0
            base_dir = ensure_unique_folder(self.out_root / self.export_name) if self.export_name else self.out_root
            for i, dev in enumerate(self.devices, start=1):
                if self.isInterruptionRequested():
                    self.finished_err.emit("Cancelled")
                    return
                if not dev:
                    done += 1; self.progress.emit(int(done/max(1,total)*100)); continue
                self.status.emit(f"Exporting device {dev} ({i}/{total})")
                meas_dir = base_dir
                meas_dir.mkdir(parents=True, exist_ok=True)
                # Use per-device subfolder when exporting multiple devices to avoid name collisions
                try:
                    dev_code = _device_code_from_dir(dev)
                except Exception:
                    dev_code = dev
                meas_dir_dev = meas_dir / dev_code if len(self.devices) > 1 else meas_dir
                meas_dir_dev.mkdir(parents=True, exist_ok=True)

                # Aggregated
                agg_written = 0
                agg_acc: Dict[str, List[pd.DataFrame]] = {}
                for dstr in _date_strs_between(self.start_agg, self.end_agg):
                    csvs = load_csv_files(self.data_root, dstr, dev)
                    for csvf in csvs:
                        try:
                            df = pd.read_csv(csvf)
                            time_col = None
                            for cand in ["timestamp_iso", "timestamp_unix", "timestamp", "ts", "t"]:
                                if cand in df.columns:
                                    time_col = cand; break
                            if time_col is not None:
                                try:
                                    if time_col == "timestamp_unix":
                                        ts_utc = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                                    else:
                                        ts_utc = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                                    if "timestamp_iso" not in df.columns:
                                        df = df.copy()
                                        df["timestamp_iso"] = ts_utc
                                    else:
                                        df = df.copy()
                                        df["timestamp_iso"] = ts_utc
                                except Exception:
                                    try:
                                        df["timestamp_iso"] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                                    except Exception:
                                        pass
                            df = _filter_df_by_time_daily_band(df, self.start_agg, self.end_agg)
                            if df is not None and not df.empty:
                                key = _aggregated_measurement_key(csvf.stem)
                                agg_acc.setdefault(key, []).append(df)
                        except Exception:
                            continue
                if agg_acc:
                    out_dir = meas_dir_dev / "aggregated"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for stem, parts in agg_acc.items():
                        try:
                            merged = pd.concat(parts, ignore_index=True)
                            if "timestamp_iso" in merged.columns:
                                if not pd.api.types.is_datetime64_any_dtype(merged["timestamp_iso"]):
                                    merged["timestamp_iso"] = pd.to_datetime(merged["timestamp_iso"], utc=True, errors='coerce')
                            else:
                                for cand in ["timestamp_unix", "timestamp", "ts", "t"]:
                                    if cand in merged.columns:
                                        try:
                                            merged["timestamp_iso"] = pd.to_datetime(merged[cand], utc=True, errors='coerce')
                                            break
                                        except Exception:
                                            continue
                            # Keep timestamps in UTC; no conversion to local
                            if "timestamp_iso" in merged.columns:
                                try:
                                    merged = merged.sort_values(by="timestamp_iso")
                                except Exception:
                                    pass
                            dev_code = _device_code_from_dir(dev)
                            out_stem = f"{dev_code}_{stem}" if len(self.devices) > 1 else stem
                            _write_df(merged, out_dir / out_stem, "csv")
                            agg_written += 1
                        except Exception:
                            continue

                # RAW per sensor
                avros = self._gather_avro_files_for_device(dev)
                accums: Dict[str, List[pd.DataFrame]] = {}
                # Bounded concurrency for AVRO reading to utilize CPU without blowing memory
                # Defaults in case env parsing fails
                max_workers = max(1, min(8, (os.cpu_count() or 2)))
                chunk_size = 128
                env_workers = os.getenv("EXPORTER_MAX_WORKERS")
                try:
                    env_workers_val = int(env_workers) if env_workers else None
                except ValueError:
                    env_workers_val = None
                    # Increase parallelism for faster AVRO decode (env var can override)
                    max_workers = env_workers_val or max_workers
                env_chunk = os.getenv("EXPORTER_AVRO_CHUNK")
                try:
                    env_chunk_val = int(env_chunk) if env_chunk else None
                except ValueError:
                    env_chunk_val = None
                    # Larger chunk size reduces thread-pool overhead
                    chunk_size = env_chunk_val or chunk_size
                for ci in range(0, len(avros), chunk_size):
                    if self.isInterruptionRequested():
                        self.finished_err.emit("Cancelled")
                        return
                    chunk = avros[ci:ci+chunk_size]
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = {}
                        for fp in chunk:
                            futs[ex.submit(read_single_v6_record, fp)] = fp
                        for fut in as_completed(futs):
                            fp = futs[fut]
                            try:
                                rec = fut.result()
                            except Exception:
                                continue
                            try:
                                if not rec or not isinstance(rec, dict):
                                    continue
                                # Early skip: if record's overall time range does not intersect [start_raw, end_raw), skip heavy frame builds
                                try:
                                    rng = _record_time_range_micros(rec)
                                except Exception:
                                    rng = None
                                if rng is not None:
                                    try:
                                        s_us = int(pd.Timestamp(self.start_raw).tz_convert('UTC').value // 1000)
                                    except Exception:
                                        s_us = int(pd.Timestamp(self.start_raw).tz_localize('UTC').value // 1000)
                                    try:
                                        e_us = int(pd.Timestamp(self.end_raw).tz_convert('UTC').value // 1000)
                                    except Exception:
                                        e_us = int(pd.Timestamp(self.end_raw).tz_localize('UTC').value // 1000)
                                    r0, r1 = int(rng[0]), int(rng[1])
                                    # For inclusive end: only skip when record ends strictly before start,
                                    # or starts strictly after end.
                                    if (r1 < s_us) or (r0 > e_us):
                                        continue
                                # Do not pre-skip by coarse record range; rely on per-sensor filtering below.
                                frames = v6_sensor_frames(rec) or generic_sensor_frames(rec) or {}
                                for name, df in frames.items():
                                    cut = _filter_df_by_time_daily_band(df, self.start_raw, self.end_raw)
                                    if cut is not None and not cut.empty:
                                        accums.setdefault(name, []).append(cut)
                            except Exception:
                                continue

                raw_written = 0
                sub = {"npz": "raw_npz", "mat": "raw_mat", "csv": "raw_csv"}.get(self.fmt, "raw_npz")
                raw_dir = meas_dir_dev / sub
                raw_dir_created = False
                for name, parts in accums.items():
                    try:
                        merged = pd.concat(parts, ignore_index=True)
                        if "timestamp_iso" in merged.columns:
                            try:
                                ser = merged["timestamp_iso"]
                                if not pd.api.types.is_datetime64_any_dtype(ser):
                                    ser = pd.to_datetime(ser, utc=True, errors='coerce')
                                    merged["timestamp_iso"] = ser
                                    # Skip sorting for speed; input files are processed in chronological order
                            except Exception:
                                try:
                                    merged = merged.sort_values(by="timestamp_iso")
                                except Exception:
                                    pass
                        first_ts = None; last_ts = None
                        try:
                            if "timestamp_iso" in merged.columns and not merged.empty:
                                ser2 = merged["timestamp_iso"]
                                if not pd.api.types.is_datetime64_any_dtype(ser2):
                                    ser2 = pd.to_datetime(ser2, utc=True, errors="coerce")
                                if not ser2.dropna().empty:
                                    first_ts = ser2.min().isoformat()
                                    last_ts = ser2.max().isoformat()
                        except Exception:
                            pass
                        meta = {
                            "device_id": dev,
                            "sensor": name,
                            "start_datetime": pd.Timestamp(self.selected_start).isoformat(),
                            "end_datetime": pd.Timestamp(self.selected_end).isoformat(),
                            "first_timestamp_iso": first_ts,
                            "last_timestamp_iso": last_ts,
                        }
                        dev_code = _device_code_from_dir(dev)
                        sensor_name = f"{dev_code}_{name}" if len(self.devices) > 1 else name
                        if not raw_dir_created:
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            raw_dir_created = True
                        if self.fmt == "mat":
                            _save_sensor_mat(raw_dir, sensor_name, merged, meta)
                        elif self.fmt == "csv":
                            _save_sensor_csv(raw_dir, sensor_name, merged, meta)
                        else:
                            _save_sensor_npz(raw_dir, sensor_name, merged, meta)
                        raw_written += 1
                    except Exception:
                        continue

                exported += 1
                done += 1
                self.progress.emit(int(done / max(1, total) * 100))
                self.status.emit(f"Finished {dev}: raw={raw_written}, agg={agg_written}")

            self.status.emit(f"Export finished. Devices exported: {exported}/{total}")
            self.finished_ok.emit(exported)
        except Exception as e:
            self.finished_err.emit(str(e))


class ExcelBatchWorker(QThread):
    """Process an Excel sheet where each row describes a measurement export.
    Required: device or participant_full_id; start_date/time; end_date/time. Aggregated always CSV.
    """
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(int)  # number of measurements exported
    finished_err = Signal(str)

    def __init__(self, excel_path: Path, data_root: Path, out_root: Path, raw_fmt: str = "npz", time_base_policy: str = "utc", parent=None):
        super().__init__(parent)
        self.excel_path = excel_path
        self.data_root = data_root
        self.out_root = out_root
        fmt_l = (raw_fmt or "npz").strip().lower()
        self.raw_fmt = "npz" if fmt_l in ("npz", "npx") else fmt_l
        pol = (time_base_policy or "utc").strip().lower()
        self.time_base_policy = pol if pol in ("utc", "local") else "utc"
        try:
            self._tz_local = tz.tzlocal()
        except Exception:
            self._tz_local = tz.tzutc()

    def _normalize_columns(self, cols: List[str]) -> Dict[str, str]:
        mapping = {
            "device_id": ["device id", "device_id", "deviceid", "device", "id"],
            "participant_id": ["participant_id", "participant id", "participant", "subject", "studyid", "study id", "study_id"],
            "start_date": ["start date", "start_date", "start day", "startday"],
            "start_time": ["start time", "start_time", "start hour", "starthour"],
            "end_date": ["end date", "end_date", "end day", "endday"],
            "end_time": ["end time", "end_time", "end hour", "endhour"],
            "export_name": ["export file name", "export name", "name", "output name", "folder name"],
            "storage_folder": ["storage folder", "storage folders", "storage_folder", "output folder", "output_folder", "storage", "prage_folder"],
            "participant_full_id": ["participant_full_id", "participant full id", "participant_fullid", "full participant id"],
            "export_format": ["export file format", "format", "file format", "output format"],
            "participant_name": ["participant name", "participant", "subject", "patient name"],
        }
        cols_l = [c.strip().lower() for c in cols]
        result: Dict[str, str] = {}
        for key, aliases in mapping.items():
            for alias in aliases:
                alias_l = alias.strip().lower()
                if alias_l in cols_l:
                    result[key] = cols[cols_l.index(alias_l)]
                    break
        return result

    def _parse_row_window(self, row: pd.Series, colmap: Dict[str, str]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        sd = row.get(colmap.get("start_date", ""), None)
        st = row.get(colmap.get("start_time", ""), None)
        ed = row.get(colmap.get("end_date", ""), None)
        et = row.get(colmap.get("end_time", ""), None)

        def _parse_date(val) -> Optional[Tuple[int,int,int]]:
            try:
                dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
            except Exception:
                dt = pd.NaT
            if pd.isna(dt):
                try:
                    dt = pd.to_datetime(str(val), errors='coerce', dayfirst=True)
                except Exception:
                    dt = pd.NaT
            if pd.isna(dt):
                return None
            return int(dt.year), int(dt.month), int(dt.day)

        def _parse_time(val) -> Tuple[int,int]:
            try:
                if val is None:
                    return 0, 0
                s = str(val).strip()
                if s == '' or s.lower() == 'nan':
                    return 0, 0
                tdt = pd.to_datetime(s, errors='coerce')
                if not pd.isna(tdt):
                    return int(getattr(tdt, 'hour', 0)), int(getattr(tdt, 'minute', 0))
                import re as _re
                mobj = _re.match(r"^(\d{1,2}):(\d{2})", s)
                if mobj:
                    return int(mobj.group(1)), int(mobj.group(2))
            except Exception:
                pass
            return 0, 0

        ymd_start = _parse_date(sd)
        ymd_end = _parse_date(ed)
        if not ymd_start or not ymd_end:
            return None
        sh, sm = _parse_time(st)
        eh, em = _parse_time(et)

        tzinfo = self._tz_local if self.time_base_policy == 'local' else tz.tzutc()

        try:
            start_local = pd.Timestamp(year=ymd_start[0], month=ymd_start[1], day=ymd_start[2], hour=sh, minute=sm, second=0, tz=tzinfo)
            end_local = pd.Timestamp(year=ymd_end[0], month=ymd_end[1], day=ymd_end[2], hour=eh, minute=em, second=0, tz=tzinfo)
        except Exception:
            return None

        try:
            if eh == 0 and em == 0:
                end_local = end_local + pd.Timedelta(days=1)
        except Exception:
            pass

        try:
            start_utc = start_local.tz_convert('UTC')
        except Exception:
            start_utc = pd.Timestamp(start_local).tz_localize('UTC') if pd.Timestamp(start_local).tzinfo is None else pd.Timestamp(start_local).tz_convert('UTC')
        try:
            end_utc = end_local.tz_convert('UTC')
        except Exception:
            end_utc = pd.Timestamp(end_local).tz_localize('UTC') if pd.Timestamp(end_local).tzinfo is None else pd.Timestamp(end_local).tz_convert('UTC')

        if end_utc <= start_utc:
            return None
        return start_utc, end_utc

    def _resolve_device_dir(self, start: pd.Timestamp, end: pd.Timestamp, device_id: Optional[str], participant_full_id: Optional[str], participant_id_only: Optional[str] = None) -> Optional[str]:
        cand_codes: List[str] = []
        if device_id:
            cand_codes.append(str(device_id).strip())
            try:
                parts = str(device_id).strip().split('-')
                if parts:
                    cand_codes.append(parts[0])
                    cand_codes.append(parts[-1])
            except Exception:
                pass
        if participant_full_id:
            try:
                cand_codes.append(str(participant_full_id).strip().split('-')[-1])
            except Exception:
                cand_codes.append(str(participant_full_id).strip())
        if participant_id_only:
            try:
                cand_codes.append(str(participant_id_only).strip())
            except Exception:
                pass
        cand_codes = [c for c in {c for c in cand_codes if c}]
        if not cand_codes:
            return None
        candidates = set()
        for dstr in _date_strs_between(start, end):
            day_dir = self.data_root / dstr
            if not day_dir.exists():
                continue
            for dev_dir in [p for p in day_dir.iterdir() if p.is_dir()]:
                name = dev_dir.name
                for code in cand_codes:
                    if (
                        name == code or name.startswith(f"{code}-") or name.endswith(f"-{code}") or code in name
                    ):
                        candidates.add(name)
        if not candidates:
            return device_id or None
        def score(name: str) -> Tuple[int, str]:
            s = 0
            for dstr in _date_strs_between(start, end):
                base = self.data_root / dstr / name
                if (base / "digital_biomarkers" / "aggregated_per_minute").exists():
                    s += 2
                if (base / "raw_data").exists():
                    s += 1
            return (-s, name)
        return sorted(candidates, key=score)[0]

    def _gather_avro_files(self, device_id: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
        files: List[Path] = []
        for dstr in _date_strs_between(start, end):
            files.extend(find_avro_files(self.data_root, dstr, device_id))
        return sorted(files)

    def _export_raw(self, files: List[Path], start: pd.Timestamp, end: pd.Timestamp, out_dir: Path, fmt: str, device_id: Optional[str] = None) -> int:
        sub = {"npz": "raw_npz", "mat": "raw_mat", "csv": "raw_csv"}.get((fmt or "npz").lower(), "raw_npz")
        raw_dir = out_dir / sub
        raw_dir_created = False
        accums: Dict[str, List[pd.DataFrame]] = {}

        # Bounded concurrency similar to SelectionExportWorker, with sensible defaults and env overrides
        max_workers = max(1, min(8, (os.cpu_count() or 2)))
        chunk_size = 128
        env_workers = os.getenv("EXPORTER_MAX_WORKERS")
        try:
            env_workers_val = int(env_workers) if env_workers else None
        except ValueError:
            env_workers_val = None
        if env_workers_val:
            max_workers = env_workers_val
        env_chunk = os.getenv("EXPORTER_AVRO_CHUNK")
        try:
            env_chunk_val = int(env_chunk) if env_chunk else None
        except ValueError:
            env_chunk_val = None
        if env_chunk_val:
            chunk_size = env_chunk_val

        for ci in range(0, len(files), chunk_size):
            chunk = files[ci:ci + chunk_size]
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(read_single_v6_record, fp): fp for fp in chunk}
                for fut in as_completed(futs):
                    try:
                        rec = fut.result()
                    except Exception:
                        continue
                    try:
                        if not rec or not isinstance(rec, dict):
                            continue
                        # Early skip for batch: skip records outside [start, end] (end inclusive)
                        try:
                            rng = _record_time_range_micros(rec)
                        except Exception:
                            rng = None
                        if rng is not None:
                            try:
                                s_us = int(pd.Timestamp(start).tz_convert('UTC').value // 1000)
                            except Exception:
                                s_us = int(pd.Timestamp(start).tz_localize('UTC').value // 1000)
                            try:
                                e_us = int(pd.Timestamp(end).tz_convert('UTC').value // 1000)
                            except Exception:
                                e_us = int(pd.Timestamp(end).tz_localize('UTC').value // 1000)
                            r0, r1 = int(rng[0]), int(rng[1])
                            if (r1 < s_us) or (r0 > e_us):
                                continue
                        frames = v6_sensor_frames(rec) or generic_sensor_frames(rec)
                        for name, df in frames.items():
                            cut = _filter_df_by_time_daily_band(df, start, end)
                            if cut is not None and not cut.empty:
                                accums.setdefault(name, []).append(cut)
                    except Exception:
                        continue

        written = 0
        for name, parts in accums.items():
            try:
                merged = pd.concat(parts, ignore_index=True)
                # Intentionally skip sorting for speed; data is typically near-ordered across records
            except Exception:
                merged = pd.concat(parts, ignore_index=True)
            if not merged.empty:
                try:
                    if not raw_dir_created:
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_dir_created = True
                    # Do not prefix device code in batch filenames; rely on distinct storage_folder per row
                    sensor_name = name
                    if fmt == "mat":
                        _save_sensor_mat(raw_dir, sensor_name, merged, {"sensor": name, "device_id": device_id})
                    elif fmt == "csv":
                        _save_sensor_csv(raw_dir, sensor_name, merged, {"sensor": name, "device_id": device_id})
                    else:
                        _save_sensor_npz(raw_dir, sensor_name, merged, {"sensor": name, "device_id": device_id})
                    written += 1
                except Exception:
                    continue
        return written

    def _export_aggregated(self, device_id: str, start: pd.Timestamp, end: pd.Timestamp, out_dir: Path, fmt: str) -> int:
        agg_dir = out_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        acc: Dict[str, List[pd.DataFrame]] = {}
        for dstr in _date_strs_between(start, end):
            csvs = load_csv_files(self.data_root, dstr, device_id)
            for csvf in csvs:
                try:
                    df = pd.read_csv(csvf)
                    time_col = None
                    for cand in ["timestamp_iso", "timestamp_unix", "time_local", "timestamp", "ts", "t"]:
                        if cand in df.columns:
                            time_col = cand
                            break
                    if time_col is not None:
                        if time_col == "timestamp_iso":
                            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                        df = _filter_df_by_time_daily_band(df, start, end)
                    if df is not None and not df.empty:
                        key = _aggregated_measurement_key(csvf.stem)
                        acc.setdefault(key, []).append(df)
                except Exception:
                    continue
        written = 0
        for stem, parts in acc.items():
            try:
                merged = pd.concat(parts, ignore_index=True)
                for cand in ["timestamp_iso", "timestamp_unix", "time_local", "timestamp", "ts", "t"]:
                    if cand in merged.columns:
                        if cand == "timestamp_iso" and not pd.api.types.is_datetime64_any_dtype(merged[cand]):
                            merged[cand] = pd.to_datetime(merged[cand], utc=True, errors='coerce')
                        try:
                            merged = merged.sort_values(by=cand)
                        except Exception:
                            pass
                        break
                # Keep timestamp_iso as UTC; no conversion to local
                # Do not prefix device code in batch aggregated filenames; rely on distinct storage_folder per row
                _write_df(merged, agg_dir / stem, fmt)
                written += 1
            except Exception:
                continue
        return written

    def run(self):
        try:
            # Minimal status during batch processing
            self.status.emit("Batch export in progress…")
            try:
                df = pd.read_excel(self.excel_path)
            except Exception as e:
                self.finished_err.emit(f"Failed to read Excel: {e}")
                return
            if df is None or df.empty:
                self.finished_err.emit("Excel file has no rows")
                return
            colmap = self._normalize_columns(list(df.columns))
            if not colmap.get("device_id") and not colmap.get("participant_full_id") and not colmap.get("participant_id"):
                self.finished_err.emit("Excel is missing a deviceID or participant identifier column")
                return
            total = len(df)
            exported = 0
            for row_num, (_, row) in enumerate(df.iterrows(), start=1):
                if self.isInterruptionRequested():
                    self.finished_err.emit("Cancelled")
                    return
                self.progress.emit(int((row_num)/max(1,total)*100))
                win = self._parse_row_window(row, colmap)
                if not win:
                    continue
                start_utc, end_utc = win
                device_id_val = str(row.get(colmap.get("device_id", ""), "")).strip() if colmap.get("device_id") else ""
                participant_full = str(row.get(colmap.get("participant_full_id", ""), "")).strip() if colmap.get("participant_full_id") else None
                participant_id_only = str(row.get(colmap.get("participant_id", ""), "")).strip() if colmap.get("participant_id") else None
                device_dir = self._resolve_device_dir(start_utc, end_utc, device_id_val or None, participant_full, participant_id_only)
                if not device_dir:
                    continue
                export_name = None
                if colmap.get("storage_folder"):
                    export_name = str(row.get(colmap["storage_folder"], "")).strip() or None
                if not export_name and colmap.get("export_name"):
                    export_name = str(row.get(colmap["export_name"], "")).strip() or None
                row_fmt = self.raw_fmt
                if colmap.get("export_format"):
                    try:
                        rf = str(row.get(colmap["export_format"], "")).strip().lower()
                        if rf in ("npz", "mat", "csv"):
                            row_fmt = rf
                    except Exception:
                        pass
                # Base output directory for this row (do NOT create extra per-device subfolder; filenames will include device code)
                out_dir = ensure_unique_folder(self.out_root / export_name) if export_name else self.out_root
                try:
                    _ = self._export_aggregated(device_dir, start_utc, end_utc, out_dir, fmt="csv")
                except Exception:
                    pass
                try:
                    files = self._gather_avro_files(device_dir, start_utc, end_utc)
                    _ = self._export_raw(files, start_utc, end_utc, out_dir, fmt=row_fmt, device_id=device_dir)
                except Exception:
                    continue
                exported += 1
            self.finished_ok.emit(exported)
        except Exception as e:
            self.finished_err.emit(str(e))


class RawAllExportWorker(QThread):
    """Export all RAW .avro files for devices into clinician-friendly files (no time filtering)."""
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(int)
    finished_err = Signal(str)

    def __init__(self, data_root: Path, devices: List[str], out_root: Path, date_map: Optional[Dict[str, List[str]]] = None, export_name: Optional[str] = None, fmt: str = "npz", parent=None):
        super().__init__(parent)
        self.data_root = data_root
        self.devices = devices
        self.out_root = out_root
        self.date_map = date_map or {}
        self.export_name = export_name
        fmt_l = (fmt or "npz").strip().lower()
        self.fmt = "npz" if fmt_l in ("npz", "npx") else fmt_l

    def _list_date_dirs(self) -> List[str]:
        import re
        dstrs: List[str] = []
        try:
            for p in sorted(self.data_root.iterdir()):
                if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name):
                    dstrs.append(p.name)
        except Exception:
            pass
        return dstrs

    def _gather_avro_files_for_device_all(self, device_id: str) -> List[Path]:
        files: List[Path] = []
        date_list = self.date_map.get(device_id)
        if not date_list:
            date_list = self._list_date_dirs()
        for dstr in date_list:
            files.extend(find_avro_files(self.data_root, dstr, device_id))
        return sorted(files)

    def _save_sensor(self, raw_dir: Path, sensor: str, merged: pd.DataFrame, dev: str):
        meta = {"device_id": dev, "sensor": sensor, "export_scope": "ALL"}
        if self.fmt == "mat":
            _save_sensor_mat(raw_dir, sensor, merged, meta)
        elif self.fmt == "csv":
            _save_sensor_csv(raw_dir, sensor, merged, meta)
        else:
            _save_sensor_npz(raw_dir, sensor, merged, meta)

    def run(self):
        try:
            total = len(self.devices)
            done = 0
            exported_devices = 0
            for i, dev in enumerate(self.devices, start=1):
                if self.isInterruptionRequested():
                    self.finished_err.emit("Cancelled")
                    return
                if not dev:
                    done += 1; self.progress.emit(int(done / max(1, total) * 100)); continue
                self.status.emit(f"Exporting RAW (all) for {dev} ({i}/{total})")
                meas_dir = ensure_unique_folder(self.out_root / self.export_name) if self.export_name else self.out_root
                sub = {"npz": "raw_npz", "mat": "raw_mat", "csv": "raw_csv"}.get(self.fmt, "raw_npz")
                raw_dir = meas_dir / sub
                raw_dir.mkdir(parents=True, exist_ok=True)

                avros = self._gather_avro_files_for_device_all(dev)
                self.status.emit(f"{dev}: AVRO files found: {len(avros)}")
                accums: Dict[str, List[pd.DataFrame]] = {}
                # Bounded concurrency for AVRO reading (no time filtering in ALL scope)
                env_workers = os.getenv("EXPORTER_MAX_WORKERS")
                try:
                    env_workers_val = int(env_workers) if env_workers else None
                except ValueError:
                    env_workers_val = None
                max_workers = env_workers_val or max(1, min(4, (os.cpu_count() or 2) - 1))
                env_chunk = os.getenv("EXPORTER_AVRO_CHUNK")
                try:
                    env_chunk_val = int(env_chunk) if env_chunk else None
                except ValueError:
                    env_chunk_val = None
                chunk_size = env_chunk_val or 48
                processed = 0
                for ci in range(0, len(avros), chunk_size):
                    if self.isInterruptionRequested():
                        self.finished_err.emit("Cancelled")
                        return
                    chunk = avros[ci:ci+chunk_size]
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = {ex.submit(read_single_v6_record, fp): fp for fp in chunk}
                        for fut in as_completed(futs):
                            fp = futs[fut]
                            processed += 1
                            if processed % 10 == 0 or processed == 1 or processed == len(avros):
                                self.status.emit(f"{dev}: reading {fp.name} ({processed}/{len(avros)})")
                            try:
                                rec = fut.result()
                                if not rec or not isinstance(rec, dict):
                                    continue
                                frames = v6_sensor_frames(rec) or generic_sensor_frames(rec)
                                if not frames:
                                    continue
                                for name, df in frames.items():
                                    accums.setdefault(name, []).append(df)
                            except Exception:
                                continue

                raw_written = 0
                self.status.emit(f"{dev}: sensors accumulated: {len(accums)}")
                for name, parts in accums.items():
                    try:
                        merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                        # Ensure chronological order for consistency when reading in parallel
                        try:
                            if "timestamp_iso" in merged.columns:
                                if not pd.api.types.is_datetime64_any_dtype(merged["timestamp_iso"]):
                                    merged["timestamp_iso"] = pd.to_datetime(merged["timestamp_iso"], utc=True, errors='coerce')
                                merged = merged.sort_values(by="timestamp_iso")
                            elif "timestamp_unix" in merged.columns:
                                merged = merged.sort_values(by="timestamp_unix")
                        except Exception:
                            pass
                        if not merged.empty:
                            dev_code = _device_code_from_dir(dev)
                            sensor_name = f"{dev_code}_{name}"
                            self._save_sensor(raw_dir, sensor_name, merged, dev)
                            raw_written += 1
                        else:
                            self.status.emit(f"{dev}/{name}: merged data is empty")
                    except Exception:
                        self.status.emit(f"{dev}/{name}: failed to save file")
                        continue

                exported_devices += 1
                done += 1
                self.progress.emit(int(done / max(1, total) * 100))
                self.status.emit(f"Finished {dev}: {self.fmt} files={raw_written}")

            self.status.emit(f"RAW (all) export finished. Devices: {exported_devices}/{total}")
            self.finished_ok.emit(exported_devices)
        except Exception as e:
            self.finished_err.emit(str(e))
