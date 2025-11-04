from typing import Any, Dict, List, Optional, cast
from pathlib import Path

import numpy as np
import pandas as pd

# AVRO readers
fastavro_reader = cast(Any, None)
DataFileReader = cast(Any, None)
DatumReader = cast(Any, None)
try:
    from fastavro import reader as fastavro_reader  # type: ignore
    _HAS_FASTAVRO = True
except Exception:
    _HAS_FASTAVRO = False

try:
    from avro.datafile import DataFileReader  # type: ignore
    from avro.io import DatumReader  # type: ignore
    _HAS_AVRO = True
except Exception:
    _HAS_AVRO = False


def _decode_avro_stream(avro_path: Path):
    """Yield decoded records from an AVRO file using fastavro or avro-python3."""
    if _HAS_FASTAVRO and fastavro_reader is not None:
        with open(avro_path, "rb") as f:
            for rec in fastavro_reader(f):
                yield cast(Dict[str, Any], rec)
        return
    if _HAS_AVRO and DataFileReader is not None and DatumReader is not None:
        with open(avro_path, "rb") as f:
            dreader = DataFileReader(f, DatumReader())
            for rec in dreader:
                yield cast(Dict[str, Any], rec)
            dreader.close()
        return
    raise RuntimeError("No Avro reader available. Install 'fastavro' or 'avro-python3'.")


def read_single_v6_record(avro_path: Path) -> Optional[dict]:
    """Read and return the first record from an AVRO file, or None if empty.
    Prefers fastavro for performance, falls back to avro-python3.
    """
    if not _HAS_FASTAVRO and not _HAS_AVRO:
        raise RuntimeError("No Avro reader available. Install 'fastavro' or 'avro-python3'.")
    if _HAS_FASTAVRO and fastavro_reader is not None:
        with open(avro_path, "rb") as f:
            rd = fastavro_reader(f)
            try:
                rec = next(rd)
            except StopIteration:
                rec = None
        return cast(Dict[str, Any], rec) if rec is not None else None
    if _HAS_AVRO and DataFileReader is not None and DatumReader is not None:
        with open(avro_path, "rb") as f:
            r = DataFileReader(f, DatumReader())
            try:
                rec = next(r)
            except StopIteration:
                rec = None
            finally:
                r.close()
        return cast(Dict[str, Any], rec) if rec is not None else None
    return None


def micros_list(start: int, fs: float, length: int) -> List[int]:
    """Generate a list of microsecond timestamps given start, sampling frequency, and length."""
    step = 1e6 / fs if fs else 0
    return [round(start + i * step) for i in range(length)]


def v6_sensor_frames(rec: dict) -> Dict[str, pd.DataFrame]:
    """Extract per-sensor DataFrames for Empatica V6-like schema."""
    out: Dict[str, pd.DataFrame] = {}
    if not rec or "rawData" not in rec:
        return out
    rd = rec["rawData"]

    # Accelerometer (ADC -> g)
    try:
        acc = rd["accelerometer"]
        ts = micros_list(acc["timestampStart"], acc["samplingFrequency"], len(acc["x"]))
        pmax = acc["imuParams"]["physicalMax"]; pmin = acc["imuParams"]["physicalMin"]
        dmax = acc["imuParams"]["digitalMax"]; dmin = acc["imuParams"]["digitalMin"]
        delta_physical = pmax - pmin; delta_digital = dmax - dmin if (dmax - dmin) != 0 else 1
        x_g = [v * delta_physical / delta_digital for v in acc["x"]]
        y_g = [v * delta_physical / delta_digital for v in acc["y"]]
        z_g = [v * delta_physical / delta_digital for v in acc["z"]]
        df = pd.DataFrame({"unix_timestamp": ts, "x_g": x_g, "y_g": y_g, "z_g": z_g})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["accelerometer"] = df
    except Exception:
        pass

    # Gyroscope
    try:
        gyro = rd["gyroscope"]
        ts = micros_list(gyro["timestampStart"], gyro["samplingFrequency"], len(gyro["x"]))
        df = pd.DataFrame({"unix_timestamp": ts, "x": gyro["x"], "y": gyro["y"], "z": gyro["z"]})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["gyroscope"] = df
    except Exception:
        pass

    # EDA
    try:
        eda = rd["eda"]
        ts = micros_list(eda["timestampStart"], eda["samplingFrequency"], len(eda["values"]))
        df = pd.DataFrame({"unix_timestamp": ts, "eda": eda["values"]})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["eda"] = df
    except Exception:
        pass

    # Temperature
    try:
        tmp = rd["temperature"]
        ts = micros_list(tmp["timestampStart"], tmp["samplingFrequency"], len(tmp["values"]))
        df = pd.DataFrame({"unix_timestamp": ts, "temperature": tmp["values"]})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["temperature"] = df
    except Exception:
        pass

    # Tags
    try:
        tags = rd["tags"]
        df = pd.DataFrame({"tags_timestamp": tags["tagsTimeMicros"]})
        df["timestamp_iso"] = pd.to_datetime(df["tags_timestamp"], unit="us", utc=True)
        out["tags"] = df
    except Exception:
        pass

    # BVP
    try:
        bvp = rd["bvp"]
        ts = micros_list(bvp["timestampStart"], bvp["samplingFrequency"], len(bvp["values"]))
        df = pd.DataFrame({"unix_timestamp": ts, "bvp": bvp["values"]})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["bvp"] = df
    except Exception:
        pass

    # Systolic peaks
    try:
        sps = rd["systolicPeaks"]
        df = pd.DataFrame({"systolic_peak_timestamp": sps["peaksTimeNanos"]})
        try:
            df["timestamp_iso"] = pd.to_datetime(df["systolic_peak_timestamp"], unit="ns", utc=True)
        except Exception:
            pass
        out["systolic_peaks"] = df
    except Exception:
        pass

    # Steps
    try:
        steps = rd["steps"]
        ts = micros_list(steps["timestampStart"], steps["samplingFrequency"], len(steps["values"]))
        df = pd.DataFrame({"unix_timestamp": ts, "steps": steps["values"]})
        df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
        out["steps"] = df
    except Exception:
        pass

    return out


def generic_sensor_frames(rec: dict) -> Dict[str, pd.DataFrame]:
    """Fallback extractor for Empatica-like AVRO records (schema variations)."""
    out: Dict[str, pd.DataFrame] = {}
    if not rec:
        return out
    container = None
    if isinstance(rec, dict):
        container = rec.get("rawData") or rec.get("data") or rec
    if not isinstance(container, dict):
        return out
    for name, block in container.items():
        try:
            if not isinstance(block, dict):
                continue
            if "tagsTimeMicros" in block and isinstance(block["tagsTimeMicros"], list):
                df = pd.DataFrame({"tags_timestamp": block["tagsTimeMicros"]})
                try:
                    df["timestamp_iso"] = pd.to_datetime(df["tags_timestamp"], unit="us", utc=True)
                except Exception:
                    pass
                out[name] = df
                continue
            if "peaksTimeNanos" in block and isinstance(block["peaksTimeNanos"], list):
                df = pd.DataFrame({"systolic_peak_timestamp": block["peaksTimeNanos"]})
                try:
                    df["timestamp_iso"] = pd.to_datetime(df["systolic_peak_timestamp"], unit="ns", utc=True)
                except Exception:
                    pass
                out[name] = df
                continue
            ts0 = block.get("timestampStart")
            fs = block.get("samplingFrequency")
            if ts0 is None or fs in (None, 0):
                continue
            if isinstance(block.get("values"), list):
                vals = block["values"]
                ts = micros_list(int(ts0), float(fs), len(vals))
                df = pd.DataFrame({"unix_timestamp": ts, name: vals})
                df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
                out[name] = df
                continue
            if all(isinstance(block.get(k), list) for k in ("x", "y", "z")):
                ts = micros_list(int(ts0), float(fs), len(block["x"]))
                df = pd.DataFrame({"unix_timestamp": ts, "x": block["x"], "y": block["y"], "z": block["z"]})
                df["timestamp_iso"] = pd.to_datetime(df["unix_timestamp"], unit="us", utc=True)
                out[name] = df
                continue
        except Exception:
            continue
    return out


from typing import Tuple


def _record_time_range_micros(rec: dict) -> Optional[Tuple[int, int]]:
    """Approximate [start_us, end_us) for an AVRO record using available blocks; None if unknown."""
    try:
        container = rec.get("rawData") or rec.get("data") or rec
        if not isinstance(container, dict):
            return None
        starts: List[int] = []
        ends: List[int] = []
        for _name, block in container.items():
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("tagsTimeMicros"), list) and block["tagsTimeMicros"]:
                vals = block["tagsTimeMicros"]
                s = int(min(vals)); e = int(max(vals)) + 1
                starts.append(s); ends.append(e)
                continue
            if isinstance(block.get("peaksTimeNanos"), list) and block["peaksTimeNanos"]:
                valsn = block["peaksTimeNanos"]
                s = int(min(valsn) // 1000); e = int(max(valsn) // 1000) + 1
                starts.append(s); ends.append(e)
                continue
            ts0 = block.get("timestampStart")
            fs = block.get("samplingFrequency")
            if ts0 is None or not fs:
                continue
            n = None
            if isinstance(block.get("values"), list):
                n = len(block.get("values", []))
            elif all(isinstance(block.get(k), list) for k in ("x", "y", "z")):
                n = len(block.get("x", []))
            if n is None:
                continue
            dur_us = int(round(1e6 * n / float(fs))) if fs else 0
            starts.append(int(ts0))
            ends.append(int(ts0) + max(dur_us, 1))
        if not starts or not ends:
            return None
        return (min(starts), max(ends))
    except Exception:
        return None
