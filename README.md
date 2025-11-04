# Empatica Data Exporter

A lightweight GUI to export Empatica device data

This app uses PySide6 (Qt6), pandas, numpy, and decodes Empatica RAW .avro files via `fastavro` (preferred) or `avro-python3`.

## Features
- Manual export: select a date/time window and one or multiple devices
- RAW export: per-sensor in NPZ, MAT, or CSV; only written if data exists (no empty files)
- Aggregated export: always CSV; concatenates across days and sorts by time
- Batch export: process an Excel file describing multiple measurements
- Robust threading: cooperative cancellation, no QThread destruction warnings
- Per-device subfolders for multi-device exports (manual and batch)

## Quick start

### Option A: Use conda (recommended on Windows)

```bash
# In Git Bash or cmd
cd /d/TUe/exporter
conda env create -f environment.yml
conda activate empatica-exporter
python empatica_data_exporter.py
```

### Option B: Use pip in an existing Python 3.10+ environment

```bash
# In Git Bash or cmd
cd /d/TUe/exporter
python -m pip install -r requirements.txt
python empatica_data_exporter.py
```

The GUI window is titled "Empatica Data Exporter". Pick your data root (unzipped Empatica directory), choose an export option (Manual or Batch), and follow the dialogs.


## Batch Excel columns
- Supported aliases are detected for: device_id, study_id, start_date, start_time, end_date, end_time, storage_folder, export_name, export_format, etc.

## Output layout
- Manual (multi-device):
   - `<out>/<export_name>/<device_code>/aggregated/*.csv`
   - `<out>/<export_name>/<device_code>/raw_npz|raw_mat|raw_csv/*`
- Batch (respects your storage_folder – no extra per-device folder):
   - Aggregated: `<out>/<storage_folder>/aggregated/*.csv`
   - RAW: `<out>/<storage_folder>/raw_npz|raw_mat|raw_csv/*.{npz|mat|csv}`
   - Note: If you export multiple devices into the same `storage_folder`, filenames can collide. Use a unique `storage_folder` per device/row to keep files separate.
- Only real RAW data writes files; no empty placeholders or empty folders.

## Troubleshooting

- If you still get a Qt plugin error, ensure `PySide6` is installed in your active environment and retry. The app already tries to set the plugin path at runtime.
- For Excel reading, `openpyxl` is required (already included in dependencies).