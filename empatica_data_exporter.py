import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets, QtGui

try:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
except Exception:
    pass

# Ensure Qt can locate the platform plugins (Windows: qwindows.dll) when running from source.
try:
    import os, pathlib, PySide6 as _P6
    if not getattr(sys, "frozen", False):
        if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
            p_pkg = pathlib.Path(_P6.__file__).parent / "plugins" / "platforms"
            if p_pkg.exists():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(p_pkg)
            else:
                p_conda = Path(sys.executable).parent.parent / "Library" / "plugins" / "platforms"
                if p_conda.exists():
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(p_conda)
except Exception:
    pass

# Split modules: workers and utilities
from Empatica_data_exporter import (
    SelectionExportWorker,
    ExcelBatchWorker,
    RawAllExportWorker,
    ensure_unique_folder,
    load_csv_files,
    find_avro_files,
    _date_strs_between,
    _device_code_from_dir,
)

 

# --------- Local lightweight helpers for device discovery and data presence ---------
def find_devices_and_dates(root: Path) -> Dict[str, List[str]]:
    """Scan root for YYYY-MM-DD folders and collect device directory names -> list of dates.
    Example structure: <root>/<YYYY-MM-DD>/<device_id>/...
    """
    mapping: Dict[str, List[str]] = {}
    try:
        for day in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", day.name):
                continue
            try:
                for dev_dir in sorted(p for p in day.iterdir() if p.is_dir()):
                    lst = mapping.setdefault(dev_dir.name, [])
                    if day.name not in lst:
                        lst.append(day.name)
            except Exception:
                continue
    except Exception:
        pass
    return mapping


def _device_has_any_data(root: Path, device_dir_name: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Return True if any aggregated CSV or RAW .avro exists for the device across dates intersecting [start,end)."""
    try:
        for dstr in _date_strs_between(start, end):
            if load_csv_files(root, dstr, device_dir_name):
                return True
            av = find_avro_files(root, dstr, device_dir_name)
            if av:
                return True
    except Exception:
        pass
    return False


def _range_has_any_data(root: Path, devices: List[str], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    try:
        for dev in devices:
            if _device_has_any_data(root, dev, start, end):
                return True
    except Exception:
        pass
    return False


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # type: ignore[override]
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # type: ignore[override]
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        try:
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, (float, np.floating)):
                return f"{val:.6g}"
            return str(val)
        except Exception:
            return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return str(section)
        else:
            return str(section)


# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QWidget):
    """Minimal main window: pick data root, choose export (manual or batch), select RAW format."""
    def __init__(self):
        super().__init__()
        # Rename as requested
        self.setWindowTitle("Empatica Data Exporter")
        # Compact initial size (still resizable) — widen slightly so window title is fully visible
        self.resize(560, 280)
        self.setMinimumSize(520, 240)

        self.root_dir: Optional[Path] = None
        self.devices = {}
        # Main vertical layout
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)
        # Keep content pinned to the top-left and compute a tight minimum size
        main.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        main.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        # Select data folder (full width)
        self.btn_folder = QtWidgets.QPushButton("Select Data Folder")
        self.btn_folder.setMinimumHeight(26)
        try:
            self.btn_folder.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            self.btn_folder.setFixedWidth(220)
        except Exception:
            pass
        main.addWidget(self.btn_folder, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        # Selected folder label
        self.lbl_folder = QtWidgets.QLabel("No folder selected")
        main.addWidget(self.lbl_folder)
        # Export option selector as a button with menu
        self.btn_export_option = QtWidgets.QPushButton("Select Export Option")
        self.btn_export_option.setMinimumHeight(24)
        try:
            self.btn_export_option.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            self.btn_export_option.setFixedWidth(220)
        except Exception:
            pass
        self.menu_export = QtWidgets.QMenu(self)
        self.act_manual = self.menu_export.addAction("Manual export")
        self.act_batch = self.menu_export.addAction("Batch export")
        self.btn_export_option.setMenu(self.menu_export)
        main.addWidget(self.btn_export_option, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        # Format row
        row_fmt = QtWidgets.QHBoxLayout()
        row_fmt.setContentsMargins(0, 0, 0, 0)
        row_fmt.setSpacing(8)
        row_fmt.addWidget(QtWidgets.QLabel("Format:"))
        self.cmb_raw_fmt = QtWidgets.QComboBox()
        self.cmb_raw_fmt.addItems(["NPZ", "MAT", "CSV"])  # default NPZ
        # Make the format selector compact
        try:
            self.cmb_raw_fmt.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            self.cmb_raw_fmt.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            self.cmb_raw_fmt.setFixedWidth(100)
            self.cmb_raw_fmt.setFixedHeight(24)
        except Exception:
            pass
        row_fmt.addWidget(self.cmb_raw_fmt)
        row_fmt.addStretch(1)
        main.addLayout(row_fmt)

        # Start export button
        self.btn_start = QtWidgets.QPushButton("Start Export…")
        self.btn_start.setMinimumHeight(30)
        try:
            self.btn_start.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            self.btn_start.setFixedWidth(200)
        except Exception:
            pass
        main.addWidget(self.btn_start, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        # Footer
        self.lbl_footer = QtWidgets.QLabel("TU/e Empatica data exporter")
        self.lbl_footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        font = self.lbl_footer.font(); font.setPointSize(max(font.pointSize()-1, 8)); self.lbl_footer.setFont(font)
        main.addWidget(self.lbl_footer)

        # Signals
        self.btn_folder.clicked.connect(self.pick_folder)
        self.btn_start.clicked.connect(self._on_start_export)
        self.act_manual.triggered.connect(lambda: self._set_export_choice("manual"))
        self.act_batch.triggered.connect(lambda: self._set_export_choice("batch"))

        # Default export choice: require explicit selection; disable Start until chosen
        self._export_choice: Optional[str] = None
        self.btn_start.setEnabled(False)
        self._update_format_enabled()

        # State
        self._batch_worker = None
        self._rawall_worker = None
        self._selection_worker = None

        # Pack window down to content to avoid large empty areas
        try:
            self.adjustSize()
        except Exception:
            pass

    

    def _prompt_time_range(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, bool]]:
        # Dialog to collect start/end date and time
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Time Range")
        v = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        de_start = QtWidgets.QDateEdit(); de_start.setCalendarPopup(True)
        de_end = QtWidgets.QDateEdit(); de_end.setCalendarPopup(True)
        te_start = QtWidgets.QTimeEdit(); te_end = QtWidgets.QTimeEdit()
    # UTC-only time base
        for w in (de_start, de_end):
            try: w.setDisplayFormat("dd/MM/yyyy")
            except Exception: pass
        for w in (te_start, te_end):
            try: w.setDisplayFormat("HH:mm")
            except Exception: pass
    # defaults: today 00:00 to tomorrow 00:00
        base = pd.Timestamp.today(tz="UTC").normalize(); end = base + pd.Timedelta(days=1)
        de_start.setDate(QtCore.QDate(base.year, base.month, base.day))
        de_end.setDate(QtCore.QDate(end.year, end.month, end.day))
        te_start.setTime(QtCore.QTime(0,0)); te_end.setTime(QtCore.QTime(0,0))
        form.addRow("Start Date:", de_start)
        form.addRow("Start Time:", te_start)
        form.addRow("End Date:", de_end)
        form.addRow("End Time:", te_end)
    # Time base is UTC-only; remove extra label for a cleaner UI
        v.addLayout(form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        # Build timestamps with inclusive-at-midnight rule
        sd, ed = de_start.date(), de_end.date()
        st, et = te_start.time(), te_end.time()
        # Interpret input times as UTC for filtering and output.
        start_ts = pd.Timestamp(year=sd.year(), month=sd.month(), day=sd.day(), hour=st.hour(), minute=st.minute(), second=0, tz='UTC')
        end_ts = pd.Timestamp(year=ed.year(), month=ed.month(), day=ed.day(), hour=et.hour(), minute=et.minute(), second=0, tz='UTC')
        if et.hour() == 0 and et.minute() == 0:
            end_ts = end_ts + pd.Timedelta(days=1)
        # Output window equals UTC window (no Local conversion)
        output_as_local = False
        selected_start_ts, selected_end_ts = start_ts, end_ts
        if end_ts <= start_ts:
            QtWidgets.QMessageBox.information(self, "Info", "End time must be after start time."); return None
        return start_ts, end_ts, selected_start_ts, selected_end_ts, output_as_local

    def _prompt_devices(self) -> Optional[List[str]]:
        # This method is now called after we know the time window; it assumes self._device_picker_options is set.
        opts: List[Tuple[str, str]] = getattr(self, "_device_picker_options", [])  # (code, full_dir_name)
        if not opts:
            QtWidgets.QMessageBox.information(self, "Info", "No devices with data in the selected time range."); return None
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Device(s)")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for code, full in opts:
            item = QtWidgets.QListWidgetItem(code)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, full)
            listw.addItem(item)
        v.addWidget(listw)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        sels = []
        for it in listw.selectedItems():
            full = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if full:
                sels.append(full)
        return sels or None

    def _update_format_enabled(self):
        # Always allow selecting RAW format; in Batch this applies to the whole run
        try:
            self.cmb_raw_fmt.setEnabled(True)
            self.cmb_raw_fmt.setToolTip("RAW file format. In Batch, this applies to all rows; aggregated is always CSV.")
        except Exception:
            pass

    def _set_export_choice(self, choice: str):
        choice_l = (choice or "").strip().lower()
        self._export_choice = "batch" if choice_l.startswith("batch") else ("manual" if choice_l.startswith("manual") else None)
        # Reflect selection in the button text
        try:
            if self._export_choice is None:
                self.btn_export_option.setText("Select Export Option")
                self.btn_start.setEnabled(False)
            else:
                label = "Manual export" if self._export_choice == "manual" else "Batch export"
                self.btn_export_option.setText(f"Export Option: {label}")
                self.btn_start.setEnabled(True)
        except Exception:
            pass
        self._update_format_enabled()

    def _on_start_export(self):
        # Route to the chosen export flow
        try:
            if not self._export_choice:
                QtWidgets.QMessageBox.information(self, "Select export option", "Please choose an export option (Manual or Batch) before starting.")
                return
            if (self._export_choice or "").startswith("batch"):
                self.run_batch_file()
            else:
                self._export_selection()
        except Exception:
            # Fallback to manual selection
            self._export_selection()

    # ---------------- Top-level
    def pick_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Root folder (unzipped Empatica)")
        if d:
            self.root_dir = Path(d)
            self.lbl_folder.setText(str(self.root_dir))
            self.devices = find_devices_and_dates(self.root_dir)

    

    def _export_selection(self):
        # Validate base inputs
        if not self.root_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a data folder first."); return
        # Step 1: time range
        tr = self._prompt_time_range()
        if tr is None:
            return
        start_ts, end_ts, selected_start_ts, selected_end_ts, output_as_local = tr
        # Step 2: device(s) — only show devices that have any data in this window, and display codes only
        # Build filtered options based on selected window
        all_devs = sorted(self.devices.keys()) if self.devices else []
        opts: List[Tuple[str, str]] = []  # (code, full_dir)
        for dname in all_devs:
            if _device_has_any_data(self.root_dir, dname, start_ts, end_ts):
                opts.append((_device_code_from_dir(dname), dname))
        if not opts:
            QtWidgets.QMessageBox.information(self, "No data", "There is no data for any device in this date range.")
            return
        # store for picker use
        self._device_picker_options = opts
        # re-prompt until at least one selected or user cancels
        devices: Optional[List[str]] = None
        while True:
            devices = self._prompt_devices()
            if devices is None:
                # User canceled device selection; abort export without resetting time dialog
                return
            if len(devices) == 0:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please select at least one device.")
                continue
            break
        out_root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output root for exports")
        if not out_root:
            return
        out_root_p = Path(out_root)
        start_raw_ts = start_ts
        end_raw_ts = end_ts

        # Quick pre-check (lightweight): ensure there is at least one aggregated CSV present
        # in the covered dates for the selected devices; avoid reading CSV contents.
        agg_count = 0
        for dev in devices:
            for dstr in _date_strs_between(start_ts, end_ts):
                if load_csv_files(self.root_dir, dstr, dev):
                    agg_count += 1
                    break
        if agg_count == 0:
            QtWidgets.QMessageBox.information(self, "No data in this range", "There is no aggregated data on the selected dates. Please adjust the time or choose other device(s).")
            return

        # Global preflight: if neither aggregated nor raw is present on any covered date, stop
        if not _range_has_any_data(self.root_dir, devices, start_ts, end_ts):
            known = ", ".join(sorted(self.devices.get(devices[0], []))) if len(devices) == 1 else "various"
            QtWidgets.QMessageBox.information(
                self,
                "No data in selected range",
                f"The selected date range has no data for the chosen device(s).\nKnown dates: {known or 'n/a'}.\nPlease pick a date within the available range."
            )
            return

        # Ask the user for an (optional) short export folder name
        export_name, ok = QtWidgets.QInputDialog.getText(self, "Export folder name", "Optional short name for export folder (leave empty to auto-generate):")
        if not ok:
            return

        # Start worker
        progress_dialog = QtWidgets.QProgressDialog("Starting export...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Export Selection")
        progress_dialog.setAutoClose(True); progress_dialog.setAutoReset(True); progress_dialog.setMinimumDuration(0)

        worker = SelectionExportWorker(
            data_root=self.root_dir,
            devices=devices,
            start_agg=start_ts,
            end_agg=end_ts,
            start_raw=start_raw_ts,
            end_raw=end_raw_ts,
            out_root=out_root_p,
            export_name=str(export_name).strip() or None,
            export_aggregated=True,
            export_raw=True,
            fmt=(self.cmb_raw_fmt.currentText() or "NPZ").lower(),
            selected_start=selected_start_ts,
            selected_end=selected_end_ts,
            output_as_local=output_as_local,
        )
        worker.progress.connect(progress_dialog.setValue)
        worker.status.connect(progress_dialog.setLabelText)

        def _ok(n):
            progress_dialog.setValue(100)
            progress_dialog.close()
            QtWidgets.QMessageBox.information(self, "Export finished", f"Devices exported: {n}/{len(devices)}")
            self._selection_worker = None

        def _err(msg):
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(self, "Export error", msg)
            self._selection_worker = None

        worker.finished_ok.connect(_ok)
        worker.finished_err.connect(_err)

        def _cancel():
            try:
                if worker and worker.isRunning():
                    worker.requestInterruption()
                    worker.wait(5000)
            except Exception:
                pass

        progress_dialog.canceled.connect(_cancel)
        self._selection_worker = worker
        worker.start()

    def _export_raw_all(self):
        # Export all RAW.avro to clinician-friendly files (NPZ/MAT/H5/CSV) for selected devices.
        if not self.root_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a data folder first."); return
        devices = self._prompt_devices() or []
        if not devices:
            QtWidgets.QMessageBox.information(self, "Info", "No devices selected for export."); return
        out_root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output root for RAW export (All)")
        if not out_root:
            return
        out_root_p = Path(out_root)
        # Optional top-level export folder name
        export_name, ok = QtWidgets.QInputDialog.getText(self, "Export folder name", "Optional short name (leave empty to auto-generate per device):")
        if not ok:
            return
        # Start worker
        progress_dialog = QtWidgets.QProgressDialog("Starting RAW export (All)...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Export RAW (All)")
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.setMinimumDuration(0)

        self._rawall_worker = RawAllExportWorker(
            data_root=self.root_dir,
            devices=devices,
            out_root=out_root_p,
            date_map=self.devices,
            export_name=str(export_name).strip() or None,
            fmt=(self.cmb_raw_fmt.currentText() or "NPZ").lower(),
        )
        self._rawall_worker.progress.connect(progress_dialog.setValue)
        self._rawall_worker.status.connect(progress_dialog.setLabelText)

        def _ok(n):
            progress_dialog.setValue(100)
            progress_dialog.close()
            QtWidgets.QMessageBox.information(self, "Export finished", f"RAW (All) devices exported: {n}/{len(devices)}")
        def _err(msg):
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(self, "Export error", msg)

        self._rawall_worker.finished_ok.connect(_ok)
        self._rawall_worker.finished_err.connect(_err)

        def _cancel():
            try:
                if self._rawall_worker and self._rawall_worker.isRunning():
                    self._rawall_worker.requestInterruption()
                    self._rawall_worker.wait(5000)
            except Exception:
                pass
        progress_dialog.canceled.connect(_cancel)
        self._rawall_worker.start()

    def run_batch_file(self):
        # 1) Ensure data root is selected (same behavior as manual export)
        if not self.root_dir:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a data folder first.")
            return
        # 2) Choose user-provided XLSX batch file
        xlsx_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select batch XLSX file", filter="Excel Files (*.xlsx *.xls)")
        if not xlsx_path:
            return
        # 3) Choose output root (storage folder root); per-row storage_folder can further specify a subfolder
        out_root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output root for batch export")
        if not out_root:
            return
        out_root_p = Path(out_root)
        # Optional storage folder name under the chosen output root
        storage_name, ok = QtWidgets.QInputDialog.getText(self, "Storage folder name", "Folder to create inside output root (e.g., session name):")
        if ok and str(storage_name).strip():
            # Ensure storage folder does not overwrite an existing one
            out_root_p = ensure_unique_folder(out_root_p / str(storage_name).strip())
        fmt = (self.cmb_raw_fmt.currentText() or "NPZ").lower()

        # Time base: simplified to UTC-only (no dialog)
        pol = "utc"

        progress_dialog = QtWidgets.QProgressDialog("Batch export in progress... (Excel times are interpreted as UTC)", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Batch Export")
        progress_dialog.setAutoClose(True); progress_dialog.setAutoReset(True); progress_dialog.setMinimumDuration(0)

        # Launch Excel-driven batch worker
        self._batch_worker = ExcelBatchWorker(Path(xlsx_path), self.root_dir, out_root_p, raw_fmt=fmt, time_base_policy=pol)
        self._batch_worker.progress.connect(progress_dialog.setValue)
        self._batch_worker.status.connect(progress_dialog.setLabelText)

        def _ok(n):
            progress_dialog.setValue(100)
            progress_dialog.close()
            QtWidgets.QMessageBox.information(self, "Batch finished", f"Measurements exported: {n}")

        def _err(msg):
            progress_dialog.close()
            QtWidgets.QMessageBox.critical(self, "Batch error", msg)

        self._batch_worker.finished_ok.connect(_ok)
        self._batch_worker.finished_err.connect(_err)

        def _cancel():
            try:
                if self._batch_worker and self._batch_worker.isRunning():
                    self._batch_worker.requestInterruption()
                    self._batch_worker.wait(5000)
            except Exception:
                pass
        progress_dialog.canceled.connect(_cancel)
        self._batch_worker.start()

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        try:
            for w in [getattr(self, '_selection_worker', None), self._rawall_worker, self._batch_worker]:
                try:
                    if w and w.isRunning():
                        w.requestInterruption()
                        w.wait(5000)
                except Exception:
                    continue
        except Exception:
            pass
        event.accept()

    # (Preview/metadata UI intentionally omitted)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    # Light grey UI theme
    try:
        QtWidgets.QApplication.setStyle("Fusion")
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(245, 245, 245))
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(33, 33, 33))
        pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255, 255, 255))
        pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(236, 236, 236))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 255))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(33, 33, 33))
        pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(33, 33, 33))
        pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(240, 240, 240))
        pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(33, 33, 33))
        pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))
        pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(76, 163, 224))
        pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
        app.setPalette(pal)
    except Exception:
        pass
    w = MainWindow()
    w.show()
    app.exec()
