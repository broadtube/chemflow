"""フローシートクラス：全装置・全ストリームを管理し、連立方程式として解く。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root

from chemflow.errors import SolveError


class Flowsheet:
    """プロセスフローシート。"""

    def __init__(self, name: str = "Flowsheet"):
        self.name = name
        self.streams: list = []
        self.units: list = []
        self.specs: list = []

    def add_stream(self, stream) -> None:
        if stream not in self.streams:
            self.streams.append(stream)

    def add_unit(self, unit) -> None:
        self.units.append(unit)

    def add_spec(self, spec_func) -> None:
        self.specs.append(spec_func)

    def _variable_streams(self) -> list:
        return [s for s in self.streams if not getattr(s, "_fixed", False)]

    def _pack(self) -> np.ndarray:
        arrays = [s.molar_flows for s in self._variable_streams()]
        if not arrays:
            return np.array([])
        return np.concatenate(arrays)

    def _unpack(self, x: np.ndarray) -> None:
        idx = 0
        for s in self._variable_streams():
            n = s.n_components
            s.molar_flows = x[idx : idx + n].copy()
            idx += n

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        self._unpack(x)
        res_list = []

        # 装置の残差式
        for unit in self.units:
            r = unit.residuals()
            if r is not None and len(r) > 0:
                res_list.append(np.atleast_1d(r))

        # 追加仕様式
        for spec in self.specs:
            r = spec()
            if r is not None:
                res_list.append(np.atleast_1d(r))

        # ストリームの組成制約
        for s in self.streams:
            for constraint in getattr(s, "_composition_constraints", []):
                r = constraint()
                if r is not None and len(r) > 0:
                    res_list.append(np.atleast_1d(r))

        if not res_list:
            return np.array([])
        return np.concatenate(res_list)

    def solve(self, **kwargs) -> dict:
        """連立方程式を求解する。"""
        x0 = self._pack()
        if len(x0) == 0:
            return {"success": True, "message": "No variables to solve"}

        n_vars = len(x0)
        n_residuals = len(self._residuals(x0))
        if n_vars != n_residuals:
            raise SolveError(
                f"System is {'over' if n_residuals > n_vars else 'under'}-determined: "
                f"{n_vars} variables, {n_residuals} equations"
            )

        result = root(self._residuals, x0, **kwargs)
        if result.success:
            self._unpack(result.x)
        else:
            raise SolveError(f"Solver did not converge: {result.message}")
        return result

    def fix_stream(self, stream) -> None:
        stream._fixed = True

    def set_stream_order(self, order: list[str]) -> None:
        """出力時のストリーム表示順序を設定する。

        Parameters
        ----------
        order : list[str]
            ストリーム名のリスト。リストにないストリームは末尾に登録順で追加される。
        """
        self._stream_order = order

    def set_component_order(self, order: list[str]) -> None:
        """出力時の成分表示順序を設定する。

        Parameters
        ----------
        order : list[str]
            示性式のリスト（例: ["H2", "CO", "CO2", "CH4", "H2O"]）
            リストにない成分は末尾に出現順で追加される。
        """
        self._component_order = order

    def _prepare_table_data(self):
        """テーブル表示用のデータを準備する。print_streams / export_csv で共用。"""
        streams = [s for s in self.streams if s.n_components > 0]
        if not streams:
            return None

        # ストリーム順序の適用
        if hasattr(self, "_stream_order") and self._stream_order:
            name_map = {(s.name or f"S{i+1}"): s for i, s in enumerate(streams)}
            ordered = [name_map[n] for n in self._stream_order if n in name_map]
            remaining = [s for s in streams if s not in ordered]
            streams = ordered + remaining

        all_set: set[str] = set()
        all_default: list[str] = []
        for s in streams:
            for c in s.components:
                if c.formula not in all_set:
                    all_default.append(c.formula)
                    all_set.add(c.formula)

        if hasattr(self, "_component_order") and self._component_order:
            # 指定された成分は全て表示（ストリームに存在しなくても0で表示）
            ordered = list(self._component_order)
            remaining = [f for f in all_default if f not in ordered]
            all_formulas = ordered + remaining
        else:
            all_formulas = all_default

        from chemflow.registry import ComponentRegistry
        mw_map = {f: ComponentRegistry.get(f).mw for f in all_formulas}

        def _get_values(stream, formulas):
            flow_map = {c.formula: i for i, c in enumerate(stream.components)}
            mol = np.zeros(len(formulas))
            mass = np.zeros(len(formulas))
            nvol = np.zeros(len(formulas))
            for i, f in enumerate(formulas):
                if f in flow_map:
                    j = flow_map[f]
                    mol[i] = stream.molar_flows[j]
                    mass[i] = stream.mass_flows[j]
                    nvol[i] = stream.normal_volume_flows[j]
            total_mol = mol.sum()
            total_mass = mass.sum()
            total_nvol = nvol.sum()
            return {
                "mol": mol, "mol_frac": mol / total_mol if total_mol else np.zeros_like(mol), "total_mol": total_mol,
                "mass": mass, "mass_frac": mass / total_mass if total_mass else np.zeros_like(mass), "total_mass": total_mass,
                "nvol": nvol, "vol_frac": nvol / total_nvol if total_nvol else np.zeros_like(nvol), "total_nvol": total_nvol,
            }

        data = [_get_values(s, all_formulas) for s in streams]
        names = [s.name or f"S{i+1}" for i, s in enumerate(streams)]

        # 圧力をMPaGに変換
        def _to_mpag(s):
            p = getattr(s, "P_input", None)
            if p is None:
                return ""
            if isinstance(p, str):
                from chemflow.gibbs import parse_pressure
                p_pa = parse_pressure(p)
            else:
                p_pa = float(p)
            return f"{(p_pa - 101325) / 1e6:.3f}"

        pressures = [_to_mpag(s) for s in streams]
        temperatures = [f"{s.T_celsius:.1f}" if getattr(s, "T_celsius", None) is not None else "" for s in streams]
        phases = [getattr(s, "phase", None) or "" for s in streams]

        return {
            "streams": streams,
            "all_formulas": all_formulas,
            "mw_map": mw_map,
            "data": data,
            "names": names,
            "pressures": pressures,
            "temperatures": temperatures,
            "phases": phases,
        }

    def print_streams(self) -> None:
        """全ストリームを横並びの統合テーブルで表示する。"""
        t = self._prepare_table_data()
        if t is None:
            return

        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]

        fw = max(max(len(f) for f in all_formulas), 9)  # "Component" 幅確保
        mw_w, abs_w, rel_w = 8, 10, 8
        stream_w = abs_w + rel_w + 1
        lbl_w = fw + 2 + mw_w  # 左ラベル幅

        sections = [
            ("mol",    "mol/h",  "mol%",  "mol",  "mol_frac",  "total_mol"),
            ("Volume", "NL/h",   "vol%",  "nvol", "vol_frac",  "total_nvol"),
            ("weight", "g/h",    "wt%",   "mass", "mass_frac", "total_mass"),
        ]

        def _header_row(label, values, stream_values=None):
            """label: 1列目, values: 2列目がvalues[0]の場合はMW列に使用しそれ以降がストリーム列。
            stream_values が指定された場合: values はMW列用、stream_values がストリーム列用。"""
            if stream_values is not None:
                row = f"  {label:>{fw}s}  {values[0]:>{mw_w}s}" if values else f"  {label:>{fw}s}  {'':>{mw_w}s}"
                for v in stream_values:
                    row += f"  {v:^{stream_w}s}"
            else:
                row = f"  {label:>{fw}s}  {'':>{mw_w}s}"
                for v in values:
                    row += f"  {v:^{stream_w}s}"
            return row

        first_section = True
        for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in sections:
            if first_section:
                print(_header_row("No.", [str(i+1) for i in range(len(names))]))
                print(_header_row("Service", names))
                print(_header_row("Press.", ["MPaG"], pressures))
                print(_header_row("Temp.", ["°C"], temperatures))
                print(_header_row("Phase", phases))
                first_section = False
            # Component/MW + 単位行
            h2 = f"  {'Component':>{fw}s}  {'MW':>{mw_w}s}"
            for _ in names:
                h2 += f"  {abs_unit:>{abs_w}s} {rel_unit:>{rel_w}s}"
            print(h2)
            # 区切り線
            sep = f"{'':>{fw}s}  {'-' * mw_w}"
            for _ in names:
                sep += f"  {'-' * abs_w} {'-' * rel_w}"
            print(sep)
            # 成分行
            for i, f in enumerate(all_formulas):
                row = f"  {f:>{fw}s}  {mw_map[f]:{mw_w}.2f}"
                for d in data:
                    row += f"  {d[abs_key][i]:{abs_w}.4f} {d[rel_key][i]:{rel_w}.4f}"
                print(row)
            # Total行
            row = f"  {'Total':>{fw}s}  {'':>{mw_w}s}"
            for d in data:
                t_val = d[total_key]
                r_val = "1.0000" if abs(t_val) > 1e-10 else "0.0000"
                row += f"  {t_val:{abs_w}.4f} {r_val:>{rel_w}s}"
            print(row)

    def export_csv(self, path: str) -> None:
        """全ストリームの結果をCSVファイルに出力する。

        Parameters
        ----------
        path : str
            出力先CSVファイルパス
        """
        import csv as csv_mod

        t = self._prepare_table_data()
        if t is None:
            return

        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]

        sections = [
            ("mol",    "mol/h",  "mol%",  "mol",  "mol_frac",  "total_mol"),
            ("Volume", "NL/h",   "vol%",  "nvol", "vol_frac",  "total_nvol"),
            ("weight", "g/h",    "wt%",   "mass", "mass_frac", "total_mass"),
        ]

        def _csv_header_row(label, mw_val, values):
            row = [label, mw_val]
            for v in values:
                row.extend([v, ""])
            return row

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)

            # ヘッダーブロック
            w.writerow(_csv_header_row("No.", "", [str(i+1) for i in range(len(names))]))
            w.writerow(_csv_header_row("Service", "", names))
            w.writerow(_csv_header_row("Press.", "MPaG", pressures))
            w.writerow(_csv_header_row("Temp.", "°C", temperatures))
            w.writerow(_csv_header_row("Phase", "", phases))

            for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in sections:
                # Component/MW + 単位行
                unit_row = ["Component", "MW"]
                for _ in names:
                    unit_row.extend([abs_unit, rel_unit])
                w.writerow(unit_row)

                # 成分行
                for i, formula in enumerate(all_formulas):
                    row = [formula, f"{mw_map[formula]:.2f}"]
                    for d in data:
                        row.extend([f"{d[abs_key][i]:.4f}", f"{d[rel_key][i]:.4f}"])
                    w.writerow(row)

                # Total行
                total_row = ["Total", ""]
                for d in data:
                    t_val = d[total_key]
                    r_val = "1.0000" if abs(t_val) > 1e-10 else "0.0000"
                    total_row.extend([f"{t_val:.4f}", r_val])
                w.writerow(total_row)

    def export_excel(self, filename: str, sheet: str, cell: str = "A1") -> None:
        """開いている Excel ブックのシートに結果を出力する。

        Parameters
        ----------
        filename : str
            開いている Excel ファイル名（例: "result.xlsx"）
            フルパスまたはファイル名のみ。
        sheet : str
            出力先シート名（既存シートのみ）
        cell : str
            出力開始セル（例: "A1", "C3"）
        """
        try:
            import win32com.client
        except ImportError:
            raise RuntimeError(
                "win32com が必要です。Windows 環境で pip install pywin32 を実行してください。"
            )

        t = self._prepare_table_data()
        if t is None:
            raise ValueError("出力するストリームデータがありません。")

        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]

        # Excel アプリケーションを取得
        try:
            xl = win32com.client.GetActiveObject("Excel.Application")
        except Exception:
            raise RuntimeError("Excel が起動していません。Excel を開いてから実行してください。")

        # ワークブックを検索
        wb = None
        for i in range(1, xl.Workbooks.Count + 1):
            w = xl.Workbooks(i)
            if w.Name == filename or w.FullName == filename:
                wb = w
                break
        if wb is None:
            available = [xl.Workbooks(i).Name for i in range(1, xl.Workbooks.Count + 1)]
            raise ValueError(
                f"ファイル '{filename}' が開かれていません。"
                f" 開いているファイル: {available}"
            )

        # シートを検索
        ws = None
        for i in range(1, wb.Sheets.Count + 1):
            s = wb.Sheets(i)
            if s.Name == sheet:
                ws = s
                break
        if ws is None:
            available = [wb.Sheets(i).Name for i in range(1, wb.Sheets.Count + 1)]
            raise ValueError(
                f"シート '{sheet}' が存在しません。"
                f" 存在するシート: {available}"
            )

        # 開始セルの解析
        start = ws.Range(cell)
        row0 = start.Row
        col0 = start.Column

        sections = [
            ("mol",    "mol/h",  "mol%",  "mol",  "mol_frac",  "total_mol"),
            ("Volume", "NL/h",   "vol%",  "nvol", "vol_frac",  "total_nvol"),
            ("weight", "g/h",    "wt%",   "mass", "mass_frac", "total_mass"),
        ]

        r = row0  # 現在の行

        def _xl_header_row(label, mw_val, values):
            nonlocal r
            ws.Cells(r, col0).Value = label
            if mw_val:
                ws.Cells(r, col0 + 1).Value = mw_val
            for si, v in enumerate(values):
                ws.Cells(r, col0 + 2 + si * 2).Value = v
            r += 1

        _xl_header_row("No.", "", [str(i+1) for i in range(len(names))])
        _xl_header_row("Service", "", names)
        _xl_header_row("Press.", "MPaG", pressures)
        _xl_header_row("Temp.", "°C", temperatures)
        _xl_header_row("Phase", "", phases)

        for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in sections:
            # Component/MW + 単位行
            ws.Cells(r, col0).Value = "Component"
            ws.Cells(r, col0 + 1).Value = "MW"
            for si in range(len(names)):
                ws.Cells(r, col0 + 2 + si * 2).Value = abs_unit
                ws.Cells(r, col0 + 3 + si * 2).Value = rel_unit
            r += 1

            # 成分行
            for i, formula in enumerate(all_formulas):
                ws.Cells(r, col0).Value = formula
                ws.Cells(r, col0 + 1).Value = round(mw_map[formula], 2)
                for si, d in enumerate(data):
                    ws.Cells(r, col0 + 2 + si * 2).Value = round(d[abs_key][i], 4)
                    ws.Cells(r, col0 + 3 + si * 2).Value = round(d[rel_key][i], 4)
                r += 1

            # Total行
            ws.Cells(r, col0).Value = "Total"
            for si, d in enumerate(data):
                t_val = d[total_key]
                ws.Cells(r, col0 + 2 + si * 2).Value = round(t_val, 4)
                ws.Cells(r, col0 + 3 + si * 2).Value = 1.0 if abs(t_val) > 1e-10 else 0.0
            r += 1

    def generate_mermaid(self) -> str:
        """Flowsheet からMermaid フロー図を自動生成する。"""
        from chemflow.units import Mixer, Splitter, Reactor, MultiReactor, Absorber
        from chemflow.gibbs import GibbsReactor

        lines = ["graph LR"]
        stream_ids: dict[int, str] = {}  # id(stream) → mermaid node id
        unit_counter = 0

        def _sid(stream) -> str:
            """ストリームの Mermaid ノード ID を取得/生成。"""
            key = id(stream)
            if key not in stream_ids:
                name = stream.name or f"S{len(stream_ids)+1}"
                safe = name.replace(" ", "_").replace("-", "_")
                stream_ids[key] = safe
            return stream_ids[key]

        def _slabel(stream) -> str:
            """ストリームノードのラベルを生成。"""
            name = stream.name or f"S{len(stream_ids)}"
            parts = [name]
            T = getattr(stream, "T_celsius", None)
            P = getattr(stream, "P_input", None)
            phase = getattr(stream, "phase", None)
            if T is not None:
                parts.append(f"{T}°C")
            if P is not None:
                parts.append(str(P))
            if phase:
                parts.append(phase)
            # 固定ストリームの流量情報
            if stream._fixed:
                total = stream.total_molar_flow
                if abs(total) > 1e-10:
                    mass = stream.total_mass_flow
                    nvol = stream.total_normal_volume_flow
                    parts.append(f"{total:.2f} mol/h")
                    parts.append(f"{nvol:.2f} NL/h")
                    parts.append(f"{mass:.2f} g/h")
                else:
                    parts.append("(0)")
            return "\\n".join(parts)

        # ストリームノード定義
        for i, s in enumerate(self.streams):
            sid = _sid(s)
            label = _slabel(s)
            lines.append(f'    {sid}["{label}"]')

        # 各ユニットの outlet を収集（eq 経由の Mixer 検出用）
        unit_outlets: set[int] = set()
        for unit in self.units:
            for attr in ("outlet", "gas_outlet", "liquid_outlet", "water_outlet"):
                out = getattr(unit, attr, None)
                if out is not None:
                    unit_outlets.add(id(out))

        # ユニットからエッジを生成
        for unit in self.units:
            unit_counter += 1
            uid = f"U{unit_counter}"

            if isinstance(unit, Mixer):
                # outlet が他のユニットの outlet でもある場合 → eq 経由（実質 Splitter）
                is_eq_mixer = any(
                    id(getattr(u2, attr, None)) == id(unit.outlet)
                    for u2 in self.units if u2 is not unit
                    for attr in ("outlet", "gas_outlet", "liquid_outlet", "water_outlet")
                )
                if is_eq_mixer:
                    # solve 後の実際の分割比を計算
                    total_out = unit.outlet.total_molar_flow
                    ratios = []
                    for inlet in unit.inlets:
                        r = inlet.total_molar_flow / total_out if abs(total_out) > 1e-10 else 0
                        ratios.append(f"{r*100:.1f}%")
                    label = "Splitter\\n" + " / ".join(ratios)
                    lines.append(f'    {uid}(("{label}"))')
                    lines.append(f"    {_sid(unit.outlet)} --> {uid}")
                    for inlet in unit.inlets:
                        lines.append(f"    {uid} --> {_sid(inlet)}")
                else:
                    label = "Mixer"
                    lines.append(f'    {uid}(("{label}"))')
                    for inlet in unit.inlets:
                        lines.append(f"    {_sid(inlet)} --> {uid}")
                    lines.append(f"    {uid} --> {_sid(unit.outlet)}")

            elif isinstance(unit, Splitter):
                ratio_strs = [f"{r*100:.1f}%" for r in unit.ratios]
                label = "Splitter\\n" + " / ".join(ratio_strs)
                lines.append(f'    {uid}(("{label}"))')
                lines.append(f"    {_sid(unit.inlet)} --> {uid}")
                for outlet in unit.outlets:
                    lines.append(f"    {uid} --> {_sid(outlet)}")

            elif isinstance(unit, MultiReactor):
                rxns = len(unit.reactions)
                conv = unit.conversion
                label = f"Reactor\\n{rxns}反応\\nconv {conv*100:.0f}%"
                lines.append(f'    {uid}(("{label}"))')
                lines.append(f"    {_sid(unit.inlet)} --> {uid}")
                lines.append(f"    {uid} --> {_sid(unit.outlet)}")

            elif isinstance(unit, Reactor):
                conv = unit.conversion
                label = f"Reactor\\nconv {conv*100:.0f}%"
                lines.append(f'    {uid}(("{label}"))')
                lines.append(f"    {_sid(unit.inlet)} --> {uid}")
                lines.append(f"    {uid} --> {_sid(unit.outlet)}")

            elif isinstance(unit, GibbsReactor):
                T = unit.T_kelvin - 273.15
                label = f"Gibbs\\n{T:.0f}°C"
                lines.append(f'    {uid}(("{label}"))')
                lines.append(f"    {_sid(unit.inlet)} --> {uid}")
                lines.append(f"    {uid} --> {_sid(unit.outlet)}")

            elif isinstance(unit, Absorber):
                N = unit.stages
                T = unit.T_celsius
                label = f"Absorber\\n{N}段 {T:.0f}°C"
                lines.append(f'    {uid}(("{label}"))')
                lines.append(f"    {_sid(unit.gas_inlet)} --> {uid}")
                lines.append(f"    {_sid(unit.water_inlet)} --> {uid}")
                lines.append(f"    {uid} --> {_sid(unit.gas_outlet)}")
                lines.append(f"    {uid} --> {_sid(unit.liquid_outlet)}")

            else:
                # WaterSeparator 等
                label = type(unit).__name__
                lines.append(f'    {uid}(("{label}"))')
                if hasattr(unit, "inlet"):
                    lines.append(f"    {_sid(unit.inlet)} --> {uid}")
                if hasattr(unit, "gas_outlet"):
                    lines.append(f"    {uid} --> {_sid(unit.gas_outlet)}")
                if hasattr(unit, "water_outlet"):
                    lines.append(f"    {uid} --> {_sid(unit.water_outlet)}")

        # 制約ラベルを注釈ノードとして追加
        labels = getattr(self, "_constraint_labels", [])
        if labels:
            lines.append("")
            lines.append('    CONSTRAINTS["Constraints:\\n' + "\\n".join(labels) + '"]')
            lines.append("    style CONSTRAINTS fill:#ffffcc,stroke:#cccc00")

        return "\n".join(lines)

    def export_mermaid(self, path: str, title: str | None = None, description: str | None = None) -> None:
        """Mermaid フロー図を HTML ファイルとして出力する。

        Parameters
        ----------
        path : str
            出力先 HTML ファイルパス
        title : str | None
            タイトル（省略時は Flowsheet名）
        description : str | None
            説明文
        """
        mermaid_code = self.generate_mermaid()
        t = title or self.name
        desc_html = f'<p class="desc">{description}</p>' if description else ""

        # ストリームサマリー表を生成
        streams = [s for s in self.streams if s.n_components > 0]
        if hasattr(self, "_stream_order") and self._stream_order:
            name_map = {(s.name or f"S{i+1}"): s for i, s in enumerate(streams)}
            ordered = [name_map[n] for n in self._stream_order if n in name_map]
            remaining = [s for s in streams if s not in ordered]
            streams = ordered + remaining

        table_rows = ""
        for i, s in enumerate(streams):
            name = s.name or f"S{i+1}"
            T = f"{s.T_celsius}°C" if getattr(s, "T_celsius", None) is not None else ""
            P = str(s.P_input) if getattr(s, "P_input", None) is not None else ""
            phase = getattr(s, "phase", None) or ""
            mol = f"{s.total_molar_flow:.2f}" if abs(s.total_molar_flow) > 1e-10 else "0"
            nvol = f"{s.total_normal_volume_flow:.2f}" if abs(s.total_normal_volume_flow) > 1e-10 else "0"
            mass = f"{s.total_mass_flow:.2f}" if abs(s.total_mass_flow) > 1e-10 else "0"
            table_rows += f"<tr><td>{i+1}</td><td>{name}</td><td>{T}</td><td>{P}</td><td>{phase}</td><td>{mol}</td><td>{nvol}</td><td>{mass}</td></tr>\n"

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>{t} - chemflow</title>
<style>
  body {{ font-family: sans-serif; margin: 40px; background: #f9f9f9; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ccc; padding-bottom: 8px; }}
  .desc {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
  .mermaid {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
  table {{ border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
  th {{ background: #f0f0f0; text-align: center; }}
  td:nth-child(1), td:nth-child(2), td:nth-child(5) {{ text-align: left; }}
</style>
</head>
<body>

<h1>{t}</h1>
{desc_html}

<h2>Flow Diagram</h2>
<div class="mermaid">
{mermaid_code}
</div>

<h2>Stream Summary</h2>
<table>
<tr><th>No.</th><th>Service</th><th>Temp.</th><th>Press.</th><th>Phase</th><th>mol/h</th><th>NL/h</th><th>g/h</th></tr>
{table_rows}</table>

<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
