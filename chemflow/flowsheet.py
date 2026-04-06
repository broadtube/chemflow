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
        """連立方程式を求解する。

        収束しない場合、自動的に複数のソルバーメソッドを試行する。
        method を明示的に指定した場合はフォールバックしない。
        """
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

        # method が明示指定されている場合はフォールバックしない
        if 'method' in kwargs:
            result = root(self._residuals, x0, **kwargs)
            if result.success:
                self._unpack(result.x)
            else:
                raise SolveError(f"Solver did not converge: {result.message}")
            return result

        # 自動フォールバック: 複数のソルバーを試行
        methods = ['hybr', 'lm', 'broyden1', 'df-sane']
        last_result = None

        for method in methods:
            try:
                result = root(self._residuals, x0, method=method, **kwargs)
                if result.success:
                    self._unpack(result.x)
                    return result
                last_result = result
            except Exception:
                continue

        # 全て失敗した場合
        msg = last_result.message if last_result else "All solver methods failed"
        raise SolveError(f"Solver did not converge: {msg}")
        return last_result

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
                    parts.append(f"{stream.total_mass_flow:.1f} g/h")
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
        labels = [l for l in getattr(self, "_constraint_labels", []) if l]
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

    def generate_json(self) -> dict:
        """Flowsheet を JSON シリアライズ可能な dict として出力する。"""
        import json as json_mod
        from chemflow.units import Mixer, Splitter, Reactor, MultiReactor, Absorber
        from chemflow.gibbs import GibbsReactor

        streams = list(self.streams)
        if hasattr(self, "_stream_order") and self._stream_order:
            name_map = {(s.name or f"S{i+1}"): s for i, s in enumerate(streams)}
            ordered = [name_map[n] for n in self._stream_order if n in name_map]
            remaining = [s for s in streams if s not in ordered]
            streams = ordered + remaining

        stream_id_map = {id(s): (s.name or f"S{i+1}") for i, s in enumerate(streams)}

        def _sid(s):
            return stream_id_map.get(id(s), "?")

        # ストリーム
        json_streams = []
        for i, s in enumerate(streams):
            sid = _sid(s)
            comp_flows = {c.formula: round(float(s.molar_flows[j]), 6) for j, c in enumerate(s.components)}
            entry = {
                "id": sid, "name": s.name, "index": i + 1,
                "T_celsius": getattr(s, "T_celsius", None),
                "P_input": str(s.P_input) if getattr(s, "P_input", None) is not None else None,
                "phase": getattr(s, "phase", None),
                "fixed": s._fixed,
                "total_mol": round(float(s.total_molar_flow), 4),
                "total_NL": round(float(s.total_normal_volume_flow), 4),
                "total_g": round(float(s.total_mass_flow), 4),
                "components": comp_flows,
                "original_components": sorted(s._original_formulas) if getattr(s, "_original_formulas", None) else None,
                "has_composition_constraints": len(getattr(s, "_composition_constraints", [])) > 0,
            }
            json_streams.append(entry)

        # ユニット
        json_units = []
        unit_counter = 0
        for unit in self.units:
            unit_counter += 1
            uid = f"U{unit_counter}"
            entry = {"id": uid, "type": type(unit).__name__}

            if isinstance(unit, Mixer):
                is_eq = any(
                    id(getattr(u2, attr, None)) == id(unit.outlet)
                    for u2 in self.units if u2 is not unit
                    for attr in ("outlet", "gas_outlet", "liquid_outlet", "water_outlet")
                )
                if is_eq:
                    entry["type"] = "Splitter (eq)"
                    total_out = unit.outlet.total_molar_flow
                    ratios = []
                    for inlet in unit.inlets:
                        r = inlet.total_molar_flow / total_out if abs(total_out) > 1e-10 else 0
                        ratios.append(round(r, 4))
                    entry["source"] = _sid(unit.outlet)
                    entry["targets"] = [_sid(s) for s in unit.inlets]
                    entry["ratios"] = ratios
                else:
                    entry["sources"] = [_sid(s) for s in unit.inlets]
                    entry["target"] = _sid(unit.outlet)

            elif isinstance(unit, Splitter):
                entry["source"] = _sid(unit.inlet)
                entry["targets"] = [_sid(s) for s in unit.outlets]
                entry["ratios"] = [round(float(r), 4) for r in unit.ratios]

            elif isinstance(unit, MultiReactor):
                entry["source"] = _sid(unit.inlet)
                entry["target"] = _sid(unit.outlet)
                entry["reactions"] = unit.reactions
                entry["key"] = unit.key
                entry["conversion"] = unit.conversion
                entry["selectivities"] = unit.selectivities

            elif isinstance(unit, Reactor):
                entry["source"] = _sid(unit.inlet)
                entry["target"] = _sid(unit.outlet)
                entry["conversion"] = unit.conversion

            elif isinstance(unit, GibbsReactor):
                entry["source"] = _sid(unit.inlet)
                entry["target"] = _sid(unit.outlet)
                entry["T_celsius"] = unit.T_kelvin - 273.15
                entry["P_pascal"] = unit.P_pascal
                entry["species"] = unit.species

            elif isinstance(unit, Absorber):
                entry["gas_inlet"] = _sid(unit.gas_inlet)
                entry["water_inlet"] = _sid(unit.water_inlet)
                entry["gas_outlet"] = _sid(unit.gas_outlet)
                entry["liquid_outlet"] = _sid(unit.liquid_outlet)
                entry["T_celsius"] = unit.T_celsius
                entry["stages"] = unit.stages

            else:
                if hasattr(unit, "inlet"):
                    entry["source"] = _sid(unit.inlet)
                if hasattr(unit, "gas_outlet"):
                    entry["gas_outlet"] = _sid(unit.gas_outlet)
                if hasattr(unit, "water_outlet"):
                    entry["water_outlet"] = _sid(unit.water_outlet)

            json_units.append(entry)

        # 制約
        labels = getattr(self, "_constraint_labels", [])
        codes = getattr(self, "_constraint_codes", [])
        json_constraints = []
        for i in range(max(len(labels), len(codes))):
            json_constraints.append({
                "label": labels[i] if i < len(labels) else "",
                "code": codes[i] if i < len(codes) else "",
            })

        return {
            "name": self.name,
            "streams": json_streams,
            "units": json_units,
            "constraints": [c["label"] for c in json_constraints if c["label"]],
            "constraint_specs": json_constraints,
            "component_order": getattr(self, "_component_order", None),
            "stream_order": getattr(self, "_stream_order", None),
        }

    def export_json(self, path: str) -> None:
        """Flowsheet を JSON ファイルとして出力する。"""
        import json as json_mod
        data = self.generate_json()
        with open(path, "w", encoding="utf-8") as f:
            json_mod.dump(data, f, indent=2, ensure_ascii=False)

    def export_reactflow(self, path: str, title: str | None = None, description: str | None = None) -> None:
        """ReactFlow v11 UMD + dagre によるインタラクティブフロー図を HTML として出力する。"""
        import json as json_mod
        data = self.generate_json()
        t = title or self.name
        desc = description or ""
        json_str = json_mod.dumps(data, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>{t} - chemflow</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reactflow@11.11.4/dist/umd/index.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reactflow@11.11.4/dist/style.css">
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #header {{ padding: 12px 24px; background: #f5f5f5; border-bottom: 1px solid #ddd; z-index: 10; position: relative; }}
  #header h1 {{ margin: 0 0 3px 0; font-size: 18px; color: #333; }}
  #header p {{ margin: 0; font-size: 12px; color: #666; }}
  #root {{ width: 100vw; height: calc(100vh - 55px); }}
  .cf-stream {{ padding: 8px 12px; border-radius: 6px; border: 2px solid #555; background: #fff; font-size: 11px; min-width: 110px; }}
  .cf-stream.fixed {{ border-color: #1976D2; background: #E3F2FD; }}
  .cf-stream.variable {{ border-color: #388E3C; background: #E8F5E9; }}
  .cf-stream.zero {{ border-color: #9E9E9E; background: #F5F5F5; color: #999; }}
  .cf-stream .sname {{ font-weight: bold; font-size: 12px; margin-bottom: 3px; }}
  .cf-stream .sinfo {{ color: #555; font-size: 10px; line-height: 1.4; }}
  .cf-unit {{ padding: 6px 10px; border-radius: 18px; border: 2px solid #F57C00; background: #FFF3E0; font-size: 11px; text-align: center; min-width: 90px; }}
  .cf-constraint {{ padding: 6px 10px; border-radius: 4px; border: 2px dashed #999; background: #FFFDE7; font-size: 10px; }}
  #detail {{ position: fixed; top: 55px; right: 0; width: 320px; height: calc(100vh - 55px); background: #fff; border-left: 1px solid #ddd; overflow-y: auto; padding: 16px; font-size: 12px; box-shadow: -2px 0 8px rgba(0,0,0,0.1); display: none; z-index: 20; }}
  #detail.open {{ display: block; }}
  #detail h3 {{ margin: 0 0 8px; font-size: 15px; }}
  #detail .close {{ position: absolute; top: 8px; right: 12px; cursor: pointer; font-size: 18px; color: #999; }}
  #detail table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  #detail th, #detail td {{ border: 1px solid #eee; padding: 3px 6px; text-align: right; font-size: 11px; }}
  #detail th {{ background: #f5f5f5; text-align: left; }}
  #toolbar {{ position: absolute; bottom: 12px; right: 12px; z-index: 15; }}
  #toolbar button {{ padding: 6px 14px; font-size: 12px; background: #1976D2; color: #fff; border: none; border-radius: 4px; cursor: pointer; margin-left: 6px; }}
  #toolbar button:hover {{ background: #1565C0; }}
  #edit-modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.4); z-index:100; }}
  #edit-modal.open {{ display:flex; justify-content:center; align-items:center; }}
  #edit-box {{ background:#fff; border-radius:8px; padding:20px; min-width:400px; max-width:600px; max-height:80vh; overflow-y:auto; box-shadow:0 4px 20px rgba(0,0,0,0.3); }}
  #edit-box h3 {{ margin:0 0 12px; }}
  #edit-box label {{ display:block; margin:6px 0 2px; font-size:12px; color:#555; }}
  #edit-box input, #edit-box select {{ width:100%; padding:4px 8px; font-size:12px; border:1px solid #ccc; border-radius:3px; box-sizing:border-box; }}
  #edit-box .comp-row {{ display:flex; gap:8px; align-items:center; margin:2px 0; }}
  #edit-box .comp-row input {{ flex:1; }}
  #edit-box .comp-row span {{ width:60px; font-size:11px; }}
  #edit-box .btn-row {{ margin-top:12px; text-align:right; }}
  #edit-box button {{ padding:6px 16px; font-size:12px; border:none; border-radius:4px; cursor:pointer; margin-left:6px; }}
  #edit-box .btn-save {{ background:#388E3C; color:#fff; }}
  #edit-box .btn-cancel {{ background:#999; color:#fff; }}
  #ctx-menu {{ display:none; position:fixed; background:#fff; border:1px solid #ccc; border-radius:4px; box-shadow:0 2px 8px rgba(0,0,0,0.15); z-index:50; font-size:12px; }}
  #ctx-menu div {{ padding:6px 16px; cursor:pointer; }}
  #ctx-menu div:hover {{ background:#E3F2FD; }}
</style>
</head>
<body>
<div id="header"><h1>{t}</h1><p>{desc}</p></div>
<div id="root"></div>
<div id="detail"><span class="close" onclick="this.parentElement.classList.remove('open')">&times;</span><div id="detail-content"></div></div>
<div id="toolbar">
  <button onclick="exportJSON()">Export JSON</button>
  <button onclick="exportModified()">Export Modified JSON</button>
  <button onclick="exportLayout()">Export Layout</button>
</div>
<div id="edit-modal" onclick="if(event.target===this)closeEdit()"><div id="edit-box"></div></div>
<div id="ctx-menu"></div>
<script>
const D = {json_str};
const {{ default: RF, Background, Controls, MiniMap, Handle, Position, applyNodeChanges, applyEdgeChanges }} = window.ReactFlow;
const {{ useState, useCallback }} = React;
const e = React.createElement;

// --- Build nodes & edges from JSON ---
const rfNodes = [];
const rfEdges = [];
const NW=170, NH=90, UW=140, UH=65;

D.streams.forEach(s => {{
  let cls = 'variable';
  if (s.fixed && Math.abs(s.total_mol) < 1e-10) cls = 'zero';
  else if (s.fixed) cls = 'fixed';
  let info = [];
  if (s.T_celsius !== null) info.push(s.T_celsius + '°C');
  if (s.P_input) info.push(s.P_input);
  if (s.phase) info.push(s.phase);
  if (s.fixed && Math.abs(s.total_mol) > 1e-10) info.push(s.total_g.toFixed(1) + ' g/h');
  else if (s.fixed) info.push('(0)');
  rfNodes.push({{ id: s.id, type: 'stream', position: {{x:0,y:0}},
    data: {{ label: s.index+'. '+s.id, info: info.join(' | '), cls }} }});
}});

D.units.forEach(u => {{
  let lines = [u.type];
  if (u.type==='MultiReactor') lines=['Reactor',(u.reactions||[]).length+'反応','conv '+((u.conversion||0)*100).toFixed(0)+'%'];
  else if (u.type==='Reactor') lines=['Reactor','conv '+((u.conversion||0)*100).toFixed(0)+'%'];
  else if (u.type==='GibbsReactor') lines=['Gibbs',(u.T_celsius||'')+'°C'];
  else if (u.type==='Absorber') lines=['Absorber',(u.stages||'')+'段 '+(u.T_celsius||'')+'°C'];
  else if (u.type==='Splitter (eq)'||u.type==='Splitter') lines=['Splitter',(u.ratios||[]).map(r=>(r*100).toFixed(1)+'%').join(' / ')];
  else if (u.type==='WaterSeparator') lines=['Water Sep'];
  rfNodes.push({{ id: u.id, type: 'unit', position: {{x:0,y:0}}, data: {{ lines }} }});

  const addE = (from, to) => rfEdges.push({{ id: from+'_'+to, source: from, target: to, type: 'smoothstep', style: {{ strokeWidth: 2 }} }});
  if (u.type==='Mixer') {{ (u.sources||[]).forEach(s=>addE(s,u.id)); if(u.target) addE(u.id,u.target); }}
  else if (u.type==='Splitter (eq)'||u.type==='Splitter') {{ if(u.source) addE(u.source,u.id); (u.targets||[]).forEach(t=>addE(u.id,t)); }}
  else if (u.type==='Absorber') {{ if(u.gas_inlet) addE(u.gas_inlet,u.id); if(u.water_inlet) addE(u.water_inlet,u.id); if(u.gas_outlet) addE(u.id,u.gas_outlet); if(u.liquid_outlet) addE(u.id,u.liquid_outlet); }}
  else if (u.type==='WaterSeparator') {{ if(u.source) addE(u.source,u.id); if(u.gas_outlet) addE(u.id,u.gas_outlet); if(u.water_outlet) addE(u.id,u.water_outlet); }}
  else {{ if(u.source) addE(u.source,u.id); if(u.target) addE(u.id,u.target); }}
}});

const cL = (D.constraints||[]).filter(c=>c);
if (cL.length) rfNodes.push({{ id:'CONSTRAINTS', type:'constraint', position:{{x:0,y:0}}, data:{{ lines: ['Constraints:'].concat(cL) }} }});

// --- Dagre layout ---
const g = new dagre.graphlib.Graph();
g.setGraph({{ rankdir:'LR', nodesep:45, ranksep:80 }});
g.setDefaultEdgeLabel(()=>({{}}));
rfNodes.forEach(n => {{
  const w = n.type==='unit'? UW : (n.type==='constraint'? 190 : NW);
  const h = n.type==='unit'? UH : (n.type==='constraint'? 20+(n.data.lines||[]).length*15 : NH);
  g.setNode(n.id, {{width:w, height:h}});
}});
rfEdges.forEach(ed => g.setEdge(ed.source, ed.target));
dagre.layout(g);
rfNodes.forEach(n => {{ const p=g.node(n.id); n.position = {{ x: p.x-p.width/2, y: p.y-p.height/2 }}; }});

// --- Custom node components ---
function StreamNode({{ data }}) {{
  return e('div', {{ className: 'cf-stream ' + data.cls }},
    e(Handle, {{ type:'target', position: Position.Left, style:{{background:'#555'}} }}),
    e('div', {{ className:'sname' }}, data.label),
    e('div', {{ className:'sinfo' }}, data.info),
    e(Handle, {{ type:'source', position: Position.Right, style:{{background:'#555'}} }})
  );
}}
function UnitNode({{ data }}) {{
  return e('div', {{ className: 'cf-unit' }},
    e(Handle, {{ type:'target', position: Position.Left, style:{{background:'#F57C00'}} }}),
    ...(data.lines||[]).map((l,i) => e('div', {{ key:i, style:{{ fontWeight: i===0?'bold':'normal' }} }}, l)),
    e(Handle, {{ type:'source', position: Position.Right, style:{{background:'#F57C00'}} }})
  );
}}
function ConstraintNode({{ data }}) {{
  return e('div', {{ className: 'cf-constraint' }},
    ...(data.lines||[]).map((l,i) => e('div', {{ key:i, style:{{ fontWeight: i===0?'bold':'normal' }} }}, l))
  );
}}

const nodeTypes = {{ stream: StreamNode, unit: UnitNode, constraint: ConstraintNode }};

// --- App ---
// --- Stream lookup ---
const streamMap = {{}};
D.streams.forEach(s => {{ streamMap[s.id] = s; }});
const unitMap = {{}};
D.units.forEach(u => {{ unitMap[u.id] = u; }});

function showDetail(nodeId) {{
  const panel = document.getElementById('detail');
  const content = document.getElementById('detail-content');
  const s = streamMap[nodeId];
  const u = unitMap[nodeId];
  if (s) {{
    let h = '<h3>' + s.index + '. ' + s.id + '</h3>';
    h += '<div style="color:#666;margin-bottom:8px">';
    if (s.T_celsius !== null) h += s.T_celsius + '°C | ';
    if (s.P_input) h += s.P_input + ' | ';
    if (s.phase) h += s.phase;
    h += '</div>';
    h += '<div>Fixed: ' + (s.fixed ? 'Yes' : 'No') + '</div>';
    h += '<div>Total: ' + s.total_mol.toFixed(4) + ' mol/h | ' + s.total_NL.toFixed(2) + ' NL/h | ' + s.total_g.toFixed(2) + ' g/h</div>';
    h += '<table><tr><th>Component</th><th>mol/h</th><th>mol%</th></tr>';
    const comps = s.components || {{}};
    const total = s.total_mol || 1;
    for (const [k, v] of Object.entries(comps)) {{
      if (Math.abs(v) > 1e-8) {{
        h += '<tr><td style="text-align:left">' + k + '</td><td>' + v.toFixed(4) + '</td><td>' + (v/total*100).toFixed(2) + '%</td></tr>';
      }}
    }}
    h += '</table>';
    if (s.fixed) {{
      h += '<div style="margin-top:10px"><button onclick="openEdit(\\'' + nodeId + '\\')" style="padding:4px 12px;font-size:11px;background:#FF9800;color:#fff;border:none;border-radius:3px;cursor:pointer">Edit</button></div>';
    }} else {{
      h += '<div style="margin-top:8px;color:#888;font-size:10px;font-style:italic">Calculated value (edit fixed streams or constraints to change)</div>';
    }}
    content.innerHTML = h;
  }} else if (u) {{
    let h = '<h3>' + u.type + ' (' + u.id + ')</h3>';
    h += '<button onclick="openUnitEdit(\\'' + u.id + '\\')" style="padding:4px 12px;font-size:11px;background:#FF9800;color:#fff;border:none;border-radius:3px;cursor:pointer;margin-bottom:8px">Edit</button>';
    h += '<pre style="font-size:10px;background:#f5f5f5;padding:8px;border-radius:4px;overflow:auto">' + JSON.stringify(u, null, 2) + '</pre>';
    content.innerHTML = h;
  }} else if (nodeId === 'CONSTRAINTS') {{
    let h = '<h3>Constraints</h3>';
    (D.constraints||[]).forEach(c => {{ h += '<div>• ' + c + '</div>'; }});
    content.innerHTML = h;
  }} else {{
    return;
  }}
  panel.classList.add('open');
}}

function exportJSON() {{
  const blob = new Blob([JSON.stringify(D, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'flowsheet.json'; a.click();
  URL.revokeObjectURL(url);
}}

function exportModified() {{
  // ストリームの編集結果を反映した JSON
  const blob = new Blob([JSON.stringify(D, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'flowsheet_modified.json'; a.click();
  URL.revokeObjectURL(url);
}}

function exportLayout() {{
  const layoutData = {{ ...D, _layout: {{}} }};
  currentNodes.forEach(n => {{ layoutData._layout[n.id] = n.position; }});
  const blob = new Blob([JSON.stringify(layoutData, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'flowsheet_layout.json'; a.click();
  URL.revokeObjectURL(url);
}}

// --- Edit Modal ---
let editingId = null;

function openEdit(sid) {{
  const s = streamMap[sid];
  if (!s) return;
  editingId = sid;
  const box = document.getElementById('edit-box');
  let h = '<h3>Edit: ' + s.id + '</h3>';
  h += '<label>Name</label><input id="ed-name" value="' + (s.name||s.id) + '">';
  h += '<label>T (°C)</label><input id="ed-T" type="number" step="any" value="' + (s.T_celsius!==null?s.T_celsius:'') + '">';
  h += '<label>P</label><input id="ed-P" value="' + (s.P_input||'') + '">';
  h += '<label>Phase</label><select id="ed-phase"><option value="">-</option><option value="Gas"'+(s.phase==='Gas'?' selected':'')+'>Gas</option><option value="Liquid"'+(s.phase==='Liquid'?' selected':'')+'>Liquid</option><option value="Mixed"'+(s.phase==='Mixed'?' selected':'')+'>Mixed</option></select>';
  h += '<label>Components (mol/h)</label>';
  const comps = s.components || {{}};
  for (const [k,v] of Object.entries(comps)) {{
    h += '<div class="comp-row"><span>' + k + '</span><input class="ed-comp" data-comp="'+k+'" type="number" step="any" value="' + v + '"></div>';
  }}
  h += '<div class="comp-row"><span>Add:</span><input id="ed-new-comp" placeholder="formula" style="flex:0.5"><input id="ed-new-val" type="number" step="any" value="0" style="flex:0.5"><button onclick="addComp()" style="flex:0.3;background:#1976D2;color:#fff;border:none;border-radius:3px;padding:3px 8px;cursor:pointer">+</button></div>';
  h += '<div class="btn-row"><button class="btn-cancel" onclick="closeEdit()">Cancel</button><button class="btn-save" onclick="saveEdit()">Save</button></div>';
  box.innerHTML = h;
  document.getElementById('edit-modal').classList.add('open');
}}

function addComp() {{
  const f = document.getElementById('ed-new-comp').value.trim();
  const v = parseFloat(document.getElementById('ed-new-val').value) || 0;
  if (!f) return;
  const s = streamMap[editingId];
  if (!s) return;
  s.components[f] = v;
  openEdit(editingId); // refresh
}}

function closeEdit() {{
  document.getElementById('edit-modal').classList.remove('open');
  editingId = null;
}}

function saveEdit() {{
  const s = streamMap[editingId];
  if (!s) return;
  s.name = document.getElementById('ed-name').value;
  const tVal = document.getElementById('ed-T').value;
  s.T_celsius = tVal !== '' ? parseFloat(tVal) : null;
  s.P_input = document.getElementById('ed-P').value || null;
  s.phase = document.getElementById('ed-phase').value || null;
  document.querySelectorAll('.ed-comp').forEach(inp => {{
    s.components[inp.dataset.comp] = parseFloat(inp.value) || 0;
  }});
  // Recalc totals
  let tm=0, tg=0, tv=0;
  for (const [k,v] of Object.entries(s.components)) {{ tm += v; }}
  s.total_mol = tm;
  // Update D.streams
  closeEdit();
  showDetail(editingId);
}}

// --- Context Menu ---
let ctxPos = {{x:0, y:0}};
document.addEventListener('contextmenu', (ev) => {{
  if (ev.target.closest('.react-flow')) {{
    ev.preventDefault();
    ctxPos = {{x: ev.clientX, y: ev.clientY}};
    const menu = document.getElementById('ctx-menu');
    menu.style.left = ev.clientX + 'px';
    menu.style.top = ev.clientY + 'px';
    menu.innerHTML = '<div onclick="addStream()">+ Stream</div><div onclick="addUnit(\\'Mixer\\')">+ Mixer</div><div onclick="addUnit(\\'Reactor\\')">+ Reactor</div><div onclick="addUnit(\\'MultiReactor\\')">+ MultiReactor</div><div onclick="addUnit(\\'Absorber\\')">+ Absorber</div><div onclick="addUnit(\\'GibbsReactor\\')">+ GibbsReactor</div><div onclick="addConstraint()">+ Constraint</div><hr style="margin:2px 0"><div onclick="hideCtx()">Cancel</div>';
    menu.style.display = 'block';
  }}
}});
document.addEventListener('click', () => {{ document.getElementById('ctx-menu').style.display = 'none'; }});
function hideCtx() {{ document.getElementById('ctx-menu').style.display = 'none'; }}

// --- Unit Edit ---
function openUnitEdit(uid) {{
  const u = unitMap[uid];
  if (!u) return;
  const box = document.getElementById('edit-box');
  let h = '<h3>Edit: ' + u.type + ' (' + uid + ')</h3>';
  if (u.conversion !== undefined) {{
    h += '<label>Conversion</label><input id="ued-conv" type="number" step="0.01" value="' + u.conversion + '">';
  }}
  if (u.stages !== undefined) {{
    h += '<label>Stages</label><input id="ued-stages" type="number" value="' + u.stages + '">';
  }}
  if (u.T_celsius !== undefined) {{
    h += '<label>T (°C)</label><input id="ued-T" type="number" step="any" value="' + u.T_celsius + '">';
  }}
  if (u.selectivities) {{
    u.selectivities.forEach((s,i) => {{
      h += '<label>Selectivity ' + (i+1) + '</label><input class="ued-sel" type="number" step="0.01" value="' + s + '">';
    }});
  }}
  h += '<div class="btn-row"><button class="btn-cancel" onclick="closeEdit()">Cancel</button><button class="btn-save" onclick="saveUnitEdit(\\''+uid+'\\')">Save</button></div>';
  box.innerHTML = h;
  document.getElementById('edit-modal').classList.add('open');
}}
function saveUnitEdit(uid) {{
  const u = unitMap[uid];
  if (!u) return;
  const cv = document.getElementById('ued-conv');
  if (cv) u.conversion = parseFloat(cv.value);
  const st = document.getElementById('ued-stages');
  if (st) u.stages = parseInt(st.value);
  const ut = document.getElementById('ued-T');
  if (ut) u.T_celsius = parseFloat(ut.value);
  const sels = document.querySelectorAll('.ued-sel');
  if (sels.length && u.selectivities) sels.forEach((el,i) => {{ u.selectivities[i] = parseFloat(el.value); }});
  closeEdit();
  showDetail(uid);
}}

function _addRfNode(id, type, data, w, h) {{
  rfNodes.push({{ id, type, position: {{ x: ctxPos.x - 200, y: ctxPos.y - 100 }}, data }});
  window.__setNodes && window.__setNodes([...rfNodes]);
}}

function addStream() {{
  hideCtx();
  const id = prompt('Stream name:', 'New_' + (D.streams.length + 1));
  if (!id) return;
  const ns = {{ id, name: id, index: D.streams.length+1, T_celsius: 25, P_input: null, phase: 'Gas', fixed: true, total_mol: 0, total_NL: 0, total_g: 0, components: {{}}, original_components: null, has_composition_constraints: false }};
  D.streams.push(ns);
  streamMap[id] = ns;
  _addRfNode(id, 'stream', {{ label: ns.index + '. ' + id, info: '(new)', cls: 'zero' }});
  openEdit(id);
}}

function addUnit(utype) {{
  hideCtx();
  const uid = 'U' + (D.units.length + 1);
  const u = {{ id: uid, type: utype }};
  if (utype === 'Reactor' || utype === 'MultiReactor') u.conversion = 0.5;
  if (utype === 'MultiReactor') {{ u.reactions = []; u.selectivities = []; u.key = ''; }}
  if (utype === 'Absorber') {{ u.stages = 10; u.T_celsius = 25; }}
  if (utype === 'GibbsReactor') {{ u.T_celsius = 850; u.species = []; }}
  D.units.push(u);
  unitMap[uid] = u;
  let lines = [utype];
  _addRfNode(uid, 'unit', {{ lines }});
  openUnitEdit(uid);
}}

function addConstraint() {{
  hideCtx();
  const label = prompt('Constraint label:', 'New constraint');
  const code = prompt('Constraint code (lambda):', 'lambda: Mixed.total_molar_flow - 100');
  if (!label) return;
  D.constraints = D.constraints || [];
  D.constraints.push(label);
  D.constraint_specs = D.constraint_specs || [];
  D.constraint_specs.push({{ label, code: code || '' }});
  // Update constraint node
  const existing = rfNodes.find(n => n.id === 'CONSTRAINTS');
  if (existing) {{
    existing.data.lines = ['Constraints:'].concat(D.constraints);
  }} else {{
    rfNodes.push({{ id: 'CONSTRAINTS', type: 'constraint', position: {{ x: ctxPos.x-200, y: ctxPos.y-100 }}, data: {{ lines: ['Constraints:', label] }} }});
  }}
  window.__setNodes && window.__setNodes([...rfNodes]);
}}

// --- Edge addition via onConnect ---

let currentNodes = rfNodes;

function App() {{
  const [nodes, setNodes] = useState(rfNodes);
  const [edges, setEdges] = useState(rfEdges);
  window.__setNodes = setNodes;
  window.__setEdges = setEdges;
  const onNodesChange = useCallback(ch => {{
    setNodes(nds => {{ const updated = applyNodeChanges(ch, nds); currentNodes = updated; return updated; }});
  }}, []);
  const onEdgesChange = useCallback(ch => setEdges(eds => applyEdgeChanges(ch, eds)), []);
  const onNodeClick = useCallback((ev, node) => showDetail(node.id), []);
  const onConnect = useCallback((conn) => {{
    const newEdge = {{ id: conn.source + '_' + conn.target, source: conn.source, target: conn.target, type: 'smoothstep', style: {{ strokeWidth: 2 }} }};
    rfEdges.push(newEdge);
    setEdges([...rfEdges]);
  }}, []);
  return e(RF, {{ nodes, edges, onNodesChange, onEdgesChange, onNodeClick, onConnect, nodeTypes, fitView: true, minZoom: 0.2, maxZoom: 3 }},
    e(Background, {{ variant:'dots', gap:16, size:1 }}),
    e(Controls, null),
    e(MiniMap, {{ nodeStrokeWidth:3, zoomable:true, pannable:true }})
  );
}}
ReactDOM.createRoot(document.getElementById('root')).render(e(App));
</script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
