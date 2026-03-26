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

    def set_component_order(self, order: list[str]) -> None:
        """出力時の成分表示順序を設定する。

        Parameters
        ----------
        order : list[str]
            示性式のリスト（例: ["H2", "CO", "CO2", "CH4", "H2O"]）
            リストにない成分は末尾に出現順で追加される。
        """
        self._component_order = order

    def print_streams(self) -> None:
        """全ストリームを横並びの統合テーブルで表示する。

        行: 3セクション（mol / Volume / weight）× (成分数 + Total)
        列: 物質名 | MW | Stream1(abs, rel) | Stream2(abs, rel) | ...
        """
        streams = [s for s in self.streams if s.n_components > 0]
        if not streams:
            return

        # 全ストリームの成分和集合
        all_set: set[str] = set()
        all_default: list[str] = []
        for s in streams:
            for c in s.components:
                if c.formula not in all_set:
                    all_default.append(c.formula)
                    all_set.add(c.formula)

        # ユーザー指定順序を適用
        if hasattr(self, "_component_order") and self._component_order:
            ordered = [f for f in self._component_order if f in all_set]
            remaining = [f for f in all_default if f not in ordered]
            all_formulas = ordered + remaining
        else:
            all_formulas = all_default

        # MW テーブル
        from chemflow.registry import ComponentRegistry
        mw_map = {f: ComponentRegistry.get(f).mw for f in all_formulas}

        # 各ストリームのデータを成分和集合順に構築
        def _get_values(stream, formulas):
            """ストリームの値を全成分順に並べた dict を返す。"""
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
            mol_frac = mol / total_mol if total_mol else np.zeros_like(mol)
            mass_frac = mass / total_mass if total_mass else np.zeros_like(mass)
            vol_frac = nvol / total_nvol if total_nvol else np.zeros_like(nvol)
            return {
                "mol": mol, "mol_frac": mol_frac, "total_mol": total_mol,
                "mass": mass, "mass_frac": mass_frac, "total_mass": total_mass,
                "nvol": nvol, "vol_frac": vol_frac, "total_nvol": total_nvol,
            }

        data = [_get_values(s, all_formulas) for s in streams]

        # 列幅
        fw = max(max(len(f) for f in all_formulas), 5)  # formula 幅
        mw_w = 8   # MW 列幅
        abs_w = 10  # absolute 値幅
        rel_w = 8   # relative 値幅
        stream_w = abs_w + rel_w + 1  # ストリーム1本分の幅

        n_streams = len(streams)
        n_comp = len(all_formulas)

        # ストリーム名
        names = [s.name or f"S{i+1}" for i, s in enumerate(streams)]

        # セクション定義: (セクション名, abs単位, rel単位, absキー, relキー, totalキー)
        sections = [
            ("mol",    "mol/h",  "mol%",  "mol",  "mol_frac",  "total_mol"),
            ("Volume", "NL/h",   "vol%",  "nvol", "vol_frac",  "total_nvol"),
            ("weight", "g/h",    "wt%",   "mass", "mass_frac", "total_mass"),
        ]

        for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in sections:
            # セクションヘッダー
            print(f"[{sec_name}]")

            # ストリーム名行
            h1 = f"{'':>{fw}s}  {'MW':>{mw_w}s}"
            for nm in names:
                h1 += f"  {nm:^{stream_w}s}"
            print(h1)

            # 単位行
            h2 = f"{'':>{fw}s}  {'':>{mw_w}s}"
            for _ in names:
                h2 += f"  {abs_unit:>{abs_w}s} {rel_unit:>{rel_w}s}"
            print(h2)

            # 区切り線
            sep = f"{'':>{fw}s}  {'-' * mw_w}"
            for _ in names:
                sep += f"  {'-' * abs_w} {'-' * rel_w}"
            print(sep)

            # 各成分行
            for i, f in enumerate(all_formulas):
                row = f"  {f:>{fw}s}  {mw_map[f]:{mw_w}.2f}"
                for d in data:
                    row += f"  {d[abs_key][i]:{abs_w}.4f} {d[rel_key][i]:{rel_w}.4f}"
                print(row)

            # Total 行
            row = f"  {'Total':>{fw}s}  {'':>{mw_w}s}"
            for d in data:
                row += f"  {d[total_key]:{abs_w}.4f} {'1.0000':>{rel_w}s}"
            print(row)
            print()
