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

    def print_streams(self) -> None:
        for s in self.streams:
            label = s.name or "unnamed"
            n = s.n_components
            if n == 0:
                continue

            mol = s.molar_flows
            x = s.mole_fractions
            mass = s.mass_flows
            w = s.mass_fractions
            nvol = s.normal_volume_flows
            vf = s.volume_fractions
            formulas = [c.formula for c in s.components]

            # ヘッダー
            print(f"\n{'=' * 70}")
            print(f" {label}")
            print(f"{'=' * 70}")

            # 列幅
            fw = max(len(f) for f in formulas)  # formula 幅
            fw = max(fw, 5)
            col = 10  # 数値列幅

            # テーブルヘッダー
            header = f"  {'':>{fw}s}"
            for f in formulas:
                header += f"  {f:>{col}s}"
            header += f"  {'Total':>{col}s}"
            print(header)
            print(f"  {'':>{fw}s}" + ("  " + "-" * col) * (n + 1))

            # mol flow
            row = f"  {'mol':>{fw}s}"
            for v in mol:
                row += f"  {v:{col}.4f}"
            row += f"  {s.total_molar_flow:{col}.4f}"
            print(row)

            # mole fraction
            row = f"  {'x':>{fw}s}"
            for v in x:
                row += f"  {v:{col}.4f}"
            row += f"  {'1.0000':>{col}s}"
            print(row)

            # mass flow
            row = f"  {'mass':>{fw}s}"
            for v in mass:
                row += f"  {v:{col}.4f}"
            row += f"  {s.total_mass_flow:{col}.4f}"
            print(row)

            # mass fraction
            row = f"  {'w':>{fw}s}"
            for v in w:
                row += f"  {v:{col}.4f}"
            row += f"  {'1.0000':>{col}s}"
            print(row)

            # normal volume flow
            row = f"  {'Nm3':>{fw}s}"
            for v in nvol:
                row += f"  {v:{col}.4f}"
            row += f"  {s.total_normal_volume_flow:{col}.4f}"
            print(row)

            # volume fraction
            row = f"  {'vf':>{fw}s}"
            for v in vf:
                row += f"  {v:{col}.4f}"
            row += f"  {'1.0000':>{col}s}"
            print(row)
