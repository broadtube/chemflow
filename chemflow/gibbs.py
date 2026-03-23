"""Gibbs リアクター: Cantera 連携によるギブズ自由エネルギー最小化。"""

from __future__ import annotations

import re

import numpy as np

from chemflow.errors import CanteraError


def parse_pressure(P) -> float:
    """圧力を Pa (絶対圧) に変換する。

    Parameters
    ----------
    P : float or str
        数値の場合は Pa (absolute)。
        文字列の場合:
          "2MPaG" → 2e6 + 101325 Pa
          "2MPa"  → 2e6 Pa
          "10atm" → 10 * 101325 Pa
    """
    ATM = 101325.0

    if isinstance(P, (int, float)):
        return float(P)

    s = str(P).strip()

    # MPaG
    m = re.match(r"^([0-9.]+)\s*MPaG$", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6 + ATM

    # MPa
    m = re.match(r"^([0-9.]+)\s*MPa$", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6

    # kPaG
    m = re.match(r"^([0-9.]+)\s*kPaG$", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e3 + ATM

    # kPa
    m = re.match(r"^([0-9.]+)\s*kPa$", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e3

    # atm
    m = re.match(r"^([0-9.]+)\s*atm$", s, re.IGNORECASE)
    if m:
        return float(m.group(1)) * ATM

    raise ValueError(f"Cannot parse pressure: '{P}'")


class GibbsReactor:
    """Cantera を使ったギブズ自由エネルギー最小化リアクター。

    残差 = outlet.molar_flows - equilibrium_molar_flows
    """

    def __init__(self, inlet, outlet, T_celsius: float, P_pascal: float, species: list[str]):
        self.inlet = inlet
        self.outlet = outlet
        self.T_kelvin = T_celsius + 273.15
        self.P_pascal = P_pascal
        self.species = species

        # Cantera Solution を構築
        try:
            import cantera as ct
            all_species = ct.Species.list_from_file("gri30.yaml")
            selected = [s for s in all_species if s.name in species]
            if len(selected) != len(species):
                found = {s.name for s in selected}
                missing = set(species) - found
                raise CanteraError(
                    f"Species not found in gri30.yaml: {missing}"
                )
            self._gas = ct.Solution(thermo="ideal-gas", species=selected)
        except ImportError:
            raise CanteraError(
                "Cantera is not installed. Install with: pip install cantera"
            )
        except CanteraError:
            raise
        except Exception as e:
            raise CanteraError(f"Failed to create Cantera solution: {e}") from e

    def residuals(self) -> np.ndarray:
        """残差を計算: outlet - equilibrium(inlet)。

        Cantera Quantity を使用して、反応前後のモル数変化を正しく追跡する。
        """
        try:
            import cantera as ct
        except ImportError:
            raise CanteraError("Cantera is not installed")

        # 入口のモル流量を species 順に取得
        inlet_molar = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            for j, c in enumerate(self.inlet.components):
                if c.formula == sp:
                    inlet_molar[i] = self.inlet.molar_flows[j]
                    break

        total_inlet = inlet_molar.sum()
        if total_inlet <= 0:
            return self.outlet.molar_flows - np.zeros(len(self.outlet.components))

        inlet_frac = inlet_molar / total_inlet

        # Cantera Quantity で平衡計算（モル数変化を追跡）
        self._gas.TPX = self.T_kelvin, self.P_pascal, {
            sp: f for sp, f in zip(self.species, inlet_frac)
        }
        # Cantera は kmol 単位なので変換
        q = ct.Quantity(self._gas, moles=total_inlet / 1000.0)
        q.equilibrate("TP")

        # 平衡後の各成分モル流量
        eq_molar_flows = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            idx = self._gas.species_index(sp)
            eq_molar_flows[i] = q.moles * q.X[idx] * 1000.0  # kmol → mol

        # 出口の対応する成分のモル流量と比較
        outlet_molar = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            for j, c in enumerate(self.outlet.components):
                if c.formula == sp:
                    outlet_molar[i] = self.outlet.molar_flows[j]
                    break

        return outlet_molar - eq_molar_flows
