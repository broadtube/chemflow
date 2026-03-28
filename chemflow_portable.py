"""chemflow_portable.py — 単一ファイル版 化学プロセスシミュレーター

依存: numpy, scipy, molmass (必須), cantera (gibbs_react使用時のみ)

使い方:
    from chemflow_portable import Stream, eq, constrain, solve, reset, print_streams
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import root


# ============================================================
# Errors
# ============================================================

class ChemflowError(Exception):
    pass

class FormulaError(ChemflowError):
    pass

class BasisError(ChemflowError):
    pass

class SolveError(ChemflowError):
    pass

class ConstraintError(ChemflowError):
    pass

class CanteraError(ChemflowError):
    pass


# ============================================================
# Component / Registry
# ============================================================

class Component:
    def __init__(self, name: str, mw: float, normal_volume: float = 22.414):
        self.name = name
        self.formula = name
        self.mw = mw
        self.normal_volume = normal_volume

    def __repr__(self) -> str:
        return f"Component({self.name!r}, mw={self.mw})"


class ComponentRegistry:
    _cache: dict[str, Component] = {}

    @classmethod
    def get(cls, formula: str) -> Component:
        if formula in cls._cache:
            return cls._cache[formula]
        try:
            from molmass import Formula as MolFormula
            mw = MolFormula(formula).mass
        except Exception as e:
            raise FormulaError(f"Unknown formula: '{formula}' ({e})") from e
        comp = Component(formula, mw=mw)
        cls._cache[formula] = comp
        return comp

    @classmethod
    def get_many(cls, formulas: list[str]) -> list[Component]:
        return [cls.get(f) for f in formulas]

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()


# ============================================================
# Units (Mixer / Splitter / Reactor)
# ============================================================

def _get_flows_by_formula(stream) -> dict[str, float]:
    return {c.formula: stream.molar_flows[i] for i, c in enumerate(stream.components)}


def _build_residual_vector(outlet, flow_dict: dict[str, float]) -> np.ndarray:
    expected = np.zeros(outlet.n_components)
    for i, c in enumerate(outlet.components):
        expected[i] = flow_dict.get(c.formula, 0.0)
    return outlet.molar_flows - expected


class Mixer:
    def __init__(self, name: str, inlets: list, outlet):
        self.name = name
        self.inlets = inlets
        self.outlet = outlet

    def residuals(self) -> np.ndarray:
        total: dict[str, float] = {}
        for s in self.inlets:
            for formula, flow in _get_flows_by_formula(s).items():
                total[formula] = total.get(formula, 0.0) + flow
        return _build_residual_vector(self.outlet, total)


class Splitter:
    def __init__(self, name: str, inlet, outlets: list, ratios: list[float]):
        self.name = name
        self.inlet = inlet
        self.outlets = outlets
        self.ratios = np.asarray(ratios, dtype=float)

    def residuals(self) -> np.ndarray:
        inlet_flows = _get_flows_by_formula(self.inlet)
        res_list = []
        for outlet, ratio in zip(self.outlets, self.ratios):
            expected = {f: v * ratio for f, v in inlet_flows.items()}
            res_list.append(_build_residual_vector(outlet, expected))
        return np.concatenate(res_list)


class Reactor:
    def __init__(self, name: str, inlet, outlet, stoichiometry, key_component: int, conversion: float):
        self.name = name
        self.inlet = inlet
        self.outlet = outlet
        self.stoichiometry = np.asarray(stoichiometry, dtype=float)
        self.key_component = key_component
        self.conversion = conversion

    def residuals(self) -> np.ndarray:
        outlet_formulas = [c.formula for c in self.outlet.components]
        inlet_flows = _get_flows_by_formula(self.inlet)
        key_formula = self.outlet.components[self.key_component].formula
        key_inlet = inlet_flows.get(key_formula, 0.0)
        extent = self.conversion * key_inlet / abs(self.stoichiometry[self.key_component])
        expected = np.zeros(len(outlet_formulas))
        for i, f in enumerate(outlet_formulas):
            expected[i] = inlet_flows.get(f, 0.0) + self.stoichiometry[i] * extent
        return self.outlet.molar_flows - expected


class MultiReactor:
    def __init__(self, name, inlet, outlet, reactions, key, conversion, selectivities):
        self.name = name
        self.inlet = inlet
        self.outlet = outlet
        self.reactions = reactions
        self.key = key
        self.conversion = conversion
        self.selectivities = selectivities

    def residuals(self) -> np.ndarray:
        outlet_formulas = [c.formula for c in self.outlet.components]
        inlet_flows = _get_flows_by_formula(self.inlet)
        key_inlet = inlet_flows.get(self.key, 0.0)
        expected = np.zeros(len(outlet_formulas))
        for i, f in enumerate(outlet_formulas):
            expected[i] = inlet_flows.get(f, 0.0)
        for rxn, sel in zip(self.reactions, self.selectivities):
            key_consumed = sel * self.conversion * key_inlet
            key_stoich = abs(rxn[self.key])
            extent = key_consumed / key_stoich
            for f, coeff in rxn.items():
                if f in outlet_formulas:
                    expected[outlet_formulas.index(f)] += coeff * extent
        return self.outlet.molar_flows - expected


def antoine_water_psat(T_celsius: float) -> float:
    A, B, C = 8.07131, 1730.63, 233.426
    log_p_mmhg = A - B / (C + T_celsius)
    return (10.0 ** log_p_mmhg) * 133.322


# ============================================================
# Henry則定数 (Sander 2023, van't Hoff温度依存)
# ============================================================

_VM_WATER = 18.015e-6  # 水のモル体積 [m3/mol]
_T0_HENRY = 298.15

_HENRY_BUILTIN = {
    "H2":      {"Hcp": 7.8e-6,  "Tderiv": 500,  "cas": "1333-74-0"},
    "N2":      {"Hcp": 6.4e-6,  "Tderiv": 1300, "cas": "7727-37-9"},
    "O2":      {"Hcp": 1.3e-5,  "Tderiv": 1500, "cas": "7782-44-7"},
    "CO":      {"Hcp": 9.5e-6,  "Tderiv": 1300, "cas": "630-08-0"},
    "CO2":     {"Hcp": 3.3e-4,  "Tderiv": 2400, "cas": "124-38-9"},
    "CH4":     {"Hcp": 1.4e-5,  "Tderiv": 1600, "cas": "74-82-8"},
    "NH3":     {"Hcp": 5.9e-1,  "Tderiv": 4200, "cas": "7664-41-7"},
    "H2S":     {"Hcp": 1.0e-3,  "Tderiv": 2100, "cas": "7783-06-4"},
    "SO2":     {"Hcp": 1.2e-2,  "Tderiv": 2900, "cas": "7446-09-5"},
    "CH3CHO":  {"Hcp": 1.3e-1,  "Tderiv": 5900, "cas": "75-07-0"},
    "CH3COOH": {"Hcp": 4.1e+3,  "Tderiv": 6300, "cas": "64-19-7"},
    "CH3OH":   {"Hcp": 2.2e+0,  "Tderiv": 5200, "cas": "67-56-1"},
    "C2H5OH":  {"Hcp": 1.9e+0,  "Tderiv": 6600, "cas": "64-17-5"},
    "HCHO":    {"Hcp": 3.2e+3,  "Tderiv": 6800, "cas": "50-00-0"},
    "HCOOH":   {"Hcp": 8.9e+3,  "Tderiv": 5700, "cas": "64-18-6"},
}

_henry_runtime_cache: dict[str, dict] = {}


def _henry_pa(Hcp_298, Tderiv, T_kelvin):
    import math
    hcp = Hcp_298 * math.exp(Tderiv * (1.0 / T_kelvin - 1.0 / _T0_HENRY))
    return 1.0 / (hcp * _VM_WATER) if hcp > 0 else 1e15


def _henry_fetch_online(formula):
    """PubChem + henrys-law.org からHenry定数を取得。"""
    import urllib.request
    import html as html_mod

    cas = None
    cas_pat = re.compile(r"^\d{2,7}-\d{2}-\d$")
    for url in [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/{formula}/synonyms/JSON?MaxRecords=1",
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{formula}/synonyms/JSON",
    ]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chemflow/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if "Waiting" in data:
                continue
            for syn in data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", []):
                if cas_pat.match(syn):
                    cas = syn
                    break
            if cas:
                break
        except Exception:
            continue
    if not cas:
        return None

    try:
        req = urllib.request.Request(
            f"https://www.henrys-law.org/henry/casrn/{cas}",
            headers={"User-Agent": "chemflow/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = html_mod.unescape(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    hcp_pat = re.compile(r"(\d+\.?\d*)\s*×\s*10<sup>\s*([−+-]?\d+)\s*</sup>")
    hcp_vals = []
    for m in hcp_pat.finditer(text):
        mantissa = float(m.group(1))
        exp = int(m.group(2).replace("−", "-"))
        hcp_vals.append(mantissa * (10 ** exp))
    if not hcp_vals:
        return None

    td_pat = re.compile(r"<td>\s*(\d{3,5})\s*</td>")
    td_vals = [int(m.group(1)) for m in td_pat.finditer(text) if 100 <= int(m.group(1)) <= 20000]

    result = {"Hcp": hcp_vals[0], "Tderiv": td_vals[0] if td_vals else 0, "cas": cas, "source": "henrys-law.org"}
    _henry_runtime_cache[formula] = result

    try:
        cache_dir = Path.home() / ".chemflow" / "henry_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{formula}.json").write_text(json.dumps(result))
    except Exception:
        pass
    return result


def _henry_get_data(formula):
    if formula in _HENRY_BUILTIN:
        return _HENRY_BUILTIN[formula]
    if formula in _henry_runtime_cache:
        return _henry_runtime_cache[formula]
    try:
        cache_file = Path.home() / ".chemflow" / "henry_cache" / f"{formula}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            _henry_runtime_cache[formula] = data
            return data
    except Exception:
        pass
    return _henry_fetch_online(formula)


def get_henry_constants(formulas, T_celsius):
    """複数成分のHenry定数 [Pa] を温度依存で取得。"""
    T_k = T_celsius + 273.15
    result = {}
    for f in formulas:
        if f == "H2O":
            continue
        data = _henry_get_data(f)
        if data:
            result[f] = _henry_pa(data["Hcp"], data["Tderiv"], T_k)
    return result


class WaterSeparator:
    def __init__(self, name, inlet, gas_outlet, water_outlet, T_celsius, P_pascal,
                 henry_constants=None):
        self.name = name
        self.inlet = inlet
        self.gas_outlet = gas_outlet
        self.water_outlet = water_outlet
        self.T_celsius = T_celsius
        self.P_pascal = P_pascal
        if henry_constants is not None:
            self.henry = henry_constants
        else:
            all_formulas = set()
            for c in inlet.components:
                all_formulas.add(c.formula)
            for c in gas_outlet.components:
                all_formulas.add(c.formula)
            self.henry = get_henry_constants(list(all_formulas), T_celsius)

    def residuals(self) -> np.ndarray:
        inlet_flows = _get_flows_by_formula(self.inlet)
        gas_formulas = [c.formula for c in self.gas_outlet.components]
        water_formulas = [c.formula for c in self.water_outlet.components]
        p_sat = antoine_water_psat(self.T_celsius)
        y_sat = p_sat / self.P_pascal
        P = self.P_pascal
        res = []
        # 物質収支
        for j, f in enumerate(water_formulas):
            gas_idx = gas_formulas.index(f) if f in gas_formulas else None
            gas_j = self.gas_outlet.molar_flows[gas_idx] if gas_idx is not None else 0.0
            water_j = self.water_outlet.molar_flows[j]
            res.append(gas_j + water_j - inlet_flows.get(f, 0.0))
        # Antoine
        gas_h2o_idx = gas_formulas.index("H2O") if "H2O" in gas_formulas else None
        gas_h2o = self.gas_outlet.molar_flows[gas_h2o_idx] if gas_h2o_idx is not None else 0.0
        non_h2o_gas = sum(self.gas_outlet.molar_flows[i] for i, f in enumerate(gas_formulas) if f != "H2O")
        res.append(gas_h2o - y_sat / (1.0 - y_sat) * non_h2o_gas)
        # Henry
        water_h2o_idx = water_formulas.index("H2O") if "H2O" in water_formulas else None
        n_water_liq = self.water_outlet.molar_flows[water_h2o_idx] if water_h2o_idx is not None else 0.0
        total_gas = sum(self.gas_outlet.molar_flows[i] for i, _ in enumerate(gas_formulas))
        for j, f in enumerate(water_formulas):
            if f == "H2O":
                continue
            if f not in self.henry:
                res.append(self.water_outlet.molar_flows[j])
                continue
            H_i = self.henry[f]
            gas_idx = gas_formulas.index(f) if f in gas_formulas else None
            gas_i = self.gas_outlet.molar_flows[gas_idx] if gas_idx is not None else 0.0
            water_i = self.water_outlet.molar_flows[j]
            scale = max(H_i, 1.0)
            res.append((water_i * total_gas * H_i - gas_i * P * n_water_liq) / scale)
        return np.array(res)


def _kremser_absorption_fraction(A, N):
    if abs(A - 1.0) < 1e-10:
        return N / (N + 1.0)
    AN1 = A ** (N + 1)
    return (AN1 - A) / (AN1 - 1.0)


class Absorber:
    def __init__(self, name, gas_inlet, water_inlet, gas_outlet, liquid_outlet,
                 T_celsius, P_pascal, stages, henry_constants=None):
        self.name = name
        self.gas_inlet = gas_inlet
        self.water_inlet = water_inlet
        self.gas_outlet = gas_outlet
        self.liquid_outlet = liquid_outlet
        self.T_celsius = T_celsius
        self.P_pascal = P_pascal
        self.stages = stages
        if henry_constants is not None:
            self.henry = henry_constants
        else:
            all_formulas = set()
            for s in [gas_inlet, water_inlet, gas_outlet, liquid_outlet]:
                for c in s.components:
                    all_formulas.add(c.formula)
            self.henry = get_henry_constants(list(all_formulas), T_celsius)

    def residuals(self) -> np.ndarray:
        gas_in = _get_flows_by_formula(self.gas_inlet)
        water_in = _get_flows_by_formula(self.water_inlet)
        gas_formulas = [c.formula for c in self.gas_outlet.components]
        liq_formulas = [c.formula for c in self.liquid_outlet.components]
        P, N = self.P_pascal, self.stages
        p_sat = antoine_water_psat(self.T_celsius)
        y_sat = p_sat / P
        total_gas_in = sum(gas_in.get(f, 0.0) for f in gas_formulas if f != "H2O")
        L = water_in.get("H2O", 0.0)
        V = total_gas_in
        res = []
        for j, f in enumerate(liq_formulas):
            gas_out_idx = gas_formulas.index(f) if f in gas_formulas else None
            gas_out_j = self.gas_outlet.molar_flows[gas_out_idx] if gas_out_idx is not None else 0.0
            liq_out_j = self.liquid_outlet.molar_flows[j]
            feed_j = gas_in.get(f, 0.0) + water_in.get(f, 0.0)
            res.append(gas_out_j + liq_out_j - feed_j)
        gas_h2o_idx = gas_formulas.index("H2O") if "H2O" in gas_formulas else None
        gas_h2o = self.gas_outlet.molar_flows[gas_h2o_idx] if gas_h2o_idx is not None else 0.0
        non_h2o_gas_out = sum(self.gas_outlet.molar_flows[i] for i, f in enumerate(gas_formulas) if f != "H2O")
        res.append(gas_h2o - y_sat / (1.0 - y_sat) * non_h2o_gas_out)
        for j, f in enumerate(liq_formulas):
            if f == "H2O":
                continue
            if f not in self.henry:
                res.append(self.liquid_outlet.molar_flows[j] - water_in.get(f, 0.0))
                continue
            H_i = self.henry[f]
            K_i = H_i / P
            safe_V = max(abs(V), 1e-10)
            A_i = (L / safe_V) / K_i
            frac = _kremser_absorption_fraction(A_i, N)
            feed_gas_i = gas_in.get(f, 0.0)
            absorbed = frac * feed_gas_i
            expected_liq = water_in.get(f, 0.0) + absorbed
            res.append(self.liquid_outlet.molar_flows[j] - expected_liq)
        return np.array(res)


# ============================================================
# Gibbs Reactor (Cantera)
# ============================================================

def parse_pressure(P) -> float:
    ATM = 101325.0
    if isinstance(P, (int, float)):
        return float(P)
    s = str(P).strip()
    for pattern, factor, offset in [
        (r"^([0-9.]+)\s*MPaG$", 1e6, ATM),
        (r"^([0-9.]+)\s*MPa$", 1e6, 0),
        (r"^([0-9.]+)\s*kPaG$", 1e3, ATM),
        (r"^([0-9.]+)\s*kPa$", 1e3, 0),
        (r"^([0-9.]+)\s*atm$", ATM, 0),
    ]:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            return float(m.group(1)) * factor + offset
    raise ValueError(f"Cannot parse pressure: '{P}'")


class GibbsReactor:
    def __init__(self, inlet, outlet, T_celsius: float, P_pascal: float, species: list[str]):
        self.inlet = inlet
        self.outlet = outlet
        self.T_kelvin = T_celsius + 273.15
        self.P_pascal = P_pascal
        self.species = species
        try:
            import cantera as ct
            all_species = ct.Species.list_from_file("gri30.yaml")
            selected = [s for s in all_species if s.name in species]
            if len(selected) != len(species):
                found = {s.name for s in selected}
                raise CanteraError(f"Species not found in gri30.yaml: {set(species) - found}")
            self._gas = ct.Solution(thermo="ideal-gas", species=selected)
        except ImportError:
            raise CanteraError("Cantera is not installed. Install with: pip install cantera")
        except CanteraError:
            raise
        except Exception as e:
            raise CanteraError(f"Failed to create Cantera solution: {e}") from e

    def residuals(self) -> np.ndarray:
        import cantera as ct
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
        self._gas.TPX = self.T_kelvin, self.P_pascal, {
            sp: f for sp, f in zip(self.species, inlet_frac)
        }
        q = ct.Quantity(self._gas, moles=total_inlet / 1000.0)
        q.equilibrate("TP")
        eq_molar_flows = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            idx = self._gas.species_index(sp)
            eq_molar_flows[i] = q.moles * q.X[idx] * 1000.0
        outlet_molar = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            for j, c in enumerate(self.outlet.components):
                if c.formula == sp:
                    outlet_molar[i] = self.outlet.molar_flows[j]
                    break
        return outlet_molar - eq_molar_flows


# ============================================================
# Flowsheet
# ============================================================

class Flowsheet:
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
        return [s for s in self.streams if not s._fixed]

    def _pack(self) -> np.ndarray:
        arrays = [s.molar_flows for s in self._variable_streams()]
        return np.concatenate(arrays) if arrays else np.array([])

    def _unpack(self, x: np.ndarray) -> None:
        idx = 0
        for s in self._variable_streams():
            n = s.n_components
            s.molar_flows = x[idx: idx + n].copy()
            idx += n

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        self._unpack(x)
        res_list = []
        for unit in self.units:
            r = unit.residuals()
            if r is not None and len(r) > 0:
                res_list.append(np.atleast_1d(r))
        for spec in self.specs:
            r = spec()
            if r is not None:
                res_list.append(np.atleast_1d(r))
        for s in self.streams:
            for constraint in getattr(s, "_composition_constraints", []):
                r = constraint()
                if r is not None and len(r) > 0:
                    res_list.append(np.atleast_1d(r))
        return np.concatenate(res_list) if res_list else np.array([])

    def solve(self, **kwargs) -> dict:
        x0 = self._pack()
        if len(x0) == 0:
            return {"success": True, "message": "No variables to solve"}
        n_vars = len(x0)
        n_res = len(self._residuals(x0))
        if n_vars != n_res:
            raise SolveError(
                f"System is {'over' if n_res > n_vars else 'under'}-determined: "
                f"{n_vars} variables, {n_res} equations"
            )
        result = root(self._residuals, x0, **kwargs)
        if result.success:
            self._unpack(result.x)
        else:
            raise SolveError(f"Solver did not converge: {result.message}")
        return result

    def set_component_order(self, order: list[str]) -> None:
        self._component_order = order

    def _prepare_table_data(self):
        streams = [s for s in self.streams if s.n_components > 0]
        if not streams:
            return None
        all_set, all_default = set(), []
        for s in streams:
            for c in s.components:
                if c.formula not in all_set:
                    all_default.append(c.formula)
                    all_set.add(c.formula)
        if hasattr(self, "_component_order") and self._component_order:
            ordered = list(self._component_order)
            remaining = [f for f in all_default if f not in ordered]
            all_formulas = ordered + remaining
        else:
            all_formulas = all_default
        mw_map = {f: ComponentRegistry.get(f).mw for f in all_formulas}
        def _get_values(stream, formulas):
            flow_map = {c.formula: i for i, c in enumerate(stream.components)}
            mol, mass, nvol = (np.zeros(len(formulas)) for _ in range(3))
            for i, f in enumerate(formulas):
                if f in flow_map:
                    j = flow_map[f]
                    mol[i], mass[i], nvol[i] = stream.molar_flows[j], stream.mass_flows[j], stream.normal_volume_flows[j]
            tm, tma, tv = mol.sum(), mass.sum(), nvol.sum()
            return {
                "mol": mol, "mol_frac": mol/tm if tm else np.zeros_like(mol), "total_mol": tm,
                "mass": mass, "mass_frac": mass/tma if tma else np.zeros_like(mass), "total_mass": tma,
                "nvol": nvol, "vol_frac": nvol/tv if tv else np.zeros_like(nvol), "total_nvol": tv,
            }
        return {"streams": streams, "all_formulas": all_formulas, "mw_map": mw_map,
                "data": [_get_values(s, all_formulas) for s in streams],
                "names": [s.name or f"S{i+1}" for i, s in enumerate(streams)],
                "pressures": [self._to_mpag(s) for s in streams],
                "temperatures": [f"{s.T_celsius:.1f}" if getattr(s, "T_celsius", None) is not None else "" for s in streams],
                "phases": [getattr(s, "phase", None) or "" for s in streams]}

    @staticmethod
    def _to_mpag(s):
        p = getattr(s, "P_input", None)
        if p is None:
            return ""
        if isinstance(p, str):
            p_pa = parse_pressure(p)
        else:
            p_pa = float(p)
        return f"{(p_pa - 101325) / 1e6:.3f}"

    def print_streams(self) -> None:
        t = self._prepare_table_data()
        if t is None:
            return
        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]
        fw = max(max(len(f) for f in all_formulas), 9)
        mw_w, abs_w, rel_w = 8, 10, 8
        stream_w = abs_w + rel_w + 1
        def _hr(label, values, mw_val=""):
            row = f"  {label:>{fw}s}  {mw_val:>{mw_w}s}"
            for v in values: row += f"  {v:^{stream_w}s}"
            return row
        print(_hr("No.", [str(i+1) for i in range(len(names))]))
        print(_hr("Service", names))
        print(_hr("Press.", pressures, "MPaG"))
        print(_hr("Temp.", temperatures, "°C"))
        print(_hr("Phase", phases))
        for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in [
            ("mol","mol/h","mol%","mol","mol_frac","total_mol"),
            ("Volume","NL/h","vol%","nvol","vol_frac","total_nvol"),
            ("weight","g/h","wt%","mass","mass_frac","total_mass"),
        ]:
            h2 = f"  {'Component':>{fw}s}  {'MW':>{mw_w}s}"
            for _ in names: h2 += f"  {abs_unit:>{abs_w}s} {rel_unit:>{rel_w}s}"
            print(h2)
            sep = f"{'':>{fw}s}  {'-' * mw_w}"
            for _ in names: sep += f"  {'-' * abs_w} {'-' * rel_w}"
            print(sep)
            for i, f in enumerate(all_formulas):
                row = f"  {f:>{fw}s}  {mw_map[f]:{mw_w}.2f}"
                for d in data: row += f"  {d[abs_key][i]:{abs_w}.4f} {d[rel_key][i]:{rel_w}.4f}"
                print(row)
            row = f"  {'Total':>{fw}s}  {'':>{mw_w}s}"
            for d in data: row += f"  {d[total_key]:{abs_w}.4f} {'1.0000':>{rel_w}s}"
            print(row)

    def export_csv(self, path: str) -> None:
        import csv as csv_mod
        t = self._prepare_table_data()
        if t is None:
            return
        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]
        def _cr(label, mw_val, values):
            row = [label, mw_val]
            for v in values: row.extend([v, ""])
            return row
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(_cr("No.", "", [str(i+1) for i in range(len(names))]))
            w.writerow(_cr("Service", "", names))
            w.writerow(_cr("Press.", "MPaG", pressures))
            w.writerow(_cr("Temp.", "°C", temperatures))
            w.writerow(_cr("Phase", "", phases))
            for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in [
                ("mol","mol/h","mol%","mol","mol_frac","total_mol"),
                ("Volume","NL/h","vol%","nvol","vol_frac","total_nvol"),
                ("weight","g/h","wt%","mass","mass_frac","total_mass"),
            ]:
                unit_row = ["Component", "MW"]
                for _ in names: unit_row.extend([abs_unit, rel_unit])
                w.writerow(unit_row)
                for i, formula in enumerate(all_formulas):
                    row = [formula, f"{mw_map[formula]:.2f}"]
                    for d in data: row.extend([f"{d[abs_key][i]:.4f}", f"{d[rel_key][i]:.4f}"])
                    w.writerow(row)
                total_row = ["Total", ""]
                for d in data: total_row.extend([f"{d[total_key]:.4f}", "1.0000"])
                w.writerow(total_row)
                w.writerow([])


    def export_excel(self, filename: str, sheet: str, cell: str = "A1") -> None:
        try:
            import win32com.client
        except ImportError:
            raise RuntimeError("win32com が必要です。pip install pywin32 を実行してください。")
        t = self._prepare_table_data()
        if t is None:
            raise ValueError("出力するストリームデータがありません。")
        all_formulas, mw_map, data, names = t["all_formulas"], t["mw_map"], t["data"], t["names"]
        pressures, temperatures, phases = t["pressures"], t["temperatures"], t["phases"]
        try:
            xl = win32com.client.GetActiveObject("Excel.Application")
        except Exception:
            raise RuntimeError("Excel が起動していません。")
        wb = None
        for i in range(1, xl.Workbooks.Count + 1):
            w = xl.Workbooks(i)
            if w.Name == filename or w.FullName == filename:
                wb = w
                break
        if wb is None:
            available = [xl.Workbooks(i).Name for i in range(1, xl.Workbooks.Count + 1)]
            raise ValueError(f"ファイル '{filename}' が開かれていません。開いているファイル: {available}")
        ws = None
        for i in range(1, wb.Sheets.Count + 1):
            s = wb.Sheets(i)
            if s.Name == sheet:
                ws = s
                break
        if ws is None:
            available = [wb.Sheets(i).Name for i in range(1, wb.Sheets.Count + 1)]
            raise ValueError(f"シート '{sheet}' が存在しません。存在するシート: {available}")
        start = ws.Range(cell)
        row0, col0 = start.Row, start.Column
        r = row0
        def _xr(label, mw_val, values):
            nonlocal r
            ws.Cells(r, col0).Value = label
            if mw_val:
                ws.Cells(r, col0 + 1).Value = mw_val
            for si, v in enumerate(values):
                ws.Cells(r, col0 + 2 + si * 2).Value = v
            r += 1
        _xr("No.", "", [str(i+1) for i in range(len(names))])
        _xr("Service", "", names)
        _xr("Press.", "MPaG", pressures)
        _xr("Temp.", "°C", temperatures)
        _xr("Phase", "", phases)
        for sec_name, abs_unit, rel_unit, abs_key, rel_key, total_key in [
            ("mol","mol/h","mol%","mol","mol_frac","total_mol"),
            ("Volume","NL/h","vol%","nvol","vol_frac","total_nvol"),
            ("weight","g/h","wt%","mass","mass_frac","total_mass"),
        ]:
            ws.Cells(r, col0).Value = "Component"
            ws.Cells(r, col0 + 1).Value = "MW"
            for si in range(len(names)):
                ws.Cells(r, col0 + 2 + si * 2).Value = abs_unit
                ws.Cells(r, col0 + 3 + si * 2).Value = rel_unit
            r += 1
            for i, formula in enumerate(all_formulas):
                ws.Cells(r, col0).Value = formula
                ws.Cells(r, col0 + 1).Value = round(mw_map[formula], 2)
                for si, d in enumerate(data):
                    ws.Cells(r, col0 + 2 + si * 2).Value = round(d[abs_key][i], 4)
                    ws.Cells(r, col0 + 3 + si * 2).Value = round(d[rel_key][i], 4)
                r += 1
            ws.Cells(r, col0).Value = "Total"
            for si, d in enumerate(data):
                ws.Cells(r, col0 + 2 + si * 2).Value = round(d[total_key], 4)
                ws.Cells(r, col0 + 3 + si * 2).Value = 1.0
            r += 1


# ============================================================
# Global Flowsheet
# ============================================================

_flowsheet: Flowsheet | None = None


def _get_flowsheet() -> Flowsheet:
    global _flowsheet
    if _flowsheet is None:
        _flowsheet = Flowsheet("Global")
    return _flowsheet


def reset() -> None:
    """グローバル Flowsheet をクリアする。"""
    global _flowsheet
    _flowsheet = None
    ComponentRegistry.clear_cache()


def solve(**kwargs):
    """連立方程式を求解する。"""
    return _get_flowsheet().solve(**kwargs)


def print_streams() -> None:
    """全ストリームの結果を一覧表示する。"""
    _get_flowsheet().print_streams()


def set_component_order(order: list[str]) -> None:
    """出力時の成分表示順序を設定する。"""
    _get_flowsheet().set_component_order(order)


def export_csv(path: str) -> None:
    """全ストリームの結果をCSVファイルに出力する。"""
    _get_flowsheet().export_csv(path)


def export_excel(filename: str, sheet: str, cell: str = "A1") -> None:
    """開いている Excel ブックのシートに結果を出力する。"""
    _get_flowsheet().export_excel(filename, sheet, cell)


# ============================================================
# Expression (遅延評価)
# ============================================================

class StreamExpression:
    def __init__(self):
        self._materialized: Stream | None = None

    def materialize(self, target: Stream | None = None) -> Stream:
        raise NotImplementedError

    def _ensure_materialized(self) -> Stream:
        if self._materialized is None:
            self._materialized = self.materialize()
        return self._materialized

    def react(self, stoichiometry, key, conversion):
        return self._ensure_materialized().react(stoichiometry, key, conversion)

    def gibbs_react(self, T, P, species):
        return self._ensure_materialized().gibbs_react(T, P, species)

    def multi_react(self, reactions, key, conversion, selectivities):
        return self._ensure_materialized().multi_react(reactions, key, conversion, selectivities)

    def separate_water(self, *args, **kwargs):
        return self._ensure_materialized().separate_water(*args, **kwargs)

    def absorb(self, *args, **kwargs):
        return self._ensure_materialized().absorb(*args, **kwargs)

    def __add__(self, other):
        return self._ensure_materialized().__add__(other)

    def __radd__(self, other):
        return self._ensure_materialized().__radd__(other)

    def __mul__(self, ratio):
        return self._ensure_materialized().__mul__(ratio)

    def __rmul__(self, ratio):
        return self._ensure_materialized().__rmul__(ratio)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure_materialized(), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif self._materialized is not None:
            setattr(self._materialized, name, value)
        else:
            setattr(self._ensure_materialized(), name, value)


class MixExpression(StreamExpression):
    def __init__(self, operands: list):
        super().__init__()
        self._operands = operands

    def materialize(self, target: Stream | None = None) -> Stream:
        if self._materialized is not None and target is None:
            return self._materialized
        resolved = []
        for op in self._operands:
            resolved.append(op._ensure_materialized() if isinstance(op, StreamExpression) else op)
        all_formulas = []
        seen = set()
        for s in resolved:
            for c in s.components:
                if c.formula not in seen:
                    all_formulas.append(c.formula)
                    seen.add(c.formula)
        if target is not None:
            outlet = target
            for f in all_formulas:
                if f not in [c.formula for c in outlet.components]:
                    outlet._add_component(f)
        else:
            outlet = Stream(components=all_formulas, _internal=True)
        for s in resolved:
            for f in all_formulas:
                if f not in [c.formula for c in s.components]:
                    s._add_component(f)
        _get_flowsheet().add_unit(Mixer("MIX_auto", inlets=resolved, outlet=outlet))
        if target is None:
            self._materialized = outlet
        return outlet

    def __add__(self, other):
        if isinstance(other, MixExpression):
            return MixExpression(self._operands + other._operands)
        return MixExpression(self._operands + [other])

    def __radd__(self, other):
        if isinstance(other, MixExpression):
            return MixExpression(other._operands + self._operands)
        return MixExpression([other] + self._operands)


class ScaleExpression(StreamExpression):
    def __init__(self, stream, ratio: float):
        super().__init__()
        self._stream = stream
        self._ratio = ratio

    def materialize(self, target: Stream | None = None) -> Stream:
        if self._materialized is not None and target is None:
            return self._materialized
        inlet = self._stream
        if isinstance(inlet, StreamExpression):
            inlet = inlet._ensure_materialized()
        if target is not None:
            outlet = target
        else:
            outlet = Stream(components=[c.formula for c in inlet.components], _internal=True)
        _get_flowsheet().add_unit(
            Splitter("SPL_auto", inlet=inlet, outlets=[outlet], ratios=[self._ratio])
        )
        if target is None:
            self._materialized = outlet
        return outlet


# ============================================================
# Stream
# ============================================================

class Stream:
    def __init__(
        self,
        flows: dict[str, float | tuple[float, str]] | None = None,
        *,
        basis: str = "mol",
        total: float | None = None,
        total_basis: str | None = None,
        components: list[str] | None = None,
        composition: Stream | None = None,
        name: str | None = None,
        T: float | None = None,
        P: float | str | None = None,
        phase: str | None = None,
        _internal: bool = False,
    ):
        self.name = name
        self.T_celsius = T
        self.P_input = P
        self.phase = phase
        self._fixed = False
        self._composition_constraints: list = []
        self._original_formulas: set[str] | None = None

        if composition is not None:
            self.components = list(composition.components)
            self.n_components = len(self.components)
            self.molar_flows = np.ones(self.n_components)
            self._register_composition_constraint(composition)
            self._auto_register()
            return

        if components is not None and flows is None:
            if not _internal:
                self._original_formulas = set(components)
            self.components = ComponentRegistry.get_many(components)
            self.n_components = len(self.components)
            self.molar_flows = np.ones(self.n_components)
            self._auto_register()
            return

        if flows is None:
            self.components = []
            self.n_components = 0
            self.molar_flows = np.array([])
            self._auto_register()
            return

        self._init_from_flows(flows, basis, total, total_basis)
        self._auto_register()

    def _init_from_flows(self, flows, basis, total, total_basis=None):
        first_val = next(iter(flows.values()))
        if isinstance(first_val, (tuple, list)):
            self._init_from_tuple_flows(flows)
            return
        formulas = list(flows.keys())
        values = np.array(list(flows.values()), dtype=float)
        self.components = ComponentRegistry.get_many(formulas)
        self.n_components = len(self.components)
        mws = np.array([c.mw for c in self.components])
        nvols = np.array([c.normal_volume for c in self.components])
        abs_bases = {"mol", "mass", "normal_volume"}
        frac_bases = {"mole_frac", "mass_frac", "volume_frac"}
        if basis in abs_bases:
            self.molar_flows = self._convert_abs(values, basis, mws, nvols)
            self._fixed = True
        elif basis in frac_bases:
            if total is not None:
                self.molar_flows = self._convert_frac(values, basis, total, total_basis, mws, nvols)
                self._fixed = True
            else:
                self.molar_flows = np.ones(self.n_components)
                self._register_frac_constraint(values, basis)
        else:
            raise BasisError(f"Unknown basis: '{basis}'")

    def _init_from_tuple_flows(self, flows: dict):
        formulas = list(flows.keys())
        self.components = ComponentRegistry.get_many(formulas)
        self.n_components = len(self.components)
        self.molar_flows = np.zeros(self.n_components)
        for i, (_, val_unit) in enumerate(flows.items()):
            value, unit = val_unit
            mw, nvol = self.components[i].mw, self.components[i].normal_volume
            if unit == "mol":
                self.molar_flows[i] = value
            elif unit == "mass":
                self.molar_flows[i] = value / mw
            elif unit == "normal_volume":
                self.molar_flows[i] = value / nvol
            else:
                raise BasisError(f"Tuple basis must be 'mol','mass','normal_volume', got '{unit}'")
        self._fixed = True

    @staticmethod
    def _convert_abs(values, basis, mws, nvols):
        if basis == "mol":
            return values.copy()
        elif basis == "mass":
            return values / mws
        elif basis == "normal_volume":
            return values / nvols
        raise BasisError(f"Unknown absolute basis: '{basis}'")

    @staticmethod
    def _convert_frac(fracs, basis, total, total_basis, mws, nvols):
        fracs_norm = fracs / fracs.sum()
        if total_basis is None:
            default_map = {"mole_frac": "mol", "mass_frac": "mass", "volume_frac": "normal_volume"}
            total_basis = default_map.get(basis, "mol")
        if total_basis == "mol":
            if basis == "mole_frac":
                return fracs_norm * total
            elif basis == "mass_frac":
                mol_ratios = fracs_norm / mws; mol_ratios = mol_ratios / mol_ratios.sum()
                return mol_ratios * total
            elif basis == "volume_frac":
                mol_ratios = fracs_norm / nvols; mol_ratios = mol_ratios / mol_ratios.sum()
                return mol_ratios * total
        elif total_basis == "mass":
            if basis == "mole_frac":
                avg_mw = np.sum(fracs_norm * mws)
                return fracs_norm * (total / avg_mw)
            elif basis == "mass_frac":
                return (fracs_norm * total) / mws
            elif basis == "volume_frac":
                mol_ratios = fracs_norm / nvols; mol_ratios = mol_ratios / mol_ratios.sum()
                avg_mw = np.sum(mol_ratios * mws)
                return mol_ratios * (total / avg_mw)
        elif total_basis == "normal_volume":
            if basis == "volume_frac":
                return (fracs_norm * total) / nvols
            elif basis == "mole_frac":
                vol_ratios = fracs_norm * nvols; vol_ratios = vol_ratios / vol_ratios.sum()
                return (vol_ratios * total) / nvols
            elif basis == "mass_frac":
                mol_ratios = fracs_norm / mws; mol_ratios = mol_ratios / mol_ratios.sum()
                vol_ratios = mol_ratios * nvols; vol_ratios = vol_ratios / vol_ratios.sum()
                return (vol_ratios * total) / nvols
        raise BasisError(f"Unknown total_basis: '{total_basis}'")

    def _register_frac_constraint(self, fracs, basis):
        fracs = fracs / fracs.sum()
        original_fracs = dict(zip([c.formula for c in self.components], fracs))
        def constraint():
            target = np.zeros(self.n_components)
            for i, c in enumerate(self.components):
                target[i] = original_fracs.get(c.formula, 0.0)
            s = target.sum()
            if s > 0:
                target = target / s
            mws = np.array([c.mw for c in self.components])
            nvols = np.array([c.normal_volume for c in self.components])
            if basis == "mole_frac":
                current = self.mole_fractions
            elif basis == "mass_frac":
                mass = self.molar_flows * mws
                t = mass.sum()
                current = mass / t if t > 0 else np.zeros_like(mass)
            elif basis == "volume_frac":
                vol = self.molar_flows * nvols
                t = vol.sum()
                current = vol / t if t > 0 else np.zeros_like(vol)
            else:
                return np.array([])
            return current[:-1] - target[:-1]
        self._composition_constraints.append(constraint)

    def _register_composition_constraint(self, other: Stream):
        def constraint():
            other_frac_dict = {c.formula: other.mole_fractions[i] for i, c in enumerate(other.components)}
            target = np.zeros(self.n_components)
            for i, c in enumerate(self.components):
                target[i] = other_frac_dict.get(c.formula, 0.0)
            return self.mole_fractions[:-1] - target[:-1]
        self._composition_constraints.append(constraint)

    def _auto_register(self):
        fs = _get_flowsheet()
        if self not in fs.streams:
            fs.add_stream(self)

    def _add_component(self, formula: str):
        if any(c.formula == formula for c in self.components):
            return
        self.components.append(ComponentRegistry.get(formula))
        self.n_components = len(self.components)
        if self._fixed:
            self.molar_flows = np.append(self.molar_flows, 0.0)
        elif self._original_formulas is not None and formula not in self._original_formulas:
            self.molar_flows = np.append(self.molar_flows, 0.0)
            idx = self.n_components - 1
            self._composition_constraints.append(lambda i=idx: np.array([self.molar_flows[i]]))
        else:
            self.molar_flows = np.append(self.molar_flows, 0.1)

    # --- Properties ---
    @property
    def total_molar_flow(self) -> float:
        return float(np.sum(self.molar_flows))

    @property
    def mole_fractions(self) -> np.ndarray:
        t = self.total_molar_flow
        return self.molar_flows / t if t != 0 else np.zeros(self.n_components)

    @property
    def mass_flows(self) -> np.ndarray:
        return self.molar_flows * np.array([c.mw for c in self.components])

    @property
    def total_mass_flow(self) -> float:
        return float(np.sum(self.mass_flows))

    @property
    def mass_fractions(self) -> np.ndarray:
        t = self.total_mass_flow
        return self.mass_flows / t if t != 0 else np.zeros(self.n_components)

    @property
    def normal_volume_flows(self) -> np.ndarray:
        return self.molar_flows * np.array([c.normal_volume for c in self.components])

    @property
    def total_normal_volume_flow(self) -> float:
        return float(np.sum(self.normal_volume_flows))

    @property
    def volume_fractions(self) -> np.ndarray:
        t = self.total_normal_volume_flow
        return self.normal_volume_flows / t if t != 0 else np.zeros(self.n_components)

    # --- Operators ---
    def __add__(self, other) -> MixExpression:
        if isinstance(other, MixExpression):
            return MixExpression([self] + other._operands)
        return MixExpression([self, other])

    def __radd__(self, other) -> MixExpression:
        if isinstance(other, MixExpression):
            return MixExpression(other._operands + [self])
        if other == 0:
            return MixExpression([self])
        return MixExpression([other, self])

    def __mul__(self, ratio: float) -> ScaleExpression:
        return ScaleExpression(self, float(ratio))

    def __rmul__(self, ratio: float) -> ScaleExpression:
        return ScaleExpression(self, float(ratio))

    # --- Reactors ---
    def react(self, stoichiometry: dict[str, float], key: str, conversion: float) -> Stream:
        outlet_formulas = [c.formula for c in self.components]
        for f in stoichiometry:
            if f not in outlet_formulas:
                outlet_formulas.append(f)
        outlet = Stream(components=outlet_formulas, _internal=True)
        for f in outlet_formulas:
            self._add_component(f)
        stoich_array = np.zeros(len(outlet_formulas))
        for f, coeff in stoichiometry.items():
            stoich_array[outlet_formulas.index(f)] = coeff
        _get_flowsheet().add_unit(Reactor(
            "RX_auto", inlet=self, outlet=outlet,
            stoichiometry=stoich_array.tolist(),
            key_component=outlet_formulas.index(key), conversion=conversion,
        ))
        return outlet

    def gibbs_react(self, T: float, P: float | str, species: list[str]) -> Stream:
        P_pascal = parse_pressure(P)
        outlet = Stream(components=list(species), _internal=True)
        for f in species:
            self._add_component(f)
        _get_flowsheet().add_unit(GibbsReactor(
            inlet=self, outlet=outlet,
            T_celsius=T, P_pascal=P_pascal, species=species,
        ))
        return outlet

    def multi_react(self, reactions, key, conversion, selectivities) -> Stream:
        outlet_formulas = [c.formula for c in self.components]
        for rxn in reactions:
            for f in rxn:
                if f not in outlet_formulas:
                    outlet_formulas.append(f)
        outlet = Stream(components=outlet_formulas, _internal=True)
        for f in outlet_formulas:
            self._add_component(f)
        _get_flowsheet().add_unit(MultiReactor(
            "MRX_auto", inlet=self, outlet=outlet,
            reactions=reactions, key=key,
            conversion=conversion, selectivities=selectivities,
        ))
        return outlet

    def separate_water(self, T, P, name_gas=None, name_water=None, henry_constants=None):
        P_pascal = parse_pressure(P)
        gas_formulas = [c.formula for c in self.components]
        if "H2O" not in gas_formulas:
            gas_formulas.append("H2O")
        gas_outlet = Stream(components=gas_formulas, name=name_gas, _internal=True)
        water_outlet = Stream(components=gas_formulas, name=name_water, _internal=True)
        _get_flowsheet().add_unit(WaterSeparator(
            "SEP_auto", inlet=self, gas_outlet=gas_outlet,
            water_outlet=water_outlet, T_celsius=T, P_pascal=P_pascal,
            henry_constants=henry_constants,
        ))
        return gas_outlet, water_outlet

    def absorb(self, water_flow, T, P, stages=10, name_gas=None, name_liquid=None, henry_constants=None):
        P_pascal = parse_pressure(P)
        gas_formulas = [c.formula for c in self.components]
        if "H2O" not in gas_formulas:
            gas_formulas.append("H2O")
        water_inlet = Stream({"H2O": water_flow})
        gas_outlet = Stream(components=gas_formulas, name=name_gas, _internal=True)
        liquid_outlet = Stream(components=gas_formulas, name=name_liquid, _internal=True)
        _get_flowsheet().add_unit(Absorber(
            "ABS_auto", gas_inlet=self, water_inlet=water_inlet,
            gas_outlet=gas_outlet, liquid_outlet=liquid_outlet,
            T_celsius=T, P_pascal=P_pascal, stages=stages,
            henry_constants=henry_constants,
        ))
        return gas_outlet, liquid_outlet

    @classmethod
    def from_csv(cls, path: str, name: str | None = None, **kwargs) -> Stream:
        flows = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                flows[row["component"].strip()] = float(row["molflow"])
        return cls(flows, name=name, **kwargs)

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        flows = ", ".join(f"{c.formula}={f:.4g}" for c, f in zip(self.components, self.molar_flows))
        return f"Stream({label!r}: {flows})"


# ============================================================
# API: eq / constrain
# ============================================================

def eq(target: Stream, expression) -> None:
    """等式制約: eq(C, A + B) → C = A + B"""
    if isinstance(expression, StreamExpression):
        expression.materialize(target=target)
    else:
        def residual():
            return target.molar_flows - expression.molar_flows
        _get_flowsheet().add_spec(residual)


def constrain(residual_func: Callable) -> None:
    """任意制約: constrain(lambda: C.total_molar_flow - 30)"""
    _get_flowsheet().add_spec(lambda: np.atleast_1d(residual_func()))
