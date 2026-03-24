"""chemflow_portable.py — 単一ファイル版 化学プロセスシミュレーター

依存: numpy, scipy, molmass (必須), cantera (gibbs_react使用時のみ)

使い方:
    from chemflow_portable import Stream, eq, constrain, solve, reset, print_streams
"""

from __future__ import annotations

import csv
import re
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
            fw = max(max(len(f) for f in formulas), 5)
            col = 10
            print(f"\n{'=' * 70}")
            print(f" {label}")
            print(f"{'=' * 70}")
            header = f"  {'':>{fw}s}"
            for f in formulas:
                header += f"  {f:>{col}s}"
            header += f"  {'Total':>{col}s}"
            print(header)
            print(f"  {'':>{fw}s}" + ("  " + "-" * col) * (n + 1))
            for row_label, vals, total_val in [
                ("mol", mol, s.total_molar_flow),
                ("x", x, None),
                ("mass", mass, s.total_mass_flow),
                ("w", w, None),
                ("Nm3", nvol, s.total_normal_volume_flow),
                ("vf", vf, None),
            ]:
                row = f"  {row_label:>{fw}s}"
                for v in vals:
                    row += f"  {v:{col}.4f}"
                row += f"  {total_val:{col}.4f}" if total_val is not None else f"  {'1.0000':>{col}s}"
                print(row)


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
            outlet = Stream(components=all_formulas)
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
            outlet = Stream(components=[c.formula for c in inlet.components])
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
        components: list[str] | None = None,
        composition: Stream | None = None,
        name: str | None = None,
    ):
        self.name = name
        self._fixed = False
        self._composition_constraints: list = []

        if composition is not None:
            self.components = list(composition.components)
            self.n_components = len(self.components)
            self.molar_flows = np.ones(self.n_components)
            self._register_composition_constraint(composition)
            self._auto_register()
            return

        if components is not None and flows is None:
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

        self._init_from_flows(flows, basis, total)
        self._auto_register()

    def _init_from_flows(self, flows: dict, basis: str, total: float | None):
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
                self.molar_flows = self._convert_frac(values, basis, total, mws, nvols)
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
    def _convert_frac(fracs, basis, total, mws, nvols):
        if basis == "mole_frac":
            return fracs * total
        elif basis == "mass_frac":
            return (fracs * total) / mws
        elif basis == "volume_frac":
            return (fracs * total) / nvols
        raise BasisError(f"Unknown fraction basis: '{basis}'")

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
        self.molar_flows = np.append(self.molar_flows, 0.0 if self._fixed else 0.1)

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
        outlet = Stream(components=outlet_formulas)
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
        outlet = Stream(components=list(species))
        for f in species:
            self._add_component(f)
        _get_flowsheet().add_unit(GibbsReactor(
            inlet=self, outlet=outlet,
            T_celsius=T, P_pascal=P_pascal, species=species,
        ))
        return outlet

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
