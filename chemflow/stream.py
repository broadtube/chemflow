"""Stream クラス（リデザイン版）

内部状態は各成分のモル流量のみ保持。
示性式文字列から分子量を自動計算。
演算子で装置接続を表現。
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

from chemflow.component import Component
from chemflow.errors import BasisError
from chemflow.registry import ComponentRegistry

if TYPE_CHECKING:
    from chemflow.expression import MixExpression, ScaleExpression


class Stream:
    """プロセスストリーム。

    Examples
    --------
    固定:       Stream({"H2": 20, "N2": 60})
    basis指定:  Stream({"H2": 20, "N2": 60}, basis="mass")
    比率+total: Stream({"H2": 0.75, "N2": 0.25}, basis="mole_frac", total=100)
    組成のみ:   Stream({"H2": 0.75, "N2": 0.25}, basis="mole_frac")
    未知:       Stream(components=["H2", "N2"])
    同組成:     Stream(composition=other_stream)
    タプル混在: Stream({"N2": (20, "mol"), "H2": (120, "mass")})
    """

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
        **kwargs,
    ):
        self.name = name
        self.T_celsius = T  # 温度 [°C]
        self.P_input = P    # 圧力（数値[Pa] or 文字列 "3MPaG" 等）
        self.phase = phase  # "Gas", "Liquid", "Solid", "Mixed"
        self._fixed = False
        self._composition_constraints: list = []
        self._original_formulas: set[str] | None = None  # 初期成分リスト（動的追加の0拘束用）

        if composition is not None:
            # 他ストリームと同組成、total未知
            self.components = list(composition.components)
            self.n_components = len(self.components)
            self.molar_flows = np.ones(self.n_components)  # 初期推定
            self._fixed = False
            self._register_composition_constraint(composition)
            self._auto_register()
            return

        if components is not None and flows is None:
            # 未知ストリーム
            # _internal=True の場合は内部生成（0拘束しない）
            if not _internal:
                self._original_formulas = set(components)
            self.components = ComponentRegistry.get_many(components)
            self.n_components = len(self.components)
            self.molar_flows = np.ones(self.n_components)  # 初期推定（非ゼロ）
            self._fixed = False
            self._auto_register()
            return

        if flows is None:
            self.components = []
            self.n_components = 0
            self.molar_flows = np.array([])
            self._fixed = False
            self._auto_register()
            return

        # flows dict あり
        self._init_from_flows(flows, basis, total, total_basis)
        self._auto_register()

    def _init_from_flows(self, flows: dict, basis: str, total: float | None,
                         total_basis: str | None = None):
        """flows dict からモル流量を初期化する。"""
        # タプル形式チェック: {"N2": (20, "mol"), "H2": (120, "mass")}
        first_val = next(iter(flows.values()))
        is_tuple = isinstance(first_val, (tuple, list))

        if is_tuple:
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
            self.molar_flows = self._convert_abs_to_mol(values, basis, mws, nvols)
            self._fixed = True
        elif basis in frac_bases:
            if total is not None:
                self.molar_flows = self._convert_frac_to_mol(
                    values, basis, total, total_basis, mws, nvols
                )
                self._fixed = True
            else:
                # 組成のみ、total 未知
                self.molar_flows = np.ones(self.n_components)  # 初期推定
                self._fixed = False
                self._register_frac_constraint(values, basis)
        else:
            raise BasisError(f"Unknown basis: '{basis}'")

    def _init_from_tuple_flows(self, flows: dict):
        """成分ごとに単位を指定するタプル形式。"""
        formulas = list(flows.keys())
        self.components = ComponentRegistry.get_many(formulas)
        self.n_components = len(self.components)
        self.molar_flows = np.zeros(self.n_components)

        for i, (formula, val_unit) in enumerate(flows.items()):
            value, unit = val_unit
            mw = self.components[i].mw
            nvol = self.components[i].normal_volume
            if unit == "mol":
                self.molar_flows[i] = value
            elif unit == "mass":
                self.molar_flows[i] = value / mw
            elif unit == "normal_volume":
                self.molar_flows[i] = value / nvol
            else:
                raise BasisError(
                    f"Tuple basis must be 'mol', 'mass', or 'normal_volume', got '{unit}'"
                )
        self._fixed = True

    @staticmethod
    def _convert_abs_to_mol(values, basis, mws, nvols):
        if basis == "mol":
            return values.copy()
        elif basis == "mass":
            return values / mws
        elif basis == "normal_volume":
            return values / nvols
        raise BasisError(f"Unknown absolute basis: '{basis}'")

    @staticmethod
    def _convert_frac_to_mol(fracs, basis, total, total_basis, mws, nvols):
        """比率 + total から各成分のモル流量を計算する。

        total_basis が None の場合は basis に連動:
          mole_frac → mol, mass_frac → mass, volume_frac → normal_volume
        total_basis を明示指定すると basis と独立に total の単位を指定できる:
          例: basis="mole_frac", total=1000, total_basis="mass" → モル比率で1000g/h
        """
        # fracs を正規化
        fracs_norm = fracs / fracs.sum()

        # total_basis のデフォルト
        if total_basis is None:
            default_map = {"mole_frac": "mol", "mass_frac": "mass", "volume_frac": "normal_volume"}
            total_basis = default_map.get(basis, "mol")

        if total_basis == "mol":
            # total = total mol/h
            return fracs_norm * total
        elif total_basis == "mass":
            # total = total g/h
            # まずモル比率からモル流量（仮）を求め、平均分子量を計算
            if basis == "mole_frac":
                avg_mw = np.sum(fracs_norm * mws)
                total_mol = total / avg_mw
                return fracs_norm * total_mol
            elif basis == "mass_frac":
                return (fracs_norm * total) / mws
            elif basis == "volume_frac":
                # vol比率 → mol比率に変換してから
                mol_ratios = fracs_norm / nvols
                mol_ratios = mol_ratios / mol_ratios.sum()
                avg_mw = np.sum(mol_ratios * mws)
                total_mol = total / avg_mw
                return mol_ratios * total_mol
        elif total_basis == "normal_volume":
            # total = total NL/h
            if basis == "volume_frac":
                return (fracs_norm * total) / nvols
            elif basis == "mole_frac":
                # mol比率 → vol比率に変換
                vol_ratios = fracs_norm * nvols
                vol_ratios = vol_ratios / vol_ratios.sum()
                total_vol_per_comp = vol_ratios * total
                return total_vol_per_comp / nvols
            elif basis == "mass_frac":
                # mass比率 → mol比率
                mol_ratios = fracs_norm / mws
                mol_ratios = mol_ratios / mol_ratios.sum()
                vol_ratios = mol_ratios * nvols
                vol_ratios = vol_ratios / vol_ratios.sum()
                total_vol_per_comp = vol_ratios * total
                return total_vol_per_comp / nvols

        raise BasisError(f"Unknown total_basis: '{total_basis}'")

    def _register_frac_constraint(self, fracs, basis):
        """比率系 basis で total なしの場合の組成制約。"""
        fracs = fracs / fracs.sum()  # 正規化
        # 元の成分リストを記録（後から追加された成分は0に拘束）
        original_formulas = [c.formula for c in self.components]
        original_fracs = dict(zip(original_formulas, fracs))

        def constraint():
            # 全成分（動的追加含む）に対する比率目標を構築
            target = np.zeros(self.n_components)
            for i, c in enumerate(self.components):
                target[i] = original_fracs.get(c.formula, 0.0)
            # 再正規化（追加成分のfrac=0なので元と同じだが安全のため）
            s = target.sum()
            if s > 0:
                target = target / s

            mws = np.array([c.mw for c in self.components])
            nvols = np.array([c.normal_volume for c in self.components])
            if basis == "mole_frac":
                current = self.mole_fractions
            elif basis == "mass_frac":
                mass = self.molar_flows * mws
                total_mass = mass.sum()
                current = mass / total_mass if total_mass > 0 else np.zeros_like(mass)
            elif basis == "volume_frac":
                vol = self.molar_flows * nvols
                total_vol = vol.sum()
                current = vol / total_vol if total_vol > 0 else np.zeros_like(vol)
            else:
                return np.array([])
            # n-1 個の独立な制約
            return current[:-1] - target[:-1]

        self._composition_constraints.append(constraint)

    def _register_composition_constraint(self, other: Stream):
        """他ストリームと同じ組成（モル比率）を持つ制約。

        成分が動的追加されても対応する。
        自身の成分順で other の成分を参照する。
        """
        def constraint():
            # other の組成を取得して、self の成分順にマッピング
            other_frac_dict = {}
            for i, c in enumerate(other.components):
                other_frac_dict[c.formula] = other.mole_fractions[i]

            target = np.zeros(self.n_components)
            for i, c in enumerate(self.components):
                target[i] = other_frac_dict.get(c.formula, 0.0)

            my_frac = self.mole_fractions
            # n-1 個の独立な制約
            return my_frac[:-1] - target[:-1]

        self._composition_constraints.append(constraint)

    def _auto_register(self):
        """グローバル Flowsheet に自動登録。"""
        from chemflow.global_flowsheet import _get_flowsheet
        fs = _get_flowsheet()
        if self not in fs.streams:
            fs.add_stream(self)

    def _add_component(self, formula: str):
        """成分を動的に追加する。

        固定ストリーム: 0で追加。
        未知ストリーム（_original_formulasあり）: 元の成分リストにない成分は0拘束を自動登録。
        未知ストリーム（_original_formulasなし）: 初期推定値0.1で追加（変数として扱う）。
        """
        existing = {c.formula for c in self.components}
        if formula in existing:
            return
        comp = ComponentRegistry.get(formula)
        self.components.append(comp)
        self.n_components = len(self.components)

        if self._fixed:
            self.molar_flows = np.append(self.molar_flows, 0.0)
        elif self._original_formulas is not None and formula not in self._original_formulas:
            # 元の成分リストにない → 0拘束
            self.molar_flows = np.append(self.molar_flows, 0.0)
            idx = self.n_components - 1
            self._composition_constraints.append(lambda i=idx: np.array([self.molar_flows[i]]))
        else:
            self.molar_flows = np.append(self.molar_flows, 0.1)

    # --- プロパティ ---

    @property
    def total_molar_flow(self) -> float:
        return float(np.sum(self.molar_flows))

    @property
    def mole_fractions(self) -> np.ndarray:
        total = self.total_molar_flow
        if total == 0:
            return np.zeros(self.n_components)
        return self.molar_flows / total

    @property
    def mass_flows(self) -> np.ndarray:
        mws = np.array([c.mw for c in self.components])
        return self.molar_flows * mws

    @property
    def total_mass_flow(self) -> float:
        return float(np.sum(self.mass_flows))

    @property
    def mass_fractions(self) -> np.ndarray:
        total = self.total_mass_flow
        if total == 0:
            return np.zeros(self.n_components)
        return self.mass_flows / total

    @property
    def normal_volume_flows(self) -> np.ndarray:
        nvols = np.array([c.normal_volume for c in self.components])
        return self.molar_flows * nvols

    @property
    def total_normal_volume_flow(self) -> float:
        return float(np.sum(self.normal_volume_flows))

    @property
    def volume_fractions(self) -> np.ndarray:
        total = self.total_normal_volume_flow
        if total == 0:
            return np.zeros(self.n_components)
        return self.normal_volume_flows / total

    # --- 演算子 ---

    def __add__(self, other) -> MixExpression:
        from chemflow.expression import MixExpression
        if isinstance(other, MixExpression):
            return MixExpression([self] + other._operands)
        return MixExpression([self, other])

    def __radd__(self, other) -> MixExpression:
        from chemflow.expression import MixExpression
        if isinstance(other, MixExpression):
            return MixExpression(other._operands + [self])
        if other == 0:
            # sum() 対応
            return MixExpression([self])
        return MixExpression([other, self])

    def __mul__(self, ratio: float) -> ScaleExpression:
        from chemflow.expression import ScaleExpression
        return ScaleExpression(self, float(ratio))

    def __rmul__(self, ratio: float) -> ScaleExpression:
        from chemflow.expression import ScaleExpression
        return ScaleExpression(self, float(ratio))

    # --- リアクター ---

    def react(
        self,
        stoichiometry: dict[str, float],
        key: str,
        conversion: float,
    ) -> Stream:
        """転化率指定リアクター。出口 Stream を返す。"""
        from chemflow.global_flowsheet import _get_flowsheet
        from chemflow.units import Reactor

        # 出口成分 = 入口 + 化学量論に含まれる生成物
        outlet_formulas = [c.formula for c in self.components]
        for formula in stoichiometry:
            if formula not in outlet_formulas:
                outlet_formulas.append(formula)

        outlet = Stream(components=outlet_formulas, _internal=True)

        # 入口にも不足成分を追加（固定ストリームはスキップ）
        if not self._fixed:
            for formula in outlet_formulas:
                self._add_component(formula)

        # 化学量論係数を配列に変換
        stoich_array = np.zeros(len(outlet_formulas))
        for formula, coeff in stoichiometry.items():
            idx = outlet_formulas.index(formula)
            stoich_array[idx] = coeff

        key_idx = outlet_formulas.index(key)

        reactor = Reactor(
            "RX_auto",
            inlet=self,
            outlet=outlet,
            stoichiometry=stoich_array.tolist(),
            key_component=key_idx,
            conversion=conversion,
        )
        _get_flowsheet().add_unit(reactor)

        return outlet

    def gibbs_react(
        self,
        T: float,
        P: float | str,
        species: list[str],
    ) -> Stream:
        """Gibbs リアクター。Cantera 使用。出口 Stream を返す。"""
        from chemflow.global_flowsheet import _get_flowsheet
        from chemflow.gibbs import GibbsReactor, parse_pressure

        P_pascal = parse_pressure(P)

        # 出口成分 = species リスト
        outlet_formulas = list(species)
        outlet = Stream(components=outlet_formulas, _internal=True)

        # 入口にも不足成分を追加（固定ストリームはスキップ — GibbsReactorは
        # formula マッピングで inlet に存在しない成分を 0 として扱える）
        if not self._fixed:
            for formula in outlet_formulas:
                self._add_component(formula)

        gibbs = GibbsReactor(
            inlet=self,
            outlet=outlet,
            T_celsius=T,
            P_pascal=P_pascal,
            species=species,
        )
        _get_flowsheet().add_unit(gibbs)

        return outlet

    def multi_react(
        self,
        reactions: list[dict[str, float]],
        key: str,
        conversion: float,
        selectivities: list[float],
        strict_selectivity: bool = False,
    ) -> Stream:
        """複数同時反応リアクター。

        Parameters
        ----------
        reactions : list[dict[str, float]]
            各反応の化学量論係数リスト
        key : str
            基準成分の示性式
        conversion : float
            基準成分の全体転化率 (0~1)
        selectivities : list[float]
            各反応の選択率（基準成分消費量ベース、合計=1）
        strict_selectivity : bool
            True の場合、各反応の固有生成物から選択率を明示的に拘束する
        """
        from chemflow.global_flowsheet import _get_flowsheet
        from chemflow.units import MultiReactor

        # 全反応から出口成分を収集
        outlet_formulas = [c.formula for c in self.components]
        for rxn in reactions:
            for formula in rxn:
                if formula not in outlet_formulas:
                    outlet_formulas.append(formula)

        outlet = Stream(components=outlet_formulas, _internal=True)

        if not self._fixed:
            for formula in outlet_formulas:
                self._add_component(formula)

        reactor = MultiReactor(
            "MRX_auto",
            inlet=self,
            outlet=outlet,
            reactions=reactions,
            key=key,
            conversion=conversion,
            selectivities=selectivities,
        )
        _get_flowsheet().add_unit(reactor)

        # 厳密な選択率拘束を追加
        if strict_selectivity:
            self._add_selectivity_constraints(
                outlet, reactions, key, selectivities
            )

        return outlet

    def _add_selectivity_constraints(
        self,
        outlet: Stream,
        reactions: list[dict[str, float]],
        key: str,
        selectivities: list[float],
    ) -> None:
        """選択率を明示的に拘束する制約を追加。

        各反応の固有生成物（その反応でのみ生成される成分）を特定し、
        その生成量から選択率を直接拘束する。
        """
        from chemflow.api import constrain

        inlet = self

        # 各成分がどの反応で生成されるかを特定
        product_reactions: dict[str, list[int]] = {}
        for i, rxn in enumerate(reactions):
            for formula, coeff in rxn.items():
                if coeff > 0:  # 生成物
                    if formula not in product_reactions:
                        product_reactions[formula] = []
                    product_reactions[formula].append(i)

        # 各反応の固有生成物（1つの反応でのみ生成）を特定
        markers: dict[int, tuple[str, float]] = {}  # rxn_idx -> (formula, stoich)
        for formula, rxn_indices in product_reactions.items():
            if len(rxn_indices) == 1:
                rxn_idx = rxn_indices[0]
                if rxn_idx not in markers:
                    stoich = reactions[rxn_idx][formula]
                    markers[rxn_idx] = (formula, stoich)

        # 固有生成物がない反応は選択率拘束を追加できない
        missing = [i for i in range(len(reactions)) if i not in markers]
        if missing:
            import warnings
            warnings.warn(
                f"Reactions {missing} have no unique marker product. "
                "Selectivity constraints may not be fully enforced."
            )

        # 選択率拘束を追加（最後の反応以外）
        # constraint: |key_stoich| * (marker_out - marker_in) / marker_stoich
        #           = sel * (key_in - key_out)
        for rxn_idx in range(len(reactions) - 1):
            if rxn_idx not in markers:
                continue

            marker_formula, marker_stoich = markers[rxn_idx]
            key_stoich = abs(reactions[rxn_idx][key])
            sel = selectivities[rxn_idx]

            # インデックスを取得
            inlet_formulas = [c.formula for c in inlet.components]
            outlet_formulas = [c.formula for c in outlet.components]

            marker_in_idx = inlet_formulas.index(marker_formula) if marker_formula in inlet_formulas else None
            marker_out_idx = outlet_formulas.index(marker_formula)
            key_in_idx = inlet_formulas.index(key)
            key_out_idx = outlet_formulas.index(key)

            # クロージャ用にローカル変数をキャプチャ
            _inlet = inlet
            _outlet = outlet
            _marker_in_idx = marker_in_idx
            _marker_out_idx = marker_out_idx
            _key_in_idx = key_in_idx
            _key_out_idx = key_out_idx
            _key_stoich = key_stoich
            _marker_stoich = marker_stoich
            _sel = sel
            _rxn_idx = rxn_idx

            def make_constraint(inlet, outlet, marker_in_idx, marker_out_idx,
                              key_in_idx, key_out_idx, key_stoich, marker_stoich, sel):
                def constraint():
                    marker_in = inlet.molar_flows[marker_in_idx] if marker_in_idx is not None else 0.0
                    marker_out = outlet.molar_flows[marker_out_idx]
                    marker_produced = marker_out - marker_in

                    key_in = inlet.molar_flows[key_in_idx]
                    key_out = outlet.molar_flows[key_out_idx]
                    key_consumed = key_in - key_out

                    # |key_stoich| * marker_produced / marker_stoich = sel * key_consumed
                    lhs = key_stoich * marker_produced / marker_stoich
                    rhs = sel * key_consumed
                    return lhs - rhs
                return constraint

            constraint_fn = make_constraint(
                _inlet, _outlet, _marker_in_idx, _marker_out_idx,
                _key_in_idx, _key_out_idx, _key_stoich, _marker_stoich, _sel
            )

            constrain(
                constraint_fn,
                label=f"Selectivity[{_rxn_idx}]={_sel}",
                code=f"Selectivity constraint for reaction {_rxn_idx}"
            )

    def separate_water(
        self,
        T: float,
        P: float | str,
        name_gas: str | None = None,
        name_water: str | None = None,
        henry_constants: dict[str, float] | None = None,
    ) -> tuple[Stream, Stream]:
        """Antoine式 + Henry則に基づく水分離。

        Parameters
        ----------
        T : float
            温度 [°C]
        P : float or str
            圧力 [Pa] or 文字列 ("3MPaG" 等)
        henry_constants : dict[str, float] | None
            Henry 定数の上書き [Pa]。None で 40°C デフォルト値を使用。

        Returns
        -------
        (gas_outlet, water_outlet)
        """
        from chemflow.global_flowsheet import _get_flowsheet
        from chemflow.gibbs import parse_pressure
        from chemflow.units import WaterSeparator

        P_pascal = parse_pressure(P)

        # ガス出口: 入口と同じ成分（H2O含む）
        gas_formulas = [c.formula for c in self.components]
        if "H2O" not in gas_formulas:
            gas_formulas.append("H2O")
        gas_outlet = Stream(components=gas_formulas, name=name_gas, _internal=True)

        # 液水出口: 全成分（溶解ガスを含む）
        water_outlet = Stream(components=gas_formulas, name=name_water, _internal=True)

        sep = WaterSeparator(
            "SEP_auto",
            inlet=self,
            gas_outlet=gas_outlet,
            water_outlet=water_outlet,
            T_celsius=T,
            P_pascal=P_pascal,
            henry_constants=henry_constants,
        )
        _get_flowsheet().add_unit(sep)
        return gas_outlet, water_outlet

    def absorb(
        self,
        water_flow: float,
        T: float,
        P: float | str,
        stages: int = 10,
        water_basis: str = "mass",
        water_T: float | None = None,
        water_P: float | str | None = None,
        water_phase: str | None = "Liquid",
        name_gas: str | None = None,
        name_liquid: str | None = None,
        name_water: str | None = None,
        henry_constants: dict[str, float] | None = None,
    ) -> tuple[Stream, Stream]:
        """多段吸収塔 (Kremser式)。

        Parameters
        ----------
        water_flow : float
            吸収水の流量（デフォルト: g/h）
        T : float
            温度 [°C]
        P : float or str
            圧力 [Pa] or 文字列 ("3MPaG" 等)
        stages : int
            理論段数 (デフォルト: 10)
        water_basis : str
            水流量の単位 ("mass"=g/h, "mol"=mol/h, "normal_volume"=NL/h)

        Returns
        -------
        (gas_outlet, liquid_outlet)
        """
        from chemflow.global_flowsheet import _get_flowsheet
        from chemflow.gibbs import parse_pressure
        from chemflow.units import Absorber

        P_pascal = parse_pressure(P)

        # 水流量を mol/h に変換
        MW_WATER = 18.015
        NV_WATER = 22.414
        if water_basis == "mass":
            water_flow_mol = water_flow / MW_WATER
        elif water_basis == "normal_volume":
            water_flow_mol = water_flow / NV_WATER
        elif water_basis == "mol":
            water_flow_mol = water_flow
        else:
            water_flow_mol = water_flow / MW_WATER  # デフォルト mass

        # ガス入口の全成分 + H2O
        gas_formulas = [c.formula for c in self.components]
        if "H2O" not in gas_formulas:
            gas_formulas.append("H2O")

        # 水入口ストリーム（内部生成、固定）
        water_inlet = Stream({"H2O": water_flow_mol}, name=name_water,
                             T=water_T, P=water_P, phase=water_phase)

        # ガス出口
        gas_outlet = Stream(components=gas_formulas, name=name_gas, _internal=True)

        # 液出口（全成分）
        liquid_outlet = Stream(components=gas_formulas, name=name_liquid, _internal=True)

        absorber = Absorber(
            "ABS_auto",
            gas_inlet=self,
            water_inlet=water_inlet,
            gas_outlet=gas_outlet,
            liquid_outlet=liquid_outlet,
            T_celsius=T,
            P_pascal=P_pascal,
            stages=stages,
            henry_constants=henry_constants,
        )
        _get_flowsheet().add_unit(absorber)
        return gas_outlet, liquid_outlet

    # --- CSV ---

    @classmethod
    def from_csv(cls, path: str, name: str | None = None, **kwargs) -> Stream:
        """CSV ファイルからストリームを生成する。

        CSV format: component,molflow (header row required)
        """
        flows = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp = row["component"].strip()
                val = float(row["molflow"])
                flows[comp] = val
        return cls(flows, name=name, **kwargs)

    # --- 表示 ---

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        flows = ", ".join(
            f"{c.formula}={f:.4g}"
            for c, f in zip(self.components, self.molar_flows)
        )
        return f"Stream({label!r}: {flows})"
