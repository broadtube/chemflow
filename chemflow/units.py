"""プロセス装置クラス（Mixer, Splitter, Reactor）

各装置は出口値を直接計算せず、満たすべき残差式（= 0）を返す。
成分の順序が異なるストリーム間でも formula 名でマッピングして計算する。
"""

from __future__ import annotations

import numpy as np


def _get_flows_by_formula(stream) -> dict[str, float]:
    """ストリームのモル流量を {formula: flow} の dict で返す。"""
    return {c.formula: stream.molar_flows[i] for i, c in enumerate(stream.components)}


def _build_residual_vector(outlet, flow_dict: dict[str, float]) -> np.ndarray:
    """outlet の成分順に flow_dict の値を並べた残差ベクトルを作る。
    residual = outlet.molar_flows - expected_flows
    """
    expected = np.zeros(outlet.n_components)
    for i, c in enumerate(outlet.components):
        expected[i] = flow_dict.get(c.formula, 0.0)
    return outlet.molar_flows - expected


class Mixer:
    """混合器：複数の入口ストリームを1つの出口に混合する。"""

    def __init__(self, name: str, inlets: list, outlet):
        self.name = name
        self.inlets = inlets
        self.outlet = outlet

    def residuals(self) -> np.ndarray:
        # 入口の合計を formula ベースで計算
        total: dict[str, float] = {}
        for s in self.inlets:
            for formula, flow in _get_flows_by_formula(s).items():
                total[formula] = total.get(formula, 0.0) + flow
        return _build_residual_vector(self.outlet, total)


class Splitter:
    """分割器：1つの入口を複数の出口に分割する。"""

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
    """反応器：化学量論に基づく単純な反応。"""

    def __init__(self, name: str, inlet, outlet, stoichiometry, key_component: int, conversion: float):
        self.name = name
        self.inlet = inlet
        self.outlet = outlet
        self.stoichiometry = np.asarray(stoichiometry, dtype=float)
        self.key_component = key_component
        self.conversion = conversion

    def residuals(self) -> np.ndarray:
        # outlet の成分順で化学量論係数と入口流量を構築
        outlet_formulas = [c.formula for c in self.outlet.components]
        inlet_flows = _get_flows_by_formula(self.inlet)

        # 反応進行度
        key_formula = self.outlet.components[self.key_component].formula
        key_inlet = inlet_flows.get(key_formula, 0.0)
        extent = self.conversion * key_inlet / abs(self.stoichiometry[self.key_component])

        # 期待出口流量
        expected = np.zeros(len(outlet_formulas))
        for i, f in enumerate(outlet_formulas):
            expected[i] = inlet_flows.get(f, 0.0) + self.stoichiometry[i] * extent

        return self.outlet.molar_flows - expected


class MultiReactor:
    """複数同時反応リアクター。

    Parameters
    ----------
    name : str
    inlet : Stream
    outlet : Stream
    reactions : list[dict[str, float]]
        各反応の化学量論係数 [{"CO": -2, "H2": -2, "CH3COOH": 1}, ...]
    key : str
        基準成分の示性式
    conversion : float
        基準成分の全体転化率 (0~1)
    selectivities : list[float]
        各反応の選択率（基準成分消費量ベース、合計=1）
    """

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

        # 各反応の進行度を計算
        expected = np.zeros(len(outlet_formulas))
        for i, f in enumerate(outlet_formulas):
            expected[i] = inlet_flows.get(f, 0.0)

        for rxn, sel in zip(self.reactions, self.selectivities):
            # この反応で消費される key 成分量
            key_consumed = sel * self.conversion * key_inlet
            # key の化学量論係数（負値）
            key_stoich = abs(rxn[self.key])
            extent = key_consumed / key_stoich
            for f, coeff in rxn.items():
                if f in outlet_formulas:
                    idx = outlet_formulas.index(f)
                    expected[idx] += coeff * extent

        return self.outlet.molar_flows - expected


def antoine_water_psat(T_celsius: float) -> float:
    """Antoine式で水の飽和蒸気圧を計算する。

    Returns
    -------
    float
        飽和蒸気圧 [Pa]
    """
    # Antoine constants for water (1-100°C range, NIST)
    # log10(P_mmHg) = A - B / (C + T)
    A, B, C = 8.07131, 1730.63, 233.426
    log_p_mmhg = A - B / (C + T_celsius)
    p_mmhg = 10.0 ** log_p_mmhg
    return p_mmhg * 133.322  # mmHg → Pa


class WaterSeparator:
    """Antoine式に基づく水分離器。

    40°C, 3MPaG 等の条件で、ガス中の水を飽和蒸気圧まで除去する。
    非水成分は全てガス側へ通過。
    水はガス中に飽和分だけ残り、残りは液水として除去。

    Parameters
    ----------
    inlet : Stream
    gas_outlet : Stream
        ガス出口（全成分、H2O は飽和量）
    water_outlet : Stream
        液水出口（H2O のみ）
    T_celsius : float
    P_pascal : float
        絶対圧 [Pa]
    """

    def __init__(self, name, inlet, gas_outlet, water_outlet, T_celsius, P_pascal):
        self.name = name
        self.inlet = inlet
        self.gas_outlet = gas_outlet
        self.water_outlet = water_outlet
        self.T_celsius = T_celsius
        self.P_pascal = P_pascal

    def residuals(self) -> np.ndarray:
        inlet_flows = _get_flows_by_formula(self.inlet)
        gas_formulas = [c.formula for c in self.gas_outlet.components]

        p_sat = antoine_water_psat(self.T_celsius)
        y_sat = p_sat / self.P_pascal  # 飽和モル分率

        res = []

        # 非水成分: ガス側へ全量通過
        non_water_total = 0.0
        for i, f in enumerate(gas_formulas):
            if f != "H2O":
                expected = inlet_flows.get(f, 0.0)
                res.append(self.gas_outlet.molar_flows[i] - expected)
                non_water_total += expected

        # H2O in gas: y_sat = n_H2O / (n_H2O + n_other)
        # → n_H2O = y_sat / (1 - y_sat) * n_other
        h2o_in_gas = y_sat / (1.0 - y_sat) * non_water_total
        h2o_idx = gas_formulas.index("H2O") if "H2O" in gas_formulas else None
        if h2o_idx is not None:
            res.append(self.gas_outlet.molar_flows[h2o_idx] - h2o_in_gas)

        # 液水: inlet_H2O - gas_H2O
        h2o_total = inlet_flows.get("H2O", 0.0)
        water_removed = h2o_total - h2o_in_gas
        # water_outlet は H2O のみ
        for i, c in enumerate(self.water_outlet.components):
            if c.formula == "H2O":
                res.append(self.water_outlet.molar_flows[i] - water_removed)
            else:
                res.append(self.water_outlet.molar_flows[i] - 0.0)

        return np.array(res)
