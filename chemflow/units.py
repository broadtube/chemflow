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


# Henry則定数 [Pa] at 40°C
# H_i: 気体iの水への溶解に関する Henry 定数
# x_i = P_i / H_i (x_i: 液中モル分率, P_i: 気相分圧)
# 値が大きい = 溶けにくい, 値が小さい = 溶けやすい
HENRY_CONSTANTS_40C: dict[str, float] = {
    "H2":      7.82e9,    # 非常に溶けにくい
    "N2":      9.69e9,    # 非常に溶けにくい
    "CO":      6.28e9,    # 非常に溶けにくい
    "CH4":     4.62e9,    # 溶けにくい
    "CO2":     2.45e8,    # やや溶ける
    "CH3CHO":  7.60e5,    # よく溶ける (アセトアルデヒド)
    "CH3COOH": 5.07e4,    # 極めてよく溶ける (酢酸)
}


class WaterSeparator:
    """Antoine式 + Henry則に基づく水分離器。

    条件（例: 40°C, 3MPaG）で気液分離を行う。
    - 水蒸気: Antoine式で飽和蒸気圧を計算し、ガス中に残る量を決定
    - 気体溶解: Henry則で各ガス成分の液水への溶解量を計算

    Parameters
    ----------
    inlet : Stream
    gas_outlet : Stream
        ガス出口（全成分）
    water_outlet : Stream
        液水出口（H2O + 溶解成分）
    T_celsius : float
    P_pascal : float
        絶対圧 [Pa]
    henry_constants : dict[str, float] | None
        Henry 定数の上書き [Pa]。None の場合は 40°C のデフォルト値を使用。
    """

    def __init__(self, name, inlet, gas_outlet, water_outlet, T_celsius, P_pascal,
                 henry_constants=None):
        self.name = name
        self.inlet = inlet
        self.gas_outlet = gas_outlet
        self.water_outlet = water_outlet
        self.T_celsius = T_celsius
        self.P_pascal = P_pascal
        self.henry = henry_constants or HENRY_CONSTANTS_40C

    def residuals(self) -> np.ndarray:
        inlet_flows = _get_flows_by_formula(self.inlet)
        gas_formulas = [c.formula for c in self.gas_outlet.components]
        water_formulas = [c.formula for c in self.water_outlet.components]

        p_sat = antoine_water_psat(self.T_celsius)
        y_sat = p_sat / self.P_pascal
        P = self.P_pascal

        res = []

        # --- 物質収支（各成分）: inlet_i = gas_i + water_i ---
        for j, f in enumerate(water_formulas):
            gas_idx = gas_formulas.index(f) if f in gas_formulas else None
            gas_j = self.gas_outlet.molar_flows[gas_idx] if gas_idx is not None else 0.0
            water_j = self.water_outlet.molar_flows[j]
            inlet_j = inlet_flows.get(f, 0.0)
            res.append(gas_j + water_j - inlet_j)

        # --- 水蒸気: Antoine式 ---
        # gas_H2O = y_sat / (1 - y_sat) * sum(non_H2O_gas)
        gas_h2o_idx = gas_formulas.index("H2O") if "H2O" in gas_formulas else None
        gas_h2o = self.gas_outlet.molar_flows[gas_h2o_idx] if gas_h2o_idx is not None else 0.0
        non_h2o_gas = sum(
            self.gas_outlet.molar_flows[i]
            for i, f in enumerate(gas_formulas) if f != "H2O"
        )
        res.append(gas_h2o - y_sat / (1.0 - y_sat) * non_h2o_gas)

        # --- Henry則: 溶解平衡 ---
        # water_i / n_water_liq = (gas_i / total_gas) * P / H_i
        # → water_i * total_gas * H_i = gas_i * P * n_water_liq
        # （除算を避けて乗算形式で残差を構成）
        water_h2o_idx = water_formulas.index("H2O") if "H2O" in water_formulas else None
        n_water_liq = self.water_outlet.molar_flows[water_h2o_idx] if water_h2o_idx is not None else 0.0
        total_gas = sum(self.gas_outlet.molar_flows[i] for i, _ in enumerate(gas_formulas))

        for j, f in enumerate(water_formulas):
            if f == "H2O":
                continue
            if f not in self.henry:
                # Henry定数なし: 液相への溶解なし
                res.append(self.water_outlet.molar_flows[j])
                continue
            H_i = self.henry[f]
            gas_idx = gas_formulas.index(f) if f in gas_formulas else None
            gas_i = self.gas_outlet.molar_flows[gas_idx] if gas_idx is not None else 0.0
            water_i = self.water_outlet.molar_flows[j]
            # 乗算形式: water_i * total_gas * H_i - gas_i * P * n_water_liq = 0
            # スケーリング: H_i が大きいので正規化
            scale = max(H_i, 1.0)
            res.append((water_i * total_gas * H_i - gas_i * P * n_water_liq) / scale)

        return np.array(res)
