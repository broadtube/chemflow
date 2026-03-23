"""パターン1: 3ストリーム混合 → Gibbsリアクター

CO2 5mol/h + CH4 5mol/h + H2O 5mol/h を混合し、
850°C / 2MPaG で Gibbs 平衡計算を行う。
平衡種: CO2, CH4, H2O, CO, H2
"""

from chemflow import Stream, solve, reset, print_streams

reset()

A = Stream({"CO2": 5}, name="A_CO2")
B = Stream({"CH4": 5}, name="B_CH4")
C = Stream({"H2O": 5}, name="C_H2O")

D = A + B + C
E = D.gibbs_react(T=850, P="2MPaG", species=["CO2", "CH4", "H2O", "CO", "H2"])

solve()

print("=" * 60)
print("パターン1: Gibbs平衡計算結果")
print("=" * 60)
print_streams()

# 元素保存の検証
print("\n--- 元素保存検証 ---")
inlet_C = 5 + 5  # CO2(1C) + CH4(1C) = 10 mol C
inlet_H = 5 * 4 + 5 * 2  # CH4(4H) + H2O(2H) = 30 mol H
inlet_O = 5 * 2 + 5  # CO2(2O) + H2O(1O) = 15 mol O

# E の元素
e_flows = {c.formula: E.molar_flows[i] for i, c in enumerate(E.components)}
outlet_C = e_flows.get("CO2", 0) + e_flows.get("CH4", 0) + e_flows.get("CO", 0)
outlet_H = e_flows.get("CH4", 0) * 4 + e_flows.get("H2O", 0) * 2 + e_flows.get("H2", 0) * 2
outlet_O = e_flows.get("CO2", 0) * 2 + e_flows.get("H2O", 0) + e_flows.get("CO", 0)

print(f"C: inlet={inlet_C:.4f}, outlet={outlet_C:.4f}")
print(f"H: inlet={inlet_H:.4f}, outlet={outlet_H:.4f}")
print(f"O: inlet={inlet_O:.4f}, outlet={outlet_O:.4f}")
