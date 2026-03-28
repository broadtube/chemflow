"""パターン1: 3ストリーム混合 → Gibbsリアクター → 水凝縮

CO2 75NL/h + RG(H2:20NL/h, CH4:48NL/h) + H2O 62NL/h を混合し、
850°C / 1.04MPaG で Gibbs 平衡計算を行う。
平衡種: CO2, CH4, H2O, CO, H2
その後 25°C まで冷却し、気液平衡で凝縮水を抜き出す。
"""

from chemflow import Stream, solve, reset, print_streams, set_component_order, export_csv, export_excel

reset()

A = Stream({"CO2": 75}, basis="normal_volume", name="CO2_feed", T=25, P="1.04MPaG", phase="Gas")
B = Stream({"H2": 20, "CH4": 48}, basis="normal_volume", name="RG_feed", T=25, P="1.04MPaG", phase="Gas")
C = Stream({"H2O": 62}, basis="normal_volume", name="H2O_feed", T=25, P="1.04MPaG", phase="Gas")

D = A + B + C
D.name = "Mixed"
D.T_celsius = 850
D.P_input = "1.04MPaG"
D.phase = "Gas"

E = D.gibbs_react(T=850, P="1.04MPaG", species=["CO2", "CH4", "H2O", "CO", "H2"])
E.name = "ReactOut"
E.T_celsius = 850
E.P_input = "1.04MPaG"
E.phase = "Gas"

# 25°C まで冷却 → 気液平衡で凝縮水を抜き出す
Gas, Condensate = E.separate_water(
    T=25, P="1.04MPaG",
    name_gas="DryGas", name_water="Condensate",
)
Gas.T_celsius = 25
Gas.P_input = "1.04MPaG"
Gas.phase = "Gas"
Condensate.T_celsius = 25
Condensate.P_input = "1.04MPaG"
Condensate.phase = "Liquid"

solve()

set_component_order(["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"])

print("=" * 60)
print("パターン1: Gibbs平衡 + 水凝縮 (30°C)")
print("=" * 60)
print_streams()
export_csv("pattern1_result.csv")
print("\nCSV出力: pattern1_result.csv")

try:
    export_excel("output.xlsx", "Sheet1", "A1")
    print("Excel出力: output.xlsx / Sheet1 / A1")
except Exception as e:
    print(f"Excel出力スキップ: {e}")

# 元素保存の検証
print("\n--- 元素保存検証 ---")
a = {c.formula: A.molar_flows[i] for i, c in enumerate(A.components)}
b = {c.formula: B.molar_flows[i] for i, c in enumerate(B.components)}
c = {c.formula: C.molar_flows[i] for i, c in enumerate(C.components)}
inlet_C = a.get("CO2", 0) + b.get("CH4", 0)
inlet_H = b.get("CH4", 0) * 4 + b.get("H2", 0) * 2 + c.get("H2O", 0) * 2
inlet_O = a.get("CO2", 0) * 2 + c.get("H2O", 0)

g = {c.formula: Gas.molar_flows[i] for i, c in enumerate(Gas.components)}
w = {c.formula: Condensate.molar_flows[i] for i, c in enumerate(Condensate.components)}
outlet_C = g.get("CO2", 0) + g.get("CH4", 0) + g.get("CO", 0) + w.get("CO2", 0) + w.get("CH4", 0) + w.get("CO", 0)
outlet_H = (g.get("CH4", 0) + w.get("CH4", 0)) * 4 + (g.get("H2O", 0) + w.get("H2O", 0)) * 2 + (g.get("H2", 0) + w.get("H2", 0)) * 2
outlet_O = (g.get("CO2", 0) + w.get("CO2", 0)) * 2 + (g.get("H2O", 0) + w.get("H2O", 0)) + (g.get("CO", 0) + w.get("CO", 0))

print(f"C: inlet={inlet_C:.4f}, outlet={outlet_C:.4f}")
print(f"H: inlet={inlet_H:.4f}, outlet={outlet_H:.4f}")
print(f"O: inlet={inlet_O:.4f}, outlet={outlet_O:.4f}")

print(f"\n--- 凝縮水 ---")
print(f"Condensate total: {Condensate.total_molar_flow:.4f} mol/h")
print(f"DryGas total:     {Gas.total_molar_flow:.4f} mol/h")
