"""パターン1: Mixed組成指定 → Feed逆算 → Gibbsリアクター → 水凝縮

Mixed: total 204.72 NL/h, H2:9.7vol%, CO2:36.5vol%, CH4:23.4vol%, H2O:30.4vol%
各Feedは含む成分のみ指定し、流量は逆算で求める。
850°C / 1.04MPaG で Gibbs 平衡計算を行う。
平衡種: CO2, CH4, H2O, CO, H2
その後 25°C まで冷却し、気液平衡で凝縮水を抜き出す。
"""

from chemflow import Stream, eq, constrain, solve, reset, print_streams, set_component_order, export_csv, export_excel

reset()

# Mixed: 組成と流量を指定（固定）
Mixed = Stream(
    {"H2": 0.097, "CO2": 0.365, "CH4": 0.234, "H2O": 0.304},
    basis="volume_frac",
    total=204.72,
    name="Mixed",
    T=850, P="1.04MPaG", phase="Gas",
)

# 各Feedは含む成分のみ（流量は逆算）
CO2_feed = Stream(components=["CO2"], name="CO2_feed", T=25, P="1.04MPaG", phase="Gas")
RG_feed = Stream(components=["H2", "CH4"], name="RG_feed", T=25, P="1.04MPaG", phase="Gas")
H2O_feed = Stream(components=["H2O"], name="H2O_feed", T=25, P="1.04MPaG", phase="Gas")

# Mixed = CO2_feed + RG_feed + H2O_feed
eq(Mixed, CO2_feed + RG_feed + H2O_feed)

# Gibbs平衡
ReactOut = Mixed.gibbs_react(T=850, P="1.04MPaG", species=["CO2", "CH4", "H2O", "CO", "H2"])
ReactOut.name = "ReactOut"
ReactOut.T_celsius = 850
ReactOut.P_input = "1.04MPaG"
ReactOut.phase = "Gas"

# 25°C まで冷却 → 気液平衡で凝縮水を抜き出す
Gas, Condensate = ReactOut.separate_water(
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
print("パターン1: Gibbs平衡 + 水凝縮 (25°C)")
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
g = {c.formula: Gas.molar_flows[i] for i, c in enumerate(Gas.components)}
w = {c.formula: Condensate.molar_flows[i] for i, c in enumerate(Condensate.components)}
m = {c.formula: Mixed.molar_flows[i] for i, c in enumerate(Mixed.components)}
inlet_C = m.get("CO2", 0) + m.get("CH4", 0)
inlet_H = m.get("CH4", 0) * 4 + m.get("H2", 0) * 2 + m.get("H2O", 0) * 2
inlet_O = m.get("CO2", 0) * 2 + m.get("H2O", 0)
outlet_C = g.get("CO2", 0) + g.get("CH4", 0) + g.get("CO", 0) + w.get("CO2", 0) + w.get("CH4", 0) + w.get("CO", 0)
outlet_H = (g.get("CH4", 0) + w.get("CH4", 0)) * 4 + (g.get("H2O", 0) + w.get("H2O", 0)) * 2 + (g.get("H2", 0) + w.get("H2", 0)) * 2
outlet_O = (g.get("CO2", 0) + w.get("CO2", 0)) * 2 + (g.get("H2O", 0) + w.get("H2O", 0)) + (g.get("CO", 0) + w.get("CO", 0))
print(f"C: inlet={inlet_C:.4f}, outlet={outlet_C:.4f}")
print(f"H: inlet={inlet_H:.4f}, outlet={outlet_H:.4f}")
print(f"O: inlet={inlet_O:.4f}, outlet={outlet_O:.4f}")

print(f"\n--- Feed 逆算結果 ---")
print(f"CO2_feed: {CO2_feed.total_normal_volume_flow:.2f} NL/h")
print(f"RG_feed:  {RG_feed.total_normal_volume_flow:.2f} NL/h (H2={RG_feed.molar_flows[0]*22.414:.2f}, CH4={RG_feed.molar_flows[1]*22.414:.2f} NL/h)")
print(f"H2O_feed: {H2O_feed.total_normal_volume_flow:.2f} NL/h")
print(f"Condensate: {Condensate.total_molar_flow:.4f} mol/h")
print(f"DryGas:     {Gas.total_molar_flow:.4f} mol/h")
