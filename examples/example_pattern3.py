"""パターン3: 循環系 + 複数反応 + 多段吸収

入口: Syngas(vol%指定, total逆算) + H2 + N2 + Rx_Water(全て0)
Mixed: total 500 NL/h 固定
  3反応同時進行（CO全体転化率12%）:
    2CO + 2H2 → CH3COOH (選択率70%)
    2CO + 3H2 → CH3CHO + H2O (選択率20%)
    CO + 3H2 → CH4 + H2O (選択率10%)
  25°C, 3MPaG で 10段吸収塔（水100g/h）
  パージ率 5%（Purge = Gas * 0.05）
"""

from chemflow import (
    Stream, eq, constrain, solve, reset, print_streams,
    set_component_order, set_stream_order, export_csv, export_excel, export_mermaid,
)

reset()

comps = ["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"]

# Feeds
Syngas_feed = Stream(
    {"H2": 0.476, "CO": 0.343, "CO2": 0.156, "CH4": 0.023, "H2O": 0.002},
    basis="volume_frac",
    name="Syngas_feed",
    T=25, P="0.1MPaG", phase="Gas",
)
H2_feed = Stream({"H2": 0}, name="H2_feed", T=25, P="3MPaG", phase="Gas")
N2_feed = Stream({"N2": 0}, name="N2_feed", T=25, P="3MPaG", phase="Gas")
Rx_Water_Feed = Stream({"H2O": 0}, name="Rx_Water_Feed", T=25, phase="Liquid")

# 循環ストリーム
Recycle = Stream(components=comps, name="Recycle", T=25, P="3MPaG", phase="Gas")
Mixed = Stream(components=comps, name="Mixed", T=200, P="3MPaG", phase="Gas")

eq(Mixed, Syngas_feed + H2_feed + N2_feed + Rx_Water_Feed + Recycle)

# 3反応同時 (CO全体転化率12%)
ReactOut = Mixed.multi_react(
    reactions=[
        {"CO": -2, "H2": -2, "CH3COOH": 1},          # 選択率70%
        {"CO": -2, "H2": -3, "CH3CHO": 1, "H2O": 1}, # 選択率20%
        {"CO": -1, "H2": -3, "CH4": 1, "H2O": 1},    # 選択率10%
    ],
    key="CO",
    conversion=0.12,
    selectivities=[0.7, 0.2, 0.1],
)
ReactOut.name = "ReactOut"
ReactOut.T_celsius = 280
ReactOut.P_input = "3MPaG"
ReactOut.phase = "Gas"

# 多段吸収塔 (25°C, 3MPaG, 10段, 水100g/h)
Gas, WaterOut = ReactOut.absorb(
    water_flow=100 / 18.015,  # 100 g/h → mol/h
    T=25, P="3MPaG",
    stages=10,
    name_gas="Gas", name_liquid="WaterOut", name_water="H2O_abs",
)
Gas.T_celsius = 25
Gas.P_input = "3MPaG"
Gas.phase = "Gas"
WaterOut.T_celsius = 25
WaterOut.P_input = "3MPaG"
WaterOut.phase = "Liquid"

# ガスの分割: パージ + 循環
Purge = Stream(components=comps, name="Purge", T=25, P="3MPaG", phase="Gas")
eq(Gas, Purge + Recycle)

# 均一組成分割
constrain(lambda: (Gas.mole_fractions - Purge.mole_fractions)[:-1])
# Mixed total = 500 NL/h
constrain(lambda: Mixed.total_normal_volume_flow - 500)
# パージ率 5%
constrain(lambda: Purge.total_molar_flow - Gas.total_molar_flow * 0.05)

solve()

# 表示設定
set_component_order(["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"])
set_stream_order([
    "Syngas_feed", "H2_feed", "N2_feed", "Rx_Water_Feed", "Recycle",
    "Mixed", "ReactOut", "H2O_abs", "WaterOut", "Gas", "Purge",
])

print("=" * 60)
print("パターン3: 循環系 + 3反応 + 多段吸収")
print("=" * 60)
print_streams()
export_csv("pattern3_result.csv")
export_mermaid("pattern3_flow.html")
print("\nCSV出力: pattern3_result.csv")
print("フロー図: pattern3_flow.html")

# Excel出力
try:
    export_excel("output.xlsx", "Sheet1", "A1")
    print("Excel出力: output.xlsx / Sheet1 / A1")
except Exception as e:
    print(f"Excel出力スキップ: {e}")

# 制約検証
print("\n--- 制約検証 ---")
print(f"Mixed total NL/h: {Mixed.total_normal_volume_flow:.4f} (target: 500)")
print(f"Syngas_feed NL/h: {Syngas_feed.total_normal_volume_flow:.4f} (逆算)")
print(f"Purge/Gas ratio:  {Purge.total_molar_flow / Gas.total_molar_flow:.4f} (target: 0.05)")

# H2O選択率の検証
mixed_flows = {c.formula: Mixed.molar_flows[i] for i, c in enumerate(Mixed.components)}
react_flows = {c.formula: ReactOut.molar_flows[i] for i, c in enumerate(ReactOut.components)}
co_consumed = mixed_flows.get("CO", 0) - react_flows.get("CO", 0)
h2o_produced = react_flows.get("H2O", 0) - mixed_flows.get("H2O", 0)
h2o_selectivity = h2o_produced / co_consumed * 100 if co_consumed > 0 else 0
print(f"\n--- 反応検証 ---")
print(f"CO consumed:      {co_consumed:.4f} mol/h")
print(f"H2O produced:     {h2o_produced:.4f} mol/h")
print(f"H2O selectivity:  {h2o_selectivity:.1f}% (on CO consumed basis)")
