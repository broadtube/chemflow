"""パターン3: 循環系 + 複数反応 + 多段吸収

入口: vol%指定、total未知（逆算）
Mixed: total 500 NL/h 固定
  3反応同時進行（CO全体転化率12%）:
    2CO + 2H2 → CH3COOH (選択率70%)
    2CO + 3H2 → CH3CHO + H2O (選択率20%)
    CO + 3H2 → CH4 + H2O (選択率10%)
  40°C, 3MPaG で 10段吸収塔（水30mol/h）
  パージ率 5%（Purge = Gas * 0.05）
"""

from chemflow import Stream, eq, constrain, solve, reset, print_streams, set_component_order, set_stream_order, export_csv, export_excel

reset()

comps = ["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"]

# Feed: vol%指定、total未知
A3 = Stream(
    {"H2": 0.476, "CO": 0.343, "CO2": 0.156, "CH4": 0.023, "H2O": 0.002},
    basis="volume_frac",
    name="Feed",
    T=25, P="0.1MPaG", phase="Gas",
)

# 循環ストリーム
B3 = Stream(components=comps, name="Recycle", T=40, P="3MPaG", phase="Gas")
C3 = Stream(components=comps, name="Mixed", T=200, P="3MPaG", phase="Gas")

eq(C3, A3 + B3)

# 3反応同時 (CO全体転化率12%)
D3 = C3.multi_react(
    reactions=[
        {"CO": -2, "H2": -2, "CH3COOH": 1},          # 選択率70%
        {"CO": -2, "H2": -3, "CH3CHO": 1, "H2O": 1}, # 選択率20%
        {"CO": -1, "H2": -3, "CH4": 1, "H2O": 1},    # 選択率10%
    ],
    key="CO",
    conversion=0.12,
    selectivities=[0.7, 0.2, 0.1],
)
D3.name = "ReactOut"
D3.T_celsius = 280
D3.P_input = "3MPaG"
D3.phase = "Gas"

# 多段吸収塔 (40°C, 3MPaG, 10段, 水100g/h)
G3, Water_out = D3.absorb(
    water_flow=100 / 18.015,  # 100 g/h → mol/h
    T=40, P="3MPaG",
    stages=10,
    name_gas="Gas", name_liquid="WaterOut", name_water="H2O_abs",
)
G3.T_celsius = 40
G3.P_input = "3MPaG"
G3.phase = "Gas"
Water_out.T_celsius = 40
Water_out.P_input = "3MPaG"
Water_out.phase = "Liquid"

# ガスの分割: パージ + 循環
H3 = Stream(components=comps, name="Purge", T=40, P="3MPaG", phase="Gas")
eq(G3, H3 + B3)

# 均一組成分割
constrain(lambda: (G3.mole_fractions - H3.mole_fractions)[:-1])
# Mixed total = 500 NL/h
constrain(lambda: C3.total_normal_volume_flow - 500)
# パージ率 5%: Purge.total_molar_flow = Gas.total_molar_flow * 0.05
constrain(lambda: H3.total_molar_flow - G3.total_molar_flow * 0.05)

solve()

# 表示設定
set_component_order(["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"])

print("=" * 60)
print("パターン3: 循環系 + 3反応 + 多段吸収")
print("=" * 60)
print_streams()
export_csv("pattern3_result.csv")
print("\nCSV出力: pattern3_result.csv")

# Excel出力（Excelでoutput.xlsxを開いている場合）
try:
    export_excel("output.xlsx", "Sheet1", "A1")
    print("Excel出力: output.xlsx / Sheet1 / A1")
except Exception as e:
    print(f"Excel出力スキップ: {e}")

print("\n--- 制約検証 ---")
print(f"Mixed total NL/h: {C3.total_normal_volume_flow:.4f} (target: 500)")
print(f"Feed total NL/h:  {A3.total_normal_volume_flow:.4f} (逆算)")
print(f"Purge/Gas ratio:  {H3.total_molar_flow / G3.total_molar_flow:.4f} (target: 0.05)")
print(f"WaterOut mol:     {Water_out.total_molar_flow:.4f}")
print(f"Recycle mol:      {B3.total_molar_flow:.4f}")
