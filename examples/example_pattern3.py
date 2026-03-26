"""パターン3: パターン2から派生した循環系 + 複数反応 + 水分離

パターン1の出口（全量）を入口として:
  3反応同時進行（CO全体転化率90%）:
    2CO + 2H2 → CH3COOH (選択率70%)
    2CO + 3H2 → CH3CHO + H2O (選択率20%)
    CO + 3H2 → CH4 + H2O (選択率10%)
  反応後に H2O 10mol/h を追加混合
  40°C, 3MPaG で Antoine式水分離（液水抜き出し）
  ガスは一部パージ、残り循環
"""

from chemflow import Stream, eq, constrain, solve, reset, print_streams, set_component_order

# ========================================
# パターン1: Gibbs平衡（前段）
# ========================================
reset()

A1 = Stream({"CO2": 5}, name="P1_CO2")
B1 = Stream({"CH4": 5}, name="P1_CH4")
C1 = Stream({"H2O": 5}, name="P1_H2O")
D1 = A1 + B1 + C1
E1 = D1.gibbs_react(T=850, P="2MPaG", species=["CO2", "CH4", "H2O", "CO", "H2"])

solve()
e1_comp = {c.formula: E1.mole_fractions[i] for i, c in enumerate(E1.components)}
e1_total = E1.total_molar_flow
print(f"P1 出口: total={e1_total:.4f} mol/h")
print(f"P1 組成: { {k: f'{v:.4f}' for k, v in e1_comp.items()} }")

# ========================================
# パターン3: 循環系 + 3反応 + 水分離
# ========================================
reset()

comps = ["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"]

# Feed = P1の出口（全量）
A3 = Stream(e1_comp, basis="mole_frac", total=e1_total, name="Feed")

# 循環ストリーム
B3 = Stream(components=comps, name="Recycle")
C3 = Stream(components=comps, name="Mixed")

eq(C3, A3 + B3)

# 3反応同時 (CO全体転化率90%)
D3 = C3.multi_react(
    reactions=[
        {"CO": -2, "H2": -2, "CH3COOH": 1},         # 選択率70%
        {"CO": -2, "H2": -3, "CH3CHO": 1, "H2O": 1}, # 選択率20%
        {"CO": -1, "H2": -3, "CH4": 1, "H2O": 1},    # 選択率10%
    ],
    key="CO",
    conversion=0.9,
    selectivities=[0.7, 0.2, 0.1],
)

# H2O 10mol/h を追加混合
H2O_feed = Stream({"H2O": 10}, name="H2O_feed")
D3_mixed = D3 + H2O_feed

# 水分離 (40°C, 3MPaG)
G3, Water_out = D3_mixed.separate_water(
    T=40, P="3MPaG",
    name_gas="Gas", name_water="WaterOut",
)

# ガスの分割: パージ + 循環
H3 = Stream(components=comps, name="Purge")
eq(G3, H3 + B3)

# 均一組成分割
constrain(lambda: (G3.mole_fractions - H3.mole_fractions)[:-1])
# Mixed total = 30 mol/h
constrain(lambda: C3.total_molar_flow - 30)

solve()

# 成分表示順序を設定
set_component_order(["H2", "CO", "CO2", "CH4", "H2O", "CH3CHO", "CH3COOH", "N2"])

print("\n" + "=" * 60)
print("パターン3: 循環系 + 3反応 + 水分離")
print("=" * 60)
print_streams()

print("\n--- 制約検証 ---")
print(f"C3 total mol:     {C3.total_molar_flow:.4f} (target: 30)")
print(f"Feed mass:         {A3.total_mass_flow:.4f}")
print(f"Purge mass:        {H3.total_mass_flow:.4f}")
print(f"WaterOut mol:      {Water_out.total_molar_flow:.4f}")
print(f"Recycle mol:       {B3.total_molar_flow:.4f}")
