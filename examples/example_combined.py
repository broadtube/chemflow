"""パターン1 + パターン2 連結テスト

パターン1: CO2 + CH4 + H2O → Gibbs平衡 (850°C, 2MPaG)
パターン2: パターン1の出口（全量）を入口として循環系に投入
  反応: 2CO + 2H2 → CH3COOH (転化率90% on CO)
  制約: C total = 30 mol/h, 均一組成分割
"""

from chemflow import Stream, eq, constrain, solve, reset, print_streams

# ========================================
# パターン1: Gibbs平衡
# ========================================
reset()

A1 = Stream({"CO2": 5}, name="P1_CO2")
B1 = Stream({"CH4": 5}, name="P1_CH4")
C1 = Stream({"H2O": 5}, name="P1_H2O")
D1 = A1 + B1 + C1
E1 = D1.gibbs_react(T=850, P="2MPaG", species=["CO2", "CH4", "H2O", "CO", "H2"])

solve()

print("=" * 60)
print("パターン1: Gibbs平衡計算結果")
print("=" * 60)
print_streams()

# E1 の組成と流量を取得
e1_comp = {c.formula: E1.mole_fractions[i] for i, c in enumerate(E1.components)}
e1_total_mol = E1.total_molar_flow
print(f"\nE1 組成: { {k: f'{v:.4f}' for k, v in e1_comp.items()} }")
print(f"E1 total mol:  {e1_total_mol:.4f}")
print(f"E1 total mass: {E1.total_mass_flow:.4f}")

# ========================================
# パターン2: 循環系
# ========================================
reset()

comps = ["CO2", "CH4", "H2O", "CO", "H2", "CH3COOH"]

# パターン1の出口を全量投入（組成・流量とも確定）
A2 = Stream(e1_comp, basis="mole_frac", total=e1_total_mol, name="P2_Feed")
B2 = Stream(components=comps, name="P2_Recycle")
C2 = Stream(components=comps, name="P2_Mixed")

eq(C2, A2 + B2)

D2 = C2.react({"CO": -2, "H2": -2, "CH3COOH": 1}, key="CO", conversion=0.9)

E2 = Stream(components=comps, name="P2_Purge")
eq(D2, E2 + B2)

# 均一組成分割 (n-1 個の独立制約)
constrain(lambda: (D2.mole_fractions - E2.mole_fractions)[:-1])
# C の合計モル流量 = 30
constrain(lambda: C2.total_molar_flow - 30)

solve()

print("\n" + "=" * 60)
print("パターン2: 循環系計算結果")
print("=" * 60)
print_streams()

print("\n--- 制約検証 ---")
print(f"C2 total mol:  {C2.total_molar_flow:.4f} (target: 30)")
print(f"A2 mass flow:  {A2.total_mass_flow:.4f}")
print(f"E2 mass flow:  {E2.total_mass_flow:.4f}")
print(f"B2 total mol:  {B2.total_molar_flow:.4f}")
print(f"D2/E2 組成一致: {all(abs(D2.mole_fractions - E2.mole_fractions) < 1e-6)}")
