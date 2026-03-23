"""パターン2: 循環系

パターン1の出口Eと同組成の入口A（流量未知）から、循環系を構築する。
反応: 2CO + 2H2 → CH3COOH (転化率90% on CO, 選択率100%)
制約: C total = 30 mol/h, mass(A) = mass(E_out)
均一組成分割。
"""

from chemflow import Stream, eq, constrain, solve, reset, print_streams

reset()

# パターン1の出口を仮定（Gibbs平衡後の代表的な組成）
# 実際にはパターン1の結果を使う
E_pattern1_composition = {"CO2": 0.15, "CH4": 0.05, "H2O": 0.10, "CO": 0.30, "H2": 0.40}

comps = ["CO2", "CH4", "H2O", "CO", "H2", "CH3COOH"]

A = Stream(E_pattern1_composition, basis="mole_frac", name="A_Feed")
B = Stream(components=comps, name="B_Recycle")
C = Stream(components=comps, name="C_Mixed")

eq(C, A + B)

D = C.react({"CO": -2, "H2": -2, "CH3COOH": 1}, key="CO", conversion=0.9)

E_out = Stream(components=comps, name="E_Purge")
eq(D, E_out + B)

# 均一組成分割 (n-1 個の独立制約)
constrain(lambda: (D.mole_fractions - E_out.mole_fractions)[:-1])
# C の合計モル流量 = 30
constrain(lambda: C.total_molar_flow - 30)
# A と E_out の重量流量が等しい
constrain(lambda: A.total_mass_flow - E_out.total_mass_flow)

solve()

print("=" * 60)
print("パターン2: 循環系計算結果")
print("=" * 60)
print_streams()

print("\n--- 制約検証 ---")
print(f"C total mol:  {C.total_molar_flow:.4f} (target: 30)")
print(f"A mass flow:  {A.total_mass_flow:.4f}")
print(f"E mass flow:  {E_out.total_mass_flow:.4f}")
print(f"A mole_frac:  {dict(zip([c.formula for c in A.components], A.mole_fractions))}")
