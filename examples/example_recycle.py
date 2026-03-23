"""サンプル: リサイクルあり

フロー:
  Fresh Feed (A=10) → Mixer → Reactor (A→B, 50%) → Splitter → Product (70%)
                        ↑                                    → Recycle (30%) ↩

循環系を含む定常状態計算の例。
"""

from chemflow import Component, Stream, Mixer, Reactor, Splitter, Flowsheet

# 成分定義
A = Component("A", mw=30.0)
B = Component("B", mw=50.0)
comps = [A, B]

# ストリーム
fresh_feed = Stream("FreshFeed", comps)
fresh_feed.set_molar_flows([10.0, 0.0])

recycle = Stream("Recycle", comps)
recycle.set_molar_flows([1.0, 0.5])  # 初期推定

mixed = Stream("Mixed", comps)
mixed.set_molar_flows([10.0, 1.0])  # 初期推定

reactor_out = Stream("ReactorOut", comps)
reactor_out.set_molar_flows([5.0, 5.0])  # 初期推定

product = Stream("Product", comps)
product.set_molar_flows([3.0, 3.0])  # 初期推定

# フローシート構築
fs = Flowsheet("Recycle Example")
for s in [fresh_feed, recycle, mixed, reactor_out, product]:
    fs.add_stream(s)
fs.fix_stream(fresh_feed)

# Mixer: FreshFeed + Recycle → Mixed
fs.add_unit(Mixer("MIX1", inlets=[fresh_feed, recycle], outlet=mixed))

# Reactor: Mixed → ReactorOut (A→B, 転化率50%)
fs.add_unit(
    Reactor(
        "RX1",
        inlet=mixed,
        outlet=reactor_out,
        stoichiometry=[-1.0, 1.0],
        key_component=0,
        conversion=0.5,
    )
)

# Splitter: ReactorOut → Product(70%) + Recycle(30%)
fs.add_unit(
    Splitter("SPL1", inlet=reactor_out, outlets=[product, recycle], ratios=[0.7, 0.3])
)

# 求解
result = fs.solve()
print(f"収束: {result.success}")
fs.print_streams()

# 検証: 全体の物質収支
print(f"\n--- 全体物質収支 ---")
print(f"Fresh Feed A: {fresh_feed.molar_flows[0]:.4f} mol/s")
print(f"Product A:    {product.molar_flows[0]:.4f} mol/s")
print(f"Product B:    {product.molar_flows[1]:.4f} mol/s")
print(f"A消費量:      {fresh_feed.molar_flows[0] - product.molar_flows[0]:.4f} mol/s")
print(f"B生成量:      {product.molar_flows[1]:.4f} mol/s")
