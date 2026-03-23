"""サンプル: 反応（Reactor）

A → B（転化率 80%）
"""

from chemflow import Component, Stream, Reactor, Flowsheet

# 成分定義
A = Component("A", mw=30.0)
B = Component("B", mw=50.0)
comps = [A, B]

# ストリーム
feed = Stream("Feed", comps)
feed.set_molar_flows([10.0, 0.0])

product = Stream("Product", comps)
product.set_molar_flows([5.0, 5.0])  # 初期推定

# フローシート構築
fs = Flowsheet("Reactor Example")
fs.add_stream(feed)
fs.add_stream(product)
fs.fix_stream(feed)

# A → B: stoich = [-1, +1], 基準成分A(index=0), 転化率80%
fs.add_unit(
    Reactor(
        "RX1",
        inlet=feed,
        outlet=product,
        stoichiometry=[-1.0, 1.0],
        key_component=0,
        conversion=0.8,
    )
)

# 求解
result = fs.solve()
print(f"収束: {result.success}")
fs.print_streams()
print(f"\nA転化率: {(feed.molar_flows[0] - product.molar_flows[0]) / feed.molar_flows[0] * 100:.1f}%")
