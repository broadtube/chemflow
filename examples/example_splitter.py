"""サンプル: 分割（Splitter）"""

from chemflow import Component, Stream, Splitter, Flowsheet

# 成分定義
A = Component("A", mw=30.0)
B = Component("B", mw=50.0)
comps = [A, B]

# ストリーム
feed = Stream("Feed", comps)
feed.set_molar_flows([10.0, 5.0])

out1 = Stream("Out1", comps)
out1.set_molar_flows([1.0, 1.0])  # 初期推定

out2 = Stream("Out2", comps)
out2.set_molar_flows([1.0, 1.0])  # 初期推定

# フローシート構築
fs = Flowsheet("Splitter Example")
fs.add_stream(feed)
fs.add_stream(out1)
fs.add_stream(out2)
fs.fix_stream(feed)
fs.add_unit(Splitter("SPL1", inlet=feed, outlets=[out1, out2], ratios=[0.7, 0.3]))

# 求解
result = fs.solve()
print(f"収束: {result.success}")
fs.print_streams()
