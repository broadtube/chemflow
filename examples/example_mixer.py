"""サンプル: 混合（Mixer）"""

from chemflow import Component, Stream, Mixer, Flowsheet

# 成分定義
H2 = Component("H2", mw=2.016)
N2 = Component("N2", mw=28.014)
comps = [H2, N2]

# ストリーム
s1 = Stream("Feed1", comps)
s1.set_molar_flows([3.0, 0.0])

s2 = Stream("Feed2", comps)
s2.set_molar_flows([0.0, 1.0])

s3 = Stream("Mixed", comps)
s3.set_molar_flows([1.0, 1.0])  # 初期推定

# フローシート構築
fs = Flowsheet("Mixer Example")
fs.add_stream(s1)
fs.add_stream(s2)
fs.add_stream(s3)
fs.fix_stream(s1)
fs.fix_stream(s2)
fs.add_unit(Mixer("MIX1", inlets=[s1, s2], outlet=s3))

# 求解
result = fs.solve()
print(f"収束: {result.success}")
fs.print_streams()
