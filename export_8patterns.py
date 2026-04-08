"""8通りのPattern3バリエーションをExcelの各シートに出力

順序: 流量 → Purge → 転化率/選択率
流量: 37.5NL/h, 500NL/h
Purge: 5%, 10%
転化率/選択率: 12%/[0.7, 0.2, 0.1], 20%/[0.8, 0.13, 0.07]

実行:
  python export_8patterns.py
"""

import numpy as np
from chemflow import (
    Stream, eq, constrain, solve, reset,
    set_component_order, set_stream_order, export_excel,
)


def run_pattern(purge_rate, selectivities, flow_rate, conversion):
    """1つのパターンを計算"""
    reset()

    # N2を除外して計算（表示時にset_component_orderで追加）
    comps = ['H2', 'CO', 'CO2', 'CH4', 'H2O', 'CH3CHO', 'CH3COOH']

    # Feeds
    Syngas_feed = Stream(
        {'H2': 0.476, 'CO': 0.343, 'CO2': 0.156, 'CH4': 0.023, 'H2O': 0.002},
        basis='volume_frac',
        name='Syngas_feed',
        T=25, P='0.1MPaG', phase='Gas',
    )
    H2_feed = Stream({'H2': 0}, name='H2_feed', T=25, P='5MPaG', phase='Gas')
    N2_feed = Stream({'N2': 0}, name='N2_feed', T=25, P='5MPaG', phase='Gas')  # 表示用（Mixerには含めない）
    Rx_Water_Feed = Stream({'H2O': 0}, name='Rx_Water_Feed', T=25, phase='Liquid')

    # 循環ストリーム
    Recycle = Stream(components=comps, name='Recycle', T=25, P='5MPaG', phase='Gas')
    Mixed = Stream(components=comps, name='Mixed', T=25, P='5MPaG', phase='Gas')

    eq(Mixed, Syngas_feed + H2_feed + Rx_Water_Feed + Recycle)

    # 3反応同時
    ReactOut = Mixed.multi_react(
        reactions=[
            {'CO': -2, 'H2': -2, 'CH3COOH': 1},
            {'CO': -2, 'H2': -3, 'CH3CHO': 1, 'H2O': 1},
            {'CO': -1, 'H2': -3, 'CH4': 1, 'H2O': 1},
        ],
        key='CO',
        conversion=conversion,
        selectivities=selectivities,
    )
    ReactOut.name = 'ReactOut'
    ReactOut.T_celsius = 250
    ReactOut.P_input = '5MPaG'
    ReactOut.phase = 'Gas'

    # 多段吸収塔 (25°C, 5MPaG, 10段, 水100g/h)
    Gas, WaterOut = ReactOut.absorb(
        water_flow=100,
        T=25, P='5MPaG',
        stages=10,
        water_T=25, water_P='5MPaG', water_phase='Liquid',
        name_gas='Gas', name_liquid='WaterOut', name_water='H2O_abs',
    )
    Gas.T_celsius = 25
    Gas.P_input = '5MPaG'
    Gas.phase = 'Gas'
    WaterOut.T_celsius = 25
    WaterOut.P_input = '5MPaG'
    WaterOut.phase = 'Liquid'

    # ガスの分割: パージ + 循環
    Purge = Stream(components=comps, name='Purge', T=25, P='5MPaG', phase='Gas')
    eq(Gas, Purge + Recycle)

    # 制約
    constrain(lambda: (Gas.mole_fractions - Purge.mole_fractions)[:-1])
    constrain(lambda: Mixed.total_normal_volume_flow - flow_rate)
    constrain(lambda: Purge.total_molar_flow - Gas.total_molar_flow * purge_rate)

    # 求解（小流量対応のためbounds使用）
    solve(bounds=(0, np.inf))

    # 表示順序設定
    set_component_order(['H2', 'CO', 'CO2', 'CH4', 'H2O', 'CH3CHO', 'CH3COOH', 'N2'])
    set_stream_order([
        'Syngas_feed', 'H2_feed', 'N2_feed', 'Rx_Water_Feed', 'Recycle',
        'Mixed', 'ReactOut', 'H2O_abs', 'WaterOut', 'Gas', 'Purge',
    ])

    return {
        'Mixed_flow': Mixed.total_normal_volume_flow,
        'Purge_rate': Purge.total_molar_flow / Gas.total_molar_flow,
    }


# 8通りの組み合わせ (順序: 流量 → Purge → 転化率/選択率)
patterns = [
    # 1-4: 37.5 NL/h
    {'flow': 37.5, 'purge': 0.05, 'conversion': 0.12, 'selectivities': [0.7, 0.2, 0.1]},
    {'flow': 37.5, 'purge': 0.05, 'conversion': 0.20, 'selectivities': [0.8, 0.13, 0.07]},
    {'flow': 37.5, 'purge': 0.10, 'conversion': 0.12, 'selectivities': [0.7, 0.2, 0.1]},
    {'flow': 37.5, 'purge': 0.10, 'conversion': 0.20, 'selectivities': [0.8, 0.13, 0.07]},
    # 5-8: 500 NL/h
    {'flow': 500, 'purge': 0.05, 'conversion': 0.12, 'selectivities': [0.7, 0.2, 0.1]},
    {'flow': 500, 'purge': 0.05, 'conversion': 0.20, 'selectivities': [0.8, 0.13, 0.07]},
    {'flow': 500, 'purge': 0.10, 'conversion': 0.12, 'selectivities': [0.7, 0.2, 0.1]},
    {'flow': 500, 'purge': 0.10, 'conversion': 0.20, 'selectivities': [0.8, 0.13, 0.07]},
]

output_file = 'output.xlsx'

print('=' * 70)
print('Pattern3: 8通りのバリエーションを output.xlsx に出力')
print('=' * 70)
print('注意: output.xlsx を開いた状態で実行してください')

for i, p in enumerate(patterns, 1):
    sheet_name = f'Pattern{i}'
    print(f'\n--- {sheet_name}: Flow={p["flow"]}NL/h, Purge={p["purge"]*100:.0f}%, Conv={p["conversion"]*100:.0f}%, Sel={p["selectivities"]} ---')

    result = run_pattern(p['purge'], p['selectivities'], p['flow'], p['conversion'])

    print(f'  Mixed flow: {result["Mixed_flow"]:.4f} NL/h')
    print(f'  Purge rate: {result["Purge_rate"]*100:.4f}%')

    try:
        export_excel(output_file, sheet_name, 'A15')
        print(f'  -> {sheet_name} に出力完了')
    except Exception as e:
        print(f'  Excel出力エラー: {e}')

print('\n' + '=' * 70)
print(f'完了: {output_file} の Pattern1〜Pattern8 シート')
print('=' * 70)
