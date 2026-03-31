"""JSON → Flowsheet の復元。"""

from __future__ import annotations

import json

import numpy as np


def load_json(path: str) -> dict:
    """JSON ファイルから Flowsheet を復元し、solve して結果を返す。

    Parameters
    ----------
    path : str
        JSON ファイルパス

    Returns
    -------
    dict
        {"streams": {name: Stream}, "result": solve_result}
    """
    from chemflow.stream import Stream
    from chemflow.api import eq, constrain
    from chemflow.global_flowsheet import _get_flowsheet, reset, solve

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    reset()

    # 1. ストリームを復元（ユニットが内部生成するものは後で上書き）
    stream_map: dict[str, Stream] = {}
    stream_data_map: dict[str, dict] = {}

    # ユニットが内部生成するストリーム ID を収集
    unit_generated: set[str] = set()
    for u_data in data["units"]:
        if u_data["type"] == "Absorber":
            unit_generated.add(u_data.get("gas_outlet", ""))
            unit_generated.add(u_data.get("liquid_outlet", ""))
            unit_generated.add(u_data.get("water_inlet", ""))
        elif u_data["type"] == "WaterSeparator":
            unit_generated.add(u_data.get("gas_outlet", ""))
            unit_generated.add(u_data.get("water_outlet", ""))
        elif u_data["type"] in ("MultiReactor", "Reactor", "GibbsReactor"):
            unit_generated.add(u_data.get("target", ""))

    for s_data in data["streams"]:
        sid = s_data["id"]
        stream_data_map[sid] = s_data

        # ユニットが内部生成するストリームはスキップ（後でユニット復元時に作成）
        if sid in unit_generated:
            continue

        name = s_data.get("name") or sid
        T = s_data.get("T_celsius")
        P = s_data.get("P_input")
        phase = s_data.get("phase")
        fixed = s_data.get("fixed", False)
        comps = s_data.get("components", {})

        original_comps = s_data.get("original_components")
        has_cc = s_data.get("has_composition_constraints", False)

        if fixed:
            # 固定ストリーム: mol流量で復元
            if all(abs(v) < 1e-10 for v in comps.values()):
                first_comp = list(comps.keys())[0] if comps else "N2"
                stream = Stream({first_comp: 0}, name=name, T=T, P=P, phase=phase)
            else:
                flows = {k: v for k, v in comps.items() if abs(v) > 1e-12}
                if not flows:
                    flows = {list(comps.keys())[0]: 0}
                stream = Stream(flows, name=name, T=T, P=P, phase=phase)
        elif has_cc:
            # 組成制約付き変数ストリーム: mole_frac + total 未知で復元
            total_mol = s_data.get("total_mol", 1.0)
            if total_mol > 1e-10:
                mole_fracs = {}
                for k, v in comps.items():
                    mole_fracs[k] = v / total_mol
                stream = Stream(mole_fracs, basis="mole_frac", name=name, T=T, P=P, phase=phase)
            else:
                comp_list = list(comps.keys())
                stream = Stream(components=comp_list, name=name, T=T, P=P, phase=phase)
        else:
            # 通常の変数ストリーム
            comp_list = original_comps if original_comps else list(comps.keys())
            if comp_list:
                stream = Stream(components=comp_list, name=name, T=T, P=P, phase=phase)
            else:
                stream = Stream(name=name, T=T, P=P, phase=phase)

        stream_map[sid] = stream

    # 2. ユニットを復元
    for u_data in data["units"]:
        utype = u_data["type"]

        if utype == "Mixer":
            sources = [stream_map[s] for s in u_data.get("sources", [])]
            target = stream_map[u_data["target"]]
            # eq(target, sum of sources)
            if len(sources) == 1:
                expr = sources[0]
            else:
                expr = sources[0]
                for s in sources[1:]:
                    expr = expr + s
            eq(target, expr)

        elif utype in ("Splitter", "Splitter (eq)"):
            source_stream = stream_map[u_data["source"]]
            targets = [stream_map[t] for t in u_data.get("targets", [])]
            # eq(source, target1 + target2 + ...)
            expr = targets[0]
            for t in targets[1:]:
                expr = expr + t
            eq(source_stream, expr)

        elif utype == "MultiReactor":
            inlet = stream_map[u_data["source"]]
            outlet_id = u_data["target"]
            reactions = u_data["reactions"]
            key = u_data["key"]
            conversion = u_data["conversion"]
            selectivities = u_data["selectivities"]
            result_stream = inlet.multi_react(
                reactions=reactions, key=key,
                conversion=conversion, selectivities=selectivities,
            )
            sd = stream_data_map.get(outlet_id, {})
            result_stream.name = sd.get("name") or outlet_id
            if sd.get("T_celsius") is not None:
                result_stream.T_celsius = sd["T_celsius"]
            if sd.get("P_input"):
                result_stream.P_input = sd["P_input"]
            if sd.get("phase"):
                result_stream.phase = sd["phase"]
            stream_map[outlet_id] = result_stream

        elif utype == "Reactor":
            inlet = stream_map[u_data["source"]]
            outlet_id = u_data["target"]
            # Reactor の stoichiometry は JSON に直接保存されていないので
            # 簡易復元（conversion のみ）
            # TODO: stoichiometry を JSON に保存
            pass

        elif utype == "GibbsReactor":
            inlet = stream_map[u_data["source"]]
            outlet_id = u_data["target"]
            T_c = u_data.get("T_celsius", 850)
            P_pa = u_data.get("P_pascal", 101325)
            species = u_data.get("species", [])
            result_stream = inlet.gibbs_react(T=T_c, P=P_pa, species=species)
            sd = stream_data_map.get(outlet_id, {})
            result_stream.name = sd.get("name") or outlet_id
            if sd.get("T_celsius") is not None:
                result_stream.T_celsius = sd["T_celsius"]
            if sd.get("P_input"):
                result_stream.P_input = sd["P_input"]
            if sd.get("phase"):
                result_stream.phase = sd["phase"]
            stream_map[outlet_id] = result_stream

        elif utype == "Absorber":
            gas_inlet = stream_map[u_data["gas_inlet"]]
            water_inlet_id = u_data["water_inlet"]
            gas_outlet_id = u_data["gas_outlet"]
            liquid_outlet_id = u_data["liquid_outlet"]
            T_c = u_data.get("T_celsius", 25)
            stages = u_data.get("stages", 10)

            # 水入口の流量を JSON データから取得
            wi_data = stream_data_map.get(water_inlet_id, {})
            water_flow = wi_data.get("total_mol", 5.55)

            # 圧力を gas_outlet の JSON データから取得
            go_data = stream_data_map.get(gas_outlet_id, {})
            P_str = go_data.get("P_input") or str(u_data.get("P_pascal", 101325))

            g_out, l_out = gas_inlet.absorb(
                water_flow=water_flow, T=T_c, P=P_str,
                stages=stages,
                name_gas=gas_outlet_id, name_liquid=liquid_outlet_id,
                name_water=water_inlet_id,
            )
            # 属性コピー
            for sid, s_out in [(gas_outlet_id, g_out), (liquid_outlet_id, l_out), (water_inlet_id, None)]:
                sd = stream_data_map.get(sid, {})
                target = s_out
                if target is None:
                    # water_inlet は absorb() 内部で作成済み、flowsheet から探す
                    for fs_s in _get_flowsheet().streams:
                        if fs_s.name == sid:
                            target = fs_s
                            break
                if target:
                    if sd.get("T_celsius") is not None:
                        target.T_celsius = sd["T_celsius"]
                    if sd.get("P_input"):
                        target.P_input = sd["P_input"]
                    if sd.get("phase"):
                        target.phase = sd["phase"]
                    stream_map[sid] = target

        elif utype == "WaterSeparator":
            inlet = stream_map[u_data.get("source", "")]
            gas_outlet_id = u_data.get("gas_outlet", "")
            water_outlet_id = u_data.get("water_outlet", "")
            # WaterSeparator のパラメータ復元は Absorber と似ている
            # TODO: T, P の復元
            pass

    # 3. 制約条件を復元
    constraint_specs = data.get("constraint_specs", [])
    # eval 用の名前空間を構築（ストリーム名 + numpy）
    eval_ns = dict(stream_map)
    eval_ns["np"] = np
    eval_ns["__builtins__"] = {"abs": abs, "max": max, "min": min, "len": len}

    for spec in constraint_specs:
        code = spec.get("code", "")
        label = spec.get("label", "")
        if code:
            try:
                func = eval(code, eval_ns)
                constrain(func, label=label, code=code)
            except Exception as e:
                print(f"Warning: 制約 '{label}' の復元に失敗: {e}")

    # 4. 表示設定
    fs = _get_flowsheet()
    comp_order = data.get("component_order")
    if comp_order:
        fs.set_component_order(comp_order)
    stream_order = data.get("stream_order")
    if stream_order:
        fs.set_stream_order(stream_order)

    # 5. 求解
    result = solve()

    return {"streams": stream_map, "result": result}
