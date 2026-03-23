"""Expression レイヤー: ストリーム演算の遅延表現。

A + B → MixExpression (遅延)
A * 0.4 → ScaleExpression (遅延)

materialize() 呼び出しで Stream 生成 + Flowsheet 登録。
eq() 経由では既存 Stream に紐付け。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chemflow.stream import Stream


class StreamExpression:
    """ストリーム演算の遅延評価基底クラス。"""

    def __init__(self):
        self._materialized: Stream | None = None

    def materialize(self, target: Stream | None = None) -> Stream:
        """Stream を生成し Flowsheet に登録する。サブクラスで実装。"""
        raise NotImplementedError

    def _ensure_materialized(self) -> Stream:
        if self._materialized is None:
            self._materialized = self.materialize()
        return self._materialized

    # --- Stream プロキシ ---

    def react(self, stoichiometry, key, conversion):
        return self._ensure_materialized().react(stoichiometry, key, conversion)

    def gibbs_react(self, T, P, species):
        return self._ensure_materialized().gibbs_react(T, P, species)

    def __add__(self, other):
        stream = self._ensure_materialized()
        return stream.__add__(other)

    def __radd__(self, other):
        stream = self._ensure_materialized()
        return stream.__radd__(other)

    def __mul__(self, ratio):
        stream = self._ensure_materialized()
        return stream.__mul__(ratio)

    def __rmul__(self, ratio):
        stream = self._ensure_materialized()
        return stream.__rmul__(ratio)

    def __getattr__(self, name):
        # プロパティ/メソッドへのアクセス時に自動 materialize
        if name.startswith("_"):
            raise AttributeError(name)
        stream = self._ensure_materialized()
        return getattr(stream, name)


class MixExpression(StreamExpression):
    """混合演算の遅延表現: A + B + C"""

    def __init__(self, operands: list):
        super().__init__()
        self._operands = operands

    def materialize(self, target: Stream | None = None) -> Stream:
        if self._materialized is not None and target is None:
            return self._materialized

        from chemflow.stream import Stream
        from chemflow.global_flowsheet import _get_flowsheet

        # operands 内の Expression を先に materialize
        resolved = []
        for op in self._operands:
            if isinstance(op, StreamExpression):
                resolved.append(op._ensure_materialized())
            else:
                resolved.append(op)

        # 全成分の和集合
        all_formulas = []
        seen = set()
        for s in resolved:
            for c in s.components:
                if c.formula not in seen:
                    all_formulas.append(c.formula)
                    seen.add(c.formula)

        # 出口 Stream
        if target is not None:
            outlet = target
            # target に不足成分を追加
            for f in all_formulas:
                if f not in [c.formula for c in outlet.components]:
                    outlet._add_component(f)
        else:
            outlet = Stream(components=all_formulas)

        # inlet に不足成分を追加（残差計算のため次元を揃える）
        for s in resolved:
            for f in all_formulas:
                if f not in [c.formula for c in s.components]:
                    s._add_component(f)

        # Mixer 残差式を登録
        from chemflow.units import Mixer
        mixer = Mixer(f"MIX_auto", inlets=resolved, outlet=outlet)
        _get_flowsheet().add_unit(mixer)

        if target is None:
            self._materialized = outlet
        return outlet

    def __add__(self, other):
        """連続加算: (A + B) + C → MixExpression([A, B, C])"""
        if isinstance(other, MixExpression):
            return MixExpression(self._operands + other._operands)
        return MixExpression(self._operands + [other])

    def __radd__(self, other):
        if isinstance(other, MixExpression):
            return MixExpression(other._operands + self._operands)
        return MixExpression([other] + self._operands)


class ScaleExpression(StreamExpression):
    """分割演算の遅延表現: A * 0.4"""

    def __init__(self, stream, ratio: float):
        super().__init__()
        self._stream = stream
        self._ratio = ratio

    def materialize(self, target: Stream | None = None) -> Stream:
        if self._materialized is not None and target is None:
            return self._materialized

        from chemflow.stream import Stream
        from chemflow.global_flowsheet import _get_flowsheet

        # 入口を解決
        inlet = self._stream
        if isinstance(inlet, StreamExpression):
            inlet = inlet._ensure_materialized()

        # 出口 Stream
        if target is not None:
            outlet = target
        else:
            formulas = [c.formula for c in inlet.components]
            outlet = Stream(components=formulas)

        # Split 残差式: outlet - inlet * ratio = 0
        from chemflow.units import Splitter
        splitter = Splitter(
            f"SPL_auto",
            inlet=inlet,
            outlets=[outlet],
            ratios=[self._ratio],
        )
        _get_flowsheet().add_unit(splitter)

        if target is None:
            self._materialized = outlet
        return outlet
