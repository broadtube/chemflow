"""ユーザー向けトップレベル API: eq(), constrain()"""

from __future__ import annotations

from typing import Callable

import numpy as np

from chemflow.expression import StreamExpression
from chemflow.global_flowsheet import _get_flowsheet


def eq(target, expression) -> None:
    """既存 Stream に Expression の残差式を紐付ける。

    Usage:
        eq(C, A + B)      # C = A + B (Mixer)
        eq(D, E_out + B)  # D = E_out + B (Mixer)
    """
    if isinstance(expression, StreamExpression):
        expression.materialize(target=target)
    else:
        # expression が Stream の場合: target = expression → 等式制約
        def residual():
            return target.molar_flows - expression.molar_flows
        _get_flowsheet().add_spec(residual)


def constrain(residual_func: Callable, label: str | None = None, code: str | None = None) -> None:
    """任意の制約条件を lambda で登録する。

    Parameters
    ----------
    residual_func : Callable
        残差（= 0 になるべき値）を返す関数
    label : str | None
        フロー図に表示するラベル
    code : str | None
        制約の Python 式文字列（JSON 保存・復元用）

    Usage:
        constrain(lambda: C.total_molar_flow - 30,
                  label="Mixed = 30 mol/h",
                  code="lambda: Mixed.total_molar_flow - 30")
    """
    fs = _get_flowsheet()
    fs.add_spec(lambda: np.atleast_1d(residual_func()))
    if not hasattr(fs, "_constraint_labels"):
        fs._constraint_labels = []
    if not hasattr(fs, "_constraint_codes"):
        fs._constraint_codes = []
    fs._constraint_labels.append(label or "")
    fs._constraint_codes.append(code or "")
