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


def constrain(residual_func: Callable) -> None:
    """任意の制約条件を lambda で登録する。

    Usage:
        constrain(lambda: C.total_molar_flow - 30)
        constrain(lambda: A.total_mass_flow - E.total_mass_flow)
        constrain(lambda: D.mole_fractions - E.mole_fractions)
    """
    _get_flowsheet().add_spec(lambda: np.atleast_1d(residual_func()))
