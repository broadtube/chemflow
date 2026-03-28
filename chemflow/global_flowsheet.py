"""グローバル Flowsheet 管理。

全操作が暗黙的にここに登録される。
"""

from __future__ import annotations

from chemflow.flowsheet import Flowsheet

_flowsheet: Flowsheet | None = None


def _get_flowsheet() -> Flowsheet:
    """現在のグローバル Flowsheet を取得。なければ作成。"""
    global _flowsheet
    if _flowsheet is None:
        _flowsheet = Flowsheet("Global")
    return _flowsheet


def reset() -> None:
    """グローバル Flowsheet をクリアし、新しい計算を開始する。"""
    global _flowsheet
    _flowsheet = None


def solve(**kwargs):
    """グローバル Flowsheet の連立方程式を求解する。"""
    return _get_flowsheet().solve(**kwargs)


def print_streams() -> None:
    """全ストリームの結果を一覧表示する。"""
    _get_flowsheet().print_streams()


def set_component_order(order: list[str]) -> None:
    """出力時の成分表示順序を設定する。"""
    _get_flowsheet().set_component_order(order)


def export_csv(path: str) -> None:
    """全ストリームの結果をCSVファイルに出力する。"""
    _get_flowsheet().export_csv(path)


def export_excel(filename: str, sheet: str, cell: str = "A1") -> None:
    """開いている Excel ブックのシートに結果を出力する。"""
    _get_flowsheet().export_excel(filename, sheet, cell)
