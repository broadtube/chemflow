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


def set_stream_order(order: list[str]) -> None:
    """出力時のストリーム表示順序を設定する。"""
    _get_flowsheet().set_stream_order(order)


def export_csv(path: str) -> None:
    """全ストリームの結果をCSVファイルに出力する。"""
    _get_flowsheet().export_csv(path)


def export_excel(filename: str, sheet: str, cell: str = "A1") -> None:
    """開いている Excel ブックのシートに結果を出力する。"""
    _get_flowsheet().export_excel(filename, sheet, cell)


def generate_mermaid() -> str:
    """Mermaid フロー図コードを生成する。"""
    return _get_flowsheet().generate_mermaid()


def export_mermaid(path: str, title: str | None = None, description: str | None = None) -> None:
    """Mermaid フロー図を HTML ファイルとして出力する。"""
    _get_flowsheet().export_mermaid(path, title=title, description=description)


def generate_json() -> dict:
    """Flowsheet を JSON dict として出力する。"""
    return _get_flowsheet().generate_json()


def export_json(path: str) -> None:
    """Flowsheet を JSON ファイルとして出力する。"""
    _get_flowsheet().export_json(path)


def export_reactflow(path: str, title: str | None = None, description: str | None = None) -> None:
    """ReactFlow によるインタラクティブフロー図を HTML として出力する。"""
    _get_flowsheet().export_reactflow(path, title=title, description=description)


def load_json(path: str) -> dict:
    """JSON ファイルから Flowsheet を復元・求解する。"""
    from chemflow.loader import load_json as _load
    return _load(path)
