"""chemflow エラークラス体系"""


class ChemflowError(Exception):
    """chemflow の基底例外クラス。"""
    pass


class FormulaError(ChemflowError):
    """不正な示性式。"""
    pass


class BasisError(ChemflowError):
    """不正な basis 指定。"""
    pass


class SolveError(ChemflowError):
    """求解が収束しない。"""
    pass


class ConstraintError(ChemflowError):
    """制約式の不整合。"""
    pass


class CanteraError(ChemflowError):
    """Cantera 平衡計算の失敗。"""
    pass
