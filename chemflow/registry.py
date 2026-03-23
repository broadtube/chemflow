"""ComponentRegistry: 示性式文字列 → Component の変換とキャッシュ。"""

from chemflow.component import Component
from chemflow.errors import FormulaError


class ComponentRegistry:
    """示性式からComponentを自動生成するレジストリ。molmass使用。"""

    _cache: dict[str, Component] = {}

    @classmethod
    def get(cls, formula: str) -> Component:
        """示性式からComponentを取得。キャッシュあり。"""
        if formula in cls._cache:
            return cls._cache[formula]
        try:
            from molmass import Formula as MolFormula
            mf = MolFormula(formula)
            mw = mf.mass
        except Exception as e:
            raise FormulaError(f"Unknown formula: '{formula}' ({e})") from e
        comp = Component(formula, mw=mw)
        cls._cache[formula] = comp
        return comp

    @classmethod
    def get_many(cls, formulas: list[str]) -> list[Component]:
        """複数の示性式を一括取得。"""
        return [cls.get(f) for f in formulas]

    @classmethod
    def clear_cache(cls) -> None:
        """キャッシュをクリア。"""
        cls._cache.clear()
