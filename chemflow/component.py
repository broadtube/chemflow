"""成分クラス"""


class Component:
    """化学成分を表すクラス。

    Parameters
    ----------
    name : str
        成分名
    mw : float
        分子量 [g/mol]
    normal_volume : float
        ノルマル体積 [L/mol] (デフォルト: 22.414 理想気体)
    """

    def __init__(self, name: str, mw: float, normal_volume: float = 22.414):
        self.name = name
        self.formula = name  # 示性式（name と同一）
        self.mw = mw
        self.normal_volume = normal_volume

    def __repr__(self) -> str:
        return f"Component({self.name!r}, mw={self.mw})"
