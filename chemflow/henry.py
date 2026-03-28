"""Henry則定数の管理: 内蔵データ + オンライン取得 + キャッシュ

van't Hoff式: H(T) = Hcp * exp(Tderiv * (1/T - 1/T0))
  Hcp: 298.15K での Henry 溶解度定数 [mol/(m3·Pa)]
  Tderiv: d ln(Hcp) / d(1/T) [K]
  T0 = 298.15 K

chemflow内部では H [Pa] (= 1 / (Hcp * R * T) 相当のスケール) を使用。
変換: H_pa = 1.0 / (Hcp_mol_m3_pa * R * T)  ※ 理想希薄溶液近似
簡易変換: H_pa ≈ 1.0 / Hcp_mol_m3_pa  (mol/m3·Pa → Pa の逆数)
ただしHenry則 x_i = P_i / H_i で使う H_i [Pa] は:
  H_i = 1 / (Hcp * Vm_water)
  Vm_water ≈ 18.015e-6 m3/mol (水のモル体積)
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

# 水のモル体積 [m3/mol]
VM_WATER = 18.015e-6
T0 = 298.15  # 基準温度 [K]

# --- 内蔵データ (Sander 2023 推奨値, T0=298.15K) ---
# Hcp: [mol/(m3·Pa)], Tderiv: [K]
# 出典: Sander, R., Atmos. Chem. Phys., 23, 10901–12440, 2023.
BUILTIN_DATA: dict[str, dict] = {
    "H2":      {"Hcp": 7.8e-6,  "Tderiv": 500,  "cas": "1333-74-0"},
    "N2":      {"Hcp": 6.4e-6,  "Tderiv": 1300, "cas": "7727-37-9"},
    "O2":      {"Hcp": 1.3e-5,  "Tderiv": 1500, "cas": "7782-44-7"},
    "CO":      {"Hcp": 9.5e-6,  "Tderiv": 1300, "cas": "630-08-0"},
    "CO2":     {"Hcp": 3.3e-4,  "Tderiv": 2400, "cas": "124-38-9"},
    "CH4":     {"Hcp": 1.4e-5,  "Tderiv": 1600, "cas": "74-82-8"},
    "NH3":     {"Hcp": 5.9e-1,  "Tderiv": 4200, "cas": "7664-41-7"},
    "H2S":     {"Hcp": 1.0e-3,  "Tderiv": 2100, "cas": "7783-06-4"},
    "SO2":     {"Hcp": 1.2e-2,  "Tderiv": 2900, "cas": "7446-09-5"},
    "CH3CHO":  {"Hcp": 1.3e-1,  "Tderiv": 5900, "cas": "75-07-0"},
    "CH3COOH": {"Hcp": 4.1e+3,  "Tderiv": 6300, "cas": "64-19-7"},
    "CH3OH":   {"Hcp": 2.2e+0,  "Tderiv": 5200, "cas": "67-56-1"},
    "C2H5OH":  {"Hcp": 1.9e+0,  "Tderiv": 6600, "cas": "64-17-5"},
    "HCHO":    {"Hcp": 3.2e+3,  "Tderiv": 6800, "cas": "50-00-0"},
    "HCOOH":   {"Hcp": 8.9e+3,  "Tderiv": 5700, "cas": "64-18-6"},
}


def _hcp_at_T(Hcp_298: float, Tderiv: float, T_kelvin: float) -> float:
    """van't Hoff式で任意温度の Hcp を計算する。"""
    return Hcp_298 * math.exp(Tderiv * (1.0 / T_kelvin - 1.0 / T0))


def henry_pa(Hcp_298: float, Tderiv: float, T_kelvin: float) -> float:
    """Henry定数を Pa 単位で返す。

    H [Pa] = 1 / (Hcp [mol/(m3·Pa)] * Vm_water [m3/mol])
    """
    hcp = _hcp_at_T(Hcp_298, Tderiv, T_kelvin)
    if hcp <= 0:
        return 1e15  # 極めて溶けにくいとして扱う
    return 1.0 / (hcp * VM_WATER)


# --- キャッシュ ---
_CACHE_DIR = Path.home() / ".chemflow" / "henry_cache"
_runtime_cache: dict[str, dict] = {}


def _load_cache(formula: str) -> dict | None:
    """ローカルキャッシュから読み込む。"""
    if formula in _runtime_cache:
        return _runtime_cache[formula]
    cache_file = _CACHE_DIR / f"{formula}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            _runtime_cache[formula] = data
            return data
        except Exception:
            pass
    return None


def _save_cache(formula: str, data: dict) -> None:
    """ローカルキャッシュに保存する。"""
    _runtime_cache[formula] = data
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{formula}.json"
        cache_file.write_text(json.dumps(data))
    except Exception:
        pass  # キャッシュ書き込み失敗は無視


# --- オンライン取得 ---

def _get_cas_from_pubchem(formula: str) -> str | None:
    """PubChem API で化学式から CAS 番号を取得する。

    formula API は非同期になることがあるため、name API も併用する。
    """
    import urllib.request

    cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")

    # まず formula の IUPAC名を取得し、name API でCASを探す
    for endpoint in [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/{formula}/synonyms/JSON?MaxRecords=1",
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{formula}/synonyms/JSON",
    ]:
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "chemflow/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            # 非同期応答（Waiting）の場合はスキップ
            if "Waiting" in data:
                continue
            synonyms = (
                data.get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", [])
            )
            for syn in synonyms:
                if cas_pattern.match(syn):
                    return syn
        except Exception:
            continue

    # molmass で IUPAC 名を取得して name API を試す
    try:
        from molmass import Formula as MolFormula
        name = MolFormula(formula).formula  # Hill notation
        # PubChem name API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{formula}/synonyms/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "chemflow/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        synonyms = (
            data.get("InformationList", {})
            .get("Information", [{}])[0]
            .get("Synonym", [])
        )
        for syn in synonyms:
            if cas_pattern.match(syn):
                return syn
    except Exception:
        pass

    return None


def _scrape_henry_from_sander(cas: str) -> dict | None:
    """henrys-law.org から Henry 定数をスクレイピングする。"""
    import urllib.request
    import html as html_mod

    url = f"https://www.henrys-law.org/henry/casrn/{cas}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "chemflow/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # HTML entities をデコード (&#8722; → −, &times; → × 等)
    text = html_mod.unescape(raw_html)

    # Hcp 値のパターン: "1.9×10−5" or "1.9 × 10−5"
    # <td> 内の科学表記: mantissa × 10<sup>符号付き指数</sup>
    hcp_pattern = re.compile(
        r"(\d+\.?\d*)\s*×\s*10<sup>\s*([−+-]?\d+)\s*</sup>"
    )

    hcp_values = []
    for m in hcp_pattern.finditer(text):
        mantissa = float(m.group(1))
        exp_str = m.group(2).replace("−", "-").replace("+", "")
        exponent = int(exp_str)
        hcp_values.append(mantissa * (10 ** exponent))

    # Tderiv: テーブルの <td> に整数値として入っている
    # "d ln Hcp / d(1/T)" 列の値を探す
    # 典型的には <td>2400</td> のような形
    tderiv_values = []
    # recommendation 行付近の整数を探す
    td_int_pattern = re.compile(r"<td>\s*(\d{3,5})\s*</td>")
    for m in td_int_pattern.finditer(text):
        val = int(m.group(1))
        if 100 <= val <= 20000:  # 妥当な範囲
            tderiv_values.append(val)

    if not hcp_values:
        return None

    # 最初の値（推奨値が先頭）
    hcp = hcp_values[0]
    tderiv = tderiv_values[0] if tderiv_values else 0

    return {"Hcp": hcp, "Tderiv": tderiv, "cas": cas, "source": "henrys-law.org"}


def fetch_henry_data(formula: str) -> dict | None:
    """化学式から Henry 定数データを取得する。

    検索順序:
    1. 内蔵データ
    2. ローカルキャッシュ
    3. オンライン (PubChem → henrys-law.org)

    Returns
    -------
    dict with keys: Hcp, Tderiv, cas, source (optional)
    """
    # 1. 内蔵データ
    if formula in BUILTIN_DATA:
        return BUILTIN_DATA[formula]

    # 2. ローカルキャッシュ
    cached = _load_cache(formula)
    if cached is not None:
        return cached

    # 3. オンライン取得
    # CAS番号を取得
    cas = None
    # 内蔵データからCAS番号を逆引き
    for f, d in BUILTIN_DATA.items():
        if f == formula:
            cas = d.get("cas")
            break

    if cas is None:
        cas = _get_cas_from_pubchem(formula)

    if cas is None:
        return None

    # henrys-law.org からスクレイピング
    data = _scrape_henry_from_sander(cas)
    if data is not None:
        _save_cache(formula, data)
        return data

    return None


def get_henry_pa(formula: str, T_celsius: float) -> float | None:
    """化学式と温度から Henry 定数 [Pa] を取得する。

    Parameters
    ----------
    formula : str
        化学式 (例: "CO2", "CH3COOH")
    T_celsius : float
        温度 [°C]

    Returns
    -------
    float or None
        Henry定数 [Pa]。データが見つからない場合は None。
    """
    data = fetch_henry_data(formula)
    if data is None:
        return None
    T_kelvin = T_celsius + 273.15
    return henry_pa(data["Hcp"], data["Tderiv"], T_kelvin)


def get_henry_constants(formulas: list[str], T_celsius: float) -> dict[str, float]:
    """複数成分の Henry 定数 [Pa] を一括取得する。

    Parameters
    ----------
    formulas : list[str]
    T_celsius : float

    Returns
    -------
    dict[str, float]
        {formula: H_pa} 。データがない成分は含まれない。
    """
    result = {}
    for f in formulas:
        if f == "H2O":
            continue
        h = get_henry_pa(f, T_celsius)
        if h is not None:
            result[f] = h
    return result
