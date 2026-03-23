# Requirements Document

## Introduction
chemflow の直感的APIリデザイン。現在の冗長なAPI（Component手動定義、Flowsheet手動構築、fix_stream等）を、演算子ベースの記法・示性式による自動分子量計算・柔軟な入力形式・任意制約条件・Gibbsリアクター対応に置き換える。内部のresidual方式（連立方程式として求解）は維持しつつ、ユーザー向けAPIを大幅に簡素化する。

### 対象ユースケース
- **パターン1**: 3ストリーム混合 → Gibbsリアクター（Cantera使用、850°C / 2MPaG）
- **パターン2**: 循環系（Mixer → 転化率指定Reactor → 均一組成Splitter）＋ストリーム間制約

## Requirements

### Requirement 1: 示性式による自動分子量計算
**Objective:** ユーザーとして、H2・CO2・CH3COOH等の示性式文字列を渡すだけで分子量が自動計算されてほしい。Componentクラスを手動定義したくない。

#### Acceptance Criteria
1. WHEN ストリーム定義時に示性式文字列（例: "H2", "CO2", "CH4", "H2O", "NH3", "CH3COOH"）が渡された THEN chemflow SHALL molmassライブラリを用いて分子量を自動計算し、内部でComponentを生成する
2. WHEN 不正な示性式が渡された THEN chemflow SHALL 明確なエラーメッセージを返す

### Requirement 2: Stream の簡潔な定義
**Objective:** ユーザーとして、ストリームを1行で直感的に定義したい。名前はオプションとし、複数の単位系（basis）で入力できるようにしたい。

#### Acceptance Criteria
1. WHEN dict形式 `{"N2": 20, "H2": 60}` が渡された THEN Stream SHALL 各成分のモル流量として解釈し、ストリームを生成する（デフォルトbasis="mol"）
2. WHEN `basis="mass"` が指定された THEN Stream SHALL 入力値を重量流量として解釈し、内部でモル流量に変換する
3. WHEN `basis="normal_volume"` が指定された THEN Stream SHALL 入力値をノルマル体積流量として解釈し、内部でモル流量に変換する
4. WHEN `basis="mole_frac"` と `total` パラメータが指定された THEN Stream SHALL 入力値をモル比率として解釈し、total値と合わせてモル流量を算出する
5. WHEN `basis="mass_frac"` と `total` パラメータが指定された THEN Stream SHALL 入力値を重量比率として解釈し、total値と合わせてモル流量を算出する
6. WHEN `basis="volume_frac"` と `total` パラメータが指定された THEN Stream SHALL 入力値を体積比率として解釈し、total値と合わせてモル流量を算出する
7. WHEN 比率系basis（mole_frac / mass_frac / volume_frac）が指定され、totalパラメータが省略された THEN Stream SHALL 組成情報のみ保持し、合計流量を未知数として扱う
8. WHEN 成分ごとに単位を指定するタプル形式 `{"N2": (20, "mol"), "H2": (120, "mass")}` が渡された THEN Stream SHALL 各成分を個別に変換する（絶対量系のみ混在可能、比率系との混在は不可）
9. WHEN `name` パラメータが省略された THEN Stream SHALL 名前なしで生成される（nameはオプション）
10. WHEN CSV ファイルパスが渡された THEN Stream.from_csv() SHALL ファイルを読み込みストリームを生成する
11. WHEN 成分リストのみが渡された（値なし） THEN Stream SHALL 全成分のモル流量を0で初期化し、求解対象の未知ストリームとして扱う

### Requirement 3: 演算子による装置接続
**Objective:** ユーザーとして、`+`（混合）や `*`（分割）の演算子でストリーム間の関係を記述し、Flowsheetへの登録を自動化したい。

#### Acceptance Criteria
1. WHEN `stream_c = stream_a + stream_b` が実行された THEN chemflow SHALL Mixer残差式（C - A - B = 0）を自動登録し、新しいストリームCを返す
2. WHEN 3つ以上のストリームが加算 `stream_d = stream_a + stream_b + stream_c` された THEN chemflow SHALL 全入口の混合残差式を自動登録する
3. WHEN `stream_b = stream_a * 0.4` が実行された THEN chemflow SHALL 分割残差式（B - A × 0.4 = 0）を自動登録し、新しいストリームBを返す
4. WHEN 加算時に成分が異なるストリーム同士が混合された THEN chemflow SHALL 全ストリームの成分和集合を取り、存在しない成分は0として扱う

### Requirement 4: 等式制約による関係定義（eq関数）
**Objective:** ユーザーとして、`eq(D, E + B)` のような記法で「D = E + B」という関係式を定義したい。循環系で既存ストリーム間の関係を記述するために必要。

#### Acceptance Criteria
1. WHEN `eq(D, E + B)` が実行された THEN chemflow SHALL 残差式（D - E - B = 0）をFlowsheetに登録する
2. WHEN eq() の両辺にストリームの演算結果が含まれる THEN chemflow SHALL 適切な残差式を生成する

### Requirement 5: 転化率指定リアクター（Stoichiometric Reactor）
**Objective:** ユーザーとして、反応式・基準成分・転化率を指定して反応器を簡潔に定義したい。

#### Acceptance Criteria
1. WHEN `stream.react({"CO": -2, "H2": -2, "CH3COOH": 1}, key="CO", conversion=0.9)` が実行された THEN chemflow SHALL 化学量論に基づく物質収支残差式を自動登録し、出口ストリームを返す
2. WHEN reactの化学量論dictに入口ストリームに存在しない成分（生成物）が含まれる THEN chemflow SHALL 該当成分を自動的に出口ストリームに追加する
3. WHEN key パラメータに成分の示性式文字列が渡された THEN chemflow SHALL 該当成分を基準成分として転化率を適用する

### Requirement 6: Gibbsリアクター（Cantera連携）
**Objective:** ユーザーとして、温度・圧力・平衡種を指定するだけでギブズ自由エネルギー最小化による平衡計算を行いたい。

#### Acceptance Criteria
1. WHEN `stream.gibbs_react(T=850, P=2e6, species=["CO2","CH4","H2O","CO","H2"])` が実行された THEN chemflow SHALL Canteraを用いてギブズ自由エネルギー最小化を行い、平衡組成の出口ストリームを返す
2. WHEN 温度の単位が摂氏で渡された THEN chemflow SHALL 内部でケルビンに変換してCanteraに渡す
3. WHEN 圧力がゲージ圧(G付き)で指定された THEN chemflow SHALL 大気圧を加算して絶対圧に変換する
4. WHEN species リストに入口に存在しない成分が含まれる THEN chemflow SHALL 該当成分を出口ストリームに自動追加する（初期モル流量0）
5. WHILE Gibbsリアクターが求解される THE chemflow SHALL Canteraの平衡計算結果を残差式の一部として組み込み、全体の連立方程式に統合する

### Requirement 7: 任意制約条件（constrain関数）
**Objective:** ユーザーとして、ストリームのプロパティ間の任意の制約（total流量指定、ストリーム間の流量等式等）を記述したい。

#### Acceptance Criteria
1. WHEN `constrain(C.total_molar_flow, 30)` が実行された THEN chemflow SHALL 残差式（C.total_molar_flow - 30 = 0）を登録する
2. WHEN `constrain(A.total_mass_flow, E.total_mass_flow)` が実行された THEN chemflow SHALL 残差式（A.total_mass_flow - E.total_mass_flow = 0）を登録する
3. WHEN `constrain(C.total_normal_volume_flow, 500)` が実行された THEN chemflow SHALL ノルマル体積流量に関する残差式を登録する

### Requirement 8: Flowsheetの自動構築と求解
**Objective:** ユーザーとして、add_stream / add_unit / fix_stream を手動で呼ぶことなく、演算子や関数の使用だけで自動的にFlowsheetが構築されてほしい。

#### Acceptance Criteria
1. WHEN 演算子（+, *）やreact/gibbs_react/eq/constrainが実行された THEN chemflow SHALL 関連するストリームと残差式をグローバルFlowsheetに自動登録する
2. WHEN ストリームが値付きで生成された（dictで流量指定） THEN chemflow SHALL そのストリームを固定値（既知）として自動判定する
3. WHEN ストリームが成分リストのみ、または組成のみ（totalなし）で生成された THEN chemflow SHALL そのストリームを未知（求解対象）として扱う
4. WHEN `solve()` が呼び出された THEN chemflow SHALL 全登録済み残差式を収集し、scipy.optimize.root で連立方程式を求解する
5. WHEN 求解が収束した THEN chemflow SHALL 全ストリームのモル流量を更新する
6. WHEN 求解が収束しなかった THEN chemflow SHALL エラー情報を含む結果を返す

### Requirement 9: 結果の出力
**Objective:** ユーザーとして、求解後の全ストリーム情報（モル流量、重量流量、ノルマル体積流量、各比率）を一覧表示したい。

#### Acceptance Criteria
1. WHEN 求解完了後に結果表示が要求された THEN chemflow SHALL 全ストリームについてモル流量・モル比率・重量流量・重量比率・ノルマル体積流量・体積比率を表示する
2. WHEN 個別ストリームの情報が参照された THEN chemflow SHALL 該当ストリームのプロパティ（total_molar_flow, total_mass_flow, total_normal_volume_flow, mole_fractions等）を返す

### Requirement 10: パターン1の実現（Gibbs平衡計算）
**Objective:** ユーザーとして、以下のコードイメージでパターン1を記述・求解したい。

```python
A = Stream({"CO2": 5}, basis="mol")
B = Stream({"CH4": 5}, basis="mol")
C = Stream({"H2O": 5}, basis="mol")
D = A + B + C
E = D.gibbs_react(T=850, P="2MPaG", species=["CO2","CH4","H2O","CO","H2"])
solve()
```

#### Acceptance Criteria
1. WHEN 上記パターン1のコードが実行された THEN chemflow SHALL 3ストリームを混合し、850°C / 2MPaG でのギブズ平衡組成を計算して結果ストリームEを返す

### Requirement 11: パターン2の実現（循環系）
**Objective:** ユーザーとして、以下のコードイメージでパターン2を記述・求解したい。

```python
A = Stream(composition=E, basis="mole_frac")  # パターン1のEと同組成、流量は未知
B = Stream(components=["CO2","CH4","H2O","CO","H2","CH3COOH"])  # 未知
C = Stream(components=["CO2","CH4","H2O","CO","H2","CH3COOH"])  # 未知
eq(C, A + B)
D = C.react({"CO": -2, "H2": -2, "CH3COOH": 1}, key="CO", conversion=0.9)
E_out = Stream(components=["CO2","CH4","H2O","CO","H2","CH3COOH"])  # 未知
eq(D, E_out + B)
constrain(D.mole_fractions, E_out.mole_fractions)  # 均一組成分割
constrain(C.total_molar_flow, 30)
constrain(A.total_mass_flow, E_out.total_mass_flow)
solve()
```

#### Acceptance Criteria
1. WHEN 上記パターン2のコードが実行された THEN chemflow SHALL 循環系の連立方程式を構築し、全未知ストリームのモル流量を求解する
2. WHEN 求解完了後 THEN chemflow SHALL 物質収支が成立していることを検証可能な結果を返す
