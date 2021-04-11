# bandit
[バンディット問題の理論とアルゴリズム(本多淳也, 中村篤祥)](https://www.kspub.co.jp/book/detail/1529175.html) のPython実装を書いていくリポジトリ  
- 身内の輪読会用のため、担当章（Chapter6(p.95~109): 最適腕識別とA/Bテスト)のみ実装。
- 気が向いたら、他の章もやる。

## 内容物
- lib
  - アーム(正規乱数)、方策(逐次削除、LUCB)の実装。
  - アームの実装は、以下から流用。
    - https://github.com/johnmyleswhite/BanditsBook
- jn
  - jupyter上で最適腕識別の実行結果を表示。
  - TODO: 標本平均や誤識別率の推移も見ていきたい。(気が向いたらやる)

- run_se_policy.py
- run_lucb_policy.py
  - .py形式の実行ファイル。最終結果と末尾5件の探索結果をprintする。
