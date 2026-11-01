# 学習の再実行

`data/training/short_history.jsonl.gz` に、採用モデルで実際に使った44特徴と教師値を保存した。
全6,342窓のうち学習6,096窓・6物理条件、校正246窓・別の2条件。評価用荷物を学習へ混ぜない。
元データのSHA256・選択列・圧縮データのSHA256をmanifest.jsonに残した。

特徴は判断時点までの0.45秒。正解は同じ候補指令を継続した0.25秒間の最大追加水平ズレ。
停止付近や途中で指令が変わる窓は元の抽出時に除外している。停止付近の打切りによる偏りは残る。
既存ログの重複を除き、物理条件を同じ重み、その中のエピソードも同じ重みにして学習する。
学習にある物理ラベルは重み付けと分割のためで、推論器へ与える特徴ではない。

Docker内で学習用環境を作る。ROS実行用のPythonへpipで上書きインストールしない。

```bash
cd /workspace/cargo_ws
python3 -m venv .venv-learning
.venv-learning/bin/python -m pip install -r requirements-learning.txt
.venv-learning/bin/python scripts/train_short_model.py --cross-validate
```

ExtraTrees 64本、深さ10、葉の最少標本5、seed 42で固定。交差検証は物理条件を1つずつ丸ごと外す6foldで、窓のランダム分割ではない。
パラメータ探索や結果を見た閾値変更は行わない。校正は別荷物の正残差95%分位で、相関する窓と少数条件なので安全保証ではない。

出力先は新しい `logs/retraining/日時/`。`--output` で既存でないフォルダを指定できる。
元の学習済みモデルを上書きしない。
再学習した木・校正値を採用モデルと照合し、portable JSONとscikit-learnの予測差も記録する。
並列加算の順序で校正の浮動小数点末尾が変わり得るため、木の完全一致に加えて校正差1e-12以内も検査する。

`models/cargo_experience_v1.json` は固定済み成果物で、通常実行にscikit-learnや学習用仮想環境は不要。
報告されている全区間の平均誤差ではゼロ予測が勝つ場合があり、学習の優位性を平均誤差だけで主張しない。
運搬での比較結果と、荷物・経路・使用場面の範囲を併記する。
