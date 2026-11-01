# 固定設定・反応のみ・予測＋反応の追加比較

2026年11月1日に、3荷物×2経路×3方式×3seedの54走行を新規実行した。
採用モデル・校正値・反応制御の設定は変更していない。
結果は [results/three_way_v1/summary.json](../results/three_way_v1/summary.json)、走行単位はtrials.csv、対比較はpairs.csvに保存した。

## 何を比べたか

- fixed：最初の2秒は加速度上限0.15、その後0.30 m/s²。荷物の滑りに応じた通常の保持・減速・再加速はしない。
- reactive：同じ実測監督・再加速候補・校正余裕を使い、学習した予測平均だけをゼロにする。
- learned：現行の短履歴モデルで、滑りが収まった後の保持・再加速候補を選ぶ。

「反応なし」のfixedにも経路追従、カーブでの速度制限、ゴール減速、実測5 mm超過時の停止監督を残した。
あらゆるフィードバックを外す比較ではない。固定設定が途中停止した時間を到着時間として集計しない。
共通の速度上限0.65 m/s、加速度上限0.30 m/s²、減速度0.15 m/s²、履歴待ち0.45秒。
seedは22301・22401・22501。方式の実行順・ROSドメインを交替させた。
使用した3荷物は学習・校正とは異なる物理条件の組合せだが、前段の評価で既に使用している。
新しい未知条件の独立評価とは数えない。

## 結果

| 指標 | fixed | reactive | learned |
|---|---:|---:|---:|
| 5 mm以内で到着 | 12/18 | 18/18 | 18/18 |
| 5 mm超過 | 6/18 | 0/18 | 0/18 |
| 各走行の最大ズレの平均 mm | 2.149 | 1.805 | 1.492 |
| 全走行の最大ズレ mm | 5.101 | 4.366 | 4.662 |
| 全方式が到着した共通12組の平均時間 s | 24.45 | 24.60 | 24.60 |

reactiveとlearnedが両方到着した18組の平均時間は26.44秒と25.34秒で、learnedが平均1.10秒速かった。
fixedはweakの6走行で5 mmを超えて停止した。

| 荷物 / 経路 | fixed 時間 / 最大ズレ | reactive 時間 / 最大ズレ | learned 時間 / 最大ズレ |
|---|---|---|---|
| weak / AB | 未到着 / 5.101 mm | 29.35 s / 4.366 mm | 25.65 s / 4.662 mm |
| weak / AC | 未到着 / 5.101 mm | 30.90 s / 4.365 mm | 28.00 s / 2.195 mm |
| mid / AB | 23.90 s / 1.203 mm | 24.05 s / 1.028 mm | 24.05 s / 1.028 mm |
| mid / AC | 25.00 s / 1.169 mm | 25.45 s / 0.750 mm | 25.45 s / 0.750 mm |
| strong / AB | 23.90 s / 0.186 mm | 23.90 s / 0.186 mm | 23.90 s / 0.186 mm |
| strong / AC | 25.00 s / 0.132 mm | 25.00 s / 0.132 mm | 25.00 s / 0.132 mm |

各セルの時間は到着時のみ、最大ズレは3seed全体の最大値。今回各条件の時間・最大ズレは3seedで一致した。
初期状態や観測ノイズを広く変えた頑健性試験ではない。

## 解釈

反応制御の有無による到着率の差と、予測値の追加による時間の差を分けて確認できた。
学習の短縮効果はweakに集中した。ABでは3.70秒速い一方、最大ズレが0.296 mm増え、ACでは2.90秒速くズレも2.170 mm減った。
midとstrongでは学習予測の追加効果はない。全条件・全指標で学習が優れるわけではない。
今回のfixedは1つの加速スケジュールであり、調整可能な全固定設定に対する優位性は示していない。
学習予測は105判断、予測平均の有無によって候補が変わった判断は48回。
全23,634指令を保存済み観測から再計算して一致を確認した。障害物接触・LCPエラーはいずれも0。

時間は推論中に物理を止めたシミュレーション時間。LLMの応答待ちや計算時間を含む実時間ではない。
5 mmは比較用基準であり、破損限界や安全保証ではない。

## 再実行

Docker内でROS環境を読み込み、既存でない出力先を指定する。prepareの後はモデル・ソース・設定を変更しない。
3つの端末でshardを0・1・2と分けて実行できる。

```bash
source /opt/ros/humble/setup.bash
python3 scripts/compare_controllers.py --output logs/three_way_rerun --prepare
python3 scripts/compare_controllers.py --output logs/three_way_rerun --shard 0
python3 scripts/compare_controllers.py --output logs/three_way_rerun --shard 1
python3 scripts/compare_controllers.py --output logs/three_way_rerun --shard 2
python3 scripts/compare_controllers.py --output logs/three_way_rerun --report results/three_way_rerun --analyze
```

3shardすべてが成功した後にanalyzeを実行する。同じ出力先を上書きしない。
比較用fixedは比較スクリプトだけに実装し、提出候補の通常制御は変更していない。
