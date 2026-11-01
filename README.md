# Cargo Transport AI

自然言語で工場内の搬送を指示し、荷物の反応を観測しながら走行を調整するROS 2 / Gazeboの課題です。
ローカルLLMが出発地・目的地・荷物・方針を解釈し、不足項目だけ確認します。
走行中は実測の滑りへ反応し、安定後の再加速を学習済み短履歴モデルで選びます。

## 提出対象

この公開リポジトリは、提出時点のコード・モデル・データ・検証記録をまとめたものです。
個人情報を含む開発用コミット履歴は公開対象に含めず、現在の提出版から新しい履歴を開始しています。
過去の結果JSONにあるソースハッシュ等は実験当時の記録です。実行方法は以下を参照してください。

## 起動

WindowsでDocker Desktopを起動し、リポジトリ直下のPowerShellで実行します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker\start.ps1
# ローカルLLMを初めて準備する場合のみ。モデル取得にネット接続が必要です。
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker\setup-local-llm.ps1
docker start cargo_robot_project cargo_ollama
docker exec -it --user ubuntu --workdir /workspace/cargo_ws cargo_robot_project bash
```

Docker内のLinux端末で実行します。

```bash
source /opt/ros/humble/setup.bash
python3 /workspace/cargo_ws/scripts/run_language_mission.py --backend ollama --gui
```

入力例：`検査するところまで工具箱を運んで`。
不足する出発地と方針だけ確認されたら、`部品置き場から、時間優先` と回答します。
一文で全条件を伝えることもできます。日本語のBackspace・矢印キーでの訂正に対応します。
GUIはブラウザの `http://localhost:6081/` で確認できます。

追加のAPI課金は不要です。OpenAI接続は任意で、明示的に選んだ場合だけ使います。
APIキーがある場合はローカルの `.env` に設定し、Gitへ含めません。
Ollamaが使えない場合に有料APIへ自動で切り替える処理はありません。

## LLMなしの動作確認

```bash
source /opt/ros/humble/setup.bash
python3 scripts/run_transport.py --route A C --sim-cargo slippery --gui
# 限定した語句パターンによる対話確認。自由な言い換えの評価ではありません。
python3 scripts/run_language_mission.py --backend rules --gui
```

A＝部品置き場、B＝組立場、C＝検査場。固定地図のA*経路生成で静的障害物を避けます。
慎重・標準・時間優先による速度／加速度上限は、それぞれ0.45/0.15、0.55/0.30、0.65/0.30（m/s・m/s²）です。
高さや壊れやすさなどの条件によっては、指定より慎重な上限を使います。
摩擦係数・重心を聞いたり、荷物名から正確な値を推定したりはしません。

## 構成

```text
scripts/
  run_language_mission.py       自然言語の対話と搬送
  mission_dialogue.py           条件保持・聞き返し・検証
  mission_language_backend.py   rules / OpenAI接続
  mission_ollama_backend.py     ローカルLLMと原文照合
  mission_execution.py          検証済み指示から実行への接続
  run_transport.py              単発デモと比較方式
  train_short_model.py          同じ学習データからの再学習
  verify_submission.py          対話から実運搬までの検証
  check_live_language.py        実LLMの対話評価
  cargo_transport/
    field.py                   経路・world・図の生成
    physics.py                 シミュレーションの荷物設定
    controller.py              反応と短履歴による再加速選択
    predictor.py               44特徴と学習済みモデルの推論
    simulation.py              ROS 2 / Gazeboの固定ステップ実行
    io.py                      ログ・ハッシュ・補助プログラム
    paths.py                   プロジェクト内のパス
configs/                       地図と自然言語の設定
models/cargo_experience_v1.json 固定済みモデル（使用variantはshort）
data/training/                 公開用の学習・校正窓と出典
tests/                         現行機能と保存済み走行の回帰テスト
results/                       採用判断に関わる代表結果・監査
docs/                          構成、学習、検証と限界
docker/                        Windowsからの環境・LLM準備
worlds/ tools/                 台車モデルと固定ステップ補助
```

過去の試作・使わないモデル・重複した描画スクリプトは提出版から除きました。
生ログ・仮想環境・秘密情報は公開対象に含めません。

## テストと通し確認

```bash
python3 -m unittest discover -s tests
source /opt/ros/humble/setup.bash
python3 scripts/verify_submission.py
# 実際のローカルLLMを使う1件の通し確認
python3 scripts/verify_submission.py --backend ollama --gui
```

検証スクリプトは新しいログフォルダへ保存します。`--output` を指定する場合も既存の出力は上書きしません。
`run_transport.py --method zero` は予測平均をゼロにした同じ制御、`--method gentle` は穏やかな再加速の比較用です。
再学習は [docs/learning.md](docs/learning.md)、内部構成は [docs/architecture.md](docs/architecture.md) を参照してください。

## 確認した結果と限界

提出候補の確認では3荷物×2経路を各5回、対照も含めて60走行しました。
短履歴モデル・ゼロ予測対照とも30/30が比較用5 mm以内で到着。
平均時間は25.34秒と26.44秒で、短履歴が約1.10秒速かった一方、全走行の最大ズレは4.662 mmと4.366 mmでした。
すべての指標で優位、あらゆる未知荷物で最適、という結果ではありません。

実Ollamaの不足条件確認・日本語訂正・GUI・運搬・結果保存も通し確認しました。
整理前後の一致と整理後の確認は [docs/validation.md](docs/validation.md) にまとめます。
過去の結果JSONは当時のソースハッシュを残した記録です。現在のソースと一致するという意味ではありません。

初回発進と最終減速は予測対象外で、モデルは滑り後に安定した際の保持・再加速選択に使います。
時間は推論中に物理を止めたシミュレーション時間。5 mmは破損限界ではなく比較用の基準です。
動的障害物・視覚認識・積み込み・実機のリアルタイム遅延は対象外です。

環境はWindows / Docker Desktop、Ubuntu 22.04、ROS 2 Humble、Gazebo Classic 11.10.2です。
Dockerイメージは起動スクリプトで確認済みdigestを固定しています。講義資料やイメージ自体は配布しません。
公開前の確認は [docs/submission.md](docs/submission.md) を参照してください。

## 反応なしを含む追加比較

3荷物×2経路×3方式×3seedの54走行を追加しました。
5 mm以内の到着は固定設定12/18、反応のみ18/18、予測＋反応18/18でした。
予測＋反応は反応のみより平均1.10秒速く、改善は滑りやすい荷物の再加速に集中しました。
固定設定が到着できた共通12組では予測の追加効果はなく、反応を使う両方式は固定設定より平均0.15秒遅い結果です。
条件・途中停止の扱い・再実行手順は [docs/three-way-comparison.md](docs/three-way-comparison.md) を参照してください。
