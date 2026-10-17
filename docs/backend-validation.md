# 専用バックエンドへの移行確認

以下は、CubeCL・CubeK の公開パッケージをそのまま使い、
必要な変更を専用バックエンドへ隔離した時点の検証記録です。
検証時は別 crate でしたが、公開構成では同じバックエンドのソースを本体に同梱しています。
実装の基点は `21264476269b4767d4feb98d175d78a6bb8e4331` です。
旧構成のソースは `513b7a9` で確認できます。

## 動作確認

Nu 0.116.1 から、デバイス列挙・変換・GPU 値の寿命・形状変更・要素演算と、
3 GPU の行列積・集約・NPY 入出力を実行し、13件すべて成功しました。
サーバー型が一致するときの呼び出しと、不一致を実行前に拒否することも Rust テストで確認しています。
SPIR-V の検証は、公開版コンパイラに後処理を加えた実際の出力を対象にしています。

## 変更前後の計測

2026-10-17、同一ホストで旧 0.9.0 と新 0.10.0 を順に測定し、この組を2回実行しました。
サイズは256・512・1024角、f64、warmup 2回・計測10回、演算スレッド数1です。
各実行840件、合計3360件を計測しました。各サイズ・演算・入力経路の結果は計時前に照合しています。
ビルドとベンチマークは同時に実行していません。
CPU codec は全実行で同じバイナリを使い、プラグインの変更だけを比較しました。
1組目の旧版は既存の Nu registry を使いました。2組目は両版とも空の Nu 設定と専用 registry を使い、以下の表には2組目を採用しています。
測定範囲・入力・warm cache の条件は[行列積ベンチマーク](benchmarks.md)と同じです。

2組目の1024角・NPY読込＋行列積の平均値（ms）：

| 経路 | 旧 0.9.0 | 新 0.10.0 |
| --- | ---: | ---: |
| CPU（Nu matrix） | 255.93 | 266.36 |
| RX 7800 XT 通常入力 | 38.28 | 43.57 |
| RX 7800 XT 共有入力 | 22.89 | 25.44 |
| RX 7600 通常入力 | 32.42 | 32.40 |
| RX 7600 共有入力 | 24.08 | 24.67 |
| 内蔵 Radeon 780M 通常入力 | 69.50 | 70.08 |
| 内蔵 Radeon 780M 共有入力 | 43.87 | 44.43 |

短い処理には大きな変動があり、RX 7800 XT の256角・通常入力の平均は、
1組目で旧9.58／新21.73 ms、2組目で旧20.12／新39.20 msでした。
この測定だけで性能が同等とは判断していません。README の数値は従来の測定で、
この比較に差し替えたものではありません。

全サイズ・scale／行列積・Nu取出の有無・全GPUの結果を収録しています。

- 1組目：[旧 raw](bench/backend-before-1.csv)／[新 raw](bench/backend-after-1.csv)
- 2組目：[旧 raw](bench/backend-before-2.csv)／[新 raw](bench/backend-after-2.csv)
- [全組の集計](bench/backend-comparison-summary.csv)：`run` が組、`phase` が変更前後

使用バイナリの SHA-256：

```text
旧 plugin: 8ebe566ab40637d3c8da8aeeae148655b21bd3475670640c7cbc40a7c52545bf
新 plugin: c9d5be6a94e4133e1630ae422de16691e217bff3d163d014149caa80a5b6220b
CPU codec: 92f88e9265020d3e9159f2af04ae1176273a4f0beef77eeefb6b1d5838348b8b
```

## 再実行

各バージョンを別の場所へインストールし、空の Nu 設定と専用 registry を使って測定します。
次は Nushell の例です。`/path/to/nu_plugin_vulkan_matrix` と codec のパスを置き換えてください。
旧構成と新構成で registry・出力先を分けます。

```nu
'' | save empty.nu
nu --config empty.nu --env-config empty.nu --no-history --plugin-config plugins.msgpackz -c 'plugin add /path/to/nu_plugin_vulkan_matrix'
with-env {RAYON_NUM_THREADS: '1', MATMUL_NUM_THREADS: '1'} {
    nu --config empty.nu --env-config empty.nu --no-history --plugin-config plugins.msgpackz validation/benches/file-compute.nu --sizes 256,512,1024 --codec /path/to/npy-codec --output results
}
```
