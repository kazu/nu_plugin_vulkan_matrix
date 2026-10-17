# NPY 入出力の速度比較

Nu の値から GPU へ渡す場合と、NPY ファイルを読み込む場合、結果を NPY に保存する場合を比較します。
行列積を含む比較は[NPY 読込＋計算](benchmarks.md)を参照してください。

## 結果

1024×1024、f64、10回平均。単位は ms です。

| GPU | Nu の値から入力 | NPY 通常入力 | NPY 共有入力 | NPY 出力（同期込み） |
|---|---:|---:|---:|---:|
| RX 7800 XT | 292.391 | 13.803 | 1.272 | 18.088 |
| RX 7600 | 298.395 | 16.030 | 10.291 | 18.755 |
| Radeon 780M（Ryzen 9 8945HS 内蔵） | 284.528 | 2.751 | 1.420 | 6.700 |

共有入力はマッピングの import と行列の登録までを測っています。その後の演算は含みません。
Nu の値からの入力にはシリアライズなどの負担も含むため、この表は DMA 単独の時間ではありません。
NPY 出力にはヘッダー作成、領域確保、GPU コピー、ファイルとディレクトリの同期、ファイルの確定を含みます。

## 条件と再実行

- 測定日：2026-10-17。ソースは commit `93482d72d4c387d8b29bb1d2330ad9b60124ca10`。
- 測定時の plugin 実行ファイル SHA-256：`8ebe566ab40637d3c8da8aeeae148655b21bd3475670640c7cbc40a7c52545bf`。
- 入力は単位行列の2倍、64／256／1024角、f64。warm page cache、warmup 2回、測定10回。
- `RAYON_NUM_THREADS=1`、`MATMUL_NUM_THREADS=1`。測定中に本作業のビルドや別ベンチは実行していません。
- 3 GPU × 3サイズ × 4経路の全36条件・360測定。入力生成・準備と結果の照合は計時外です。
- 最大 RSS は187592 KiB。仮想メモリ上限は4 GiB。

plugin を導入した環境でリポジトリ直下から実行します。保存先は新しいファイル名にしてください。

Nushell:

```nu
with-env {RAYON_NUM_THREADS: '1', MATMUL_NUM_THREADS: '1'} {
    nu validation/benches/file-io.nu | save file-io-results.csv
}
```

[生データ](bench/file-io.csv)と[全サイズの集計](bench/file-io-summary.csv)を収録しています。
