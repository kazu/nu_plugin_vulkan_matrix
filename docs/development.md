# 開発と検証

ビルド・実機検証・性能比較の手順をまとめます。コマンドはリポジトリ直下で実行してください。

## 構成

- `src/gpu.rs`: device 選択と ndarray 転送。
- `src/matmul.rs`: CubeK launch の共有と、常駐行列の積。
- `src/layout.rs`、`src/elementwise.rs`: buffer を共有する形状変更、GPU 生成と型別の要素演算。
- `src/reduction.rs`、`src/reduce_kernel.rs`: GPU 集約 API と CubeK reduction の起動。
- `src/plugin.rs`、`src/resident.rs`: Nu コマンド、世代付き ID、metadata、drop と GC。
- `src/convert.rs`: 多次元 list・table と ndarray の変換、shape と出力容量の検査。
- `tests/*.nu`、`tests/unit/`: Nu の ND/table、device、複製・並列参照・GC・再起動と資源寿命の検証。
- `tests/transfers.rs`: N 次元配列の形状・特殊値・精度条件・device 境界の実機検証。
- `validation/src/`: 共通転送 API を使った CubeK 行列積の検証。
- `validation/tests/`: 数値、転送、寿命、エラーの実機検証。
- `validation/benches/nu-matrix.nu`: Nu 標準 matrix と GPU plugin の実コマンド比較。
- `validation/benches/nu-elementwise.nu`: 生成・形状変更・要素演算と混合 chain の実コマンド比較。
- `validation/benches/nu-reductions.nu`、`reductions.rs`: 集約と行列積・要素演算を含む chain の Nu 比較、集約の Criterion 計測。
- `validation/benches/baseline.rs`: GPU 内部経路の Criterion 計測。
- `backend/`: 本体専用の Vulkan バックエンド。出所と変更点は [バックエンドの説明](../backend/README.md)を参照。

Rust の N 次元転送には `gpu::upload_view`、`gpu::upload_array`、
`gpu::download_array` を使う。dtype は f32/f64 に限定し、device の精度条件を
転送前に検査する。所有する連続配列はホスト buffer の所有権を渡し、
非連続・負ストライドの配列は要素順に pack する。使い方は各 API の rustdoc を参照。

## 検証と開発

Rust 1.96.1、Vulkan compute を使える GPU、Nu 0.116.1、GNU Make、SPIRV-Tools の `spirv-val` が必要。
`VM_GPU` は `d0` / `d1` が離散 GPU の番号、`i0` が内蔵 GPU の番号。未指定は `d0`。
実際に選ばれた GPU 名を実行時に表示する。

Nushell:

```nu
$env.VM_GPU = 'd0'
make verify
```

POSIX shell:

```sh
VM_GPU=d0 make verify
```

全ゲートは `make ci`、計測は `make bench`（Nushell / POSIX shell 共通）。
GPU のない GitHub Actions では `make ci-host` でビルド・静的検査を行い、
GPU テストは実機の `make ci` で実行する。公開手順は [リリース](release.md)を参照。

インストールと `plugin add` 後、Nu 接続の回帰テストを実行する（Nushell / POSIX shell 共通）。

```nu
nu tests/layouts-elementwise.nu
nu tests/reductions.nu --device d0
```
`make bench-reductions` は全体集約の scalar 読み戻し、軸集約の metadata／全部読み戻し、
入力転送込みを分ける。Nu の GPU 値登録は出力 buffer の完了を待つ。
`cargo bench --manifest-path validation/Cargo.toml --locked --bench reductions` は
f32/f64 の CPU、常駐演算＋完了待ち、結果転送、入力転送込みを比較する。

`make bench` は実際の Nu `matrix multiply` と `vulkan matrix multiply` を比較する。
両方 f64、同じ入力、64/256/1024角、warmup 2回・計測10回で、結果の一致を検査してから
平均・標準偏差・各回の時間を出す。次の境界を分けて測る。

- `compute_only`: 両入力を準備済み。結果をnative matrix/GPU値のまま返すまで。
  入力転送と結果の読み戻しは計時外。GPU側は結果の登録前に完了を待つ。
  `--multall` 1呼出しで積を1/2/4/8/16回つなぎ、回数を増やした時間の増分も比較する。
- `prepared_readback`: 準備済み入力の積から、結果をNu listに取得するまで。
- `end_to_end`: 同じNu listから両入力を変換し、積と結果取得まで。
- `chain3_readback`: 準備済み3入力で積を2回行い、最後の結果取得まで。

入力生成・Nu起動・結果の一致チェックは計時外。計算のみのケースも、正しさの確認時だけ
結果を読み戻す。長いchainのoverflowや丸め差を避けるため、最初の非定数の積の後は
密な反射行列 `R = I - (2/n)J`（Jは全要素1、R²=I）を掛ける。
両方とも一般の密行列積APIを使い、反射の専用処理はしない。
Nuコマンドの呼出し・結果値の生成コストは計時内で、GPU kernel単体の時間ではない。

Make 経由では演算スレッド数を1に固定する。通常の環境設定で比較する場合は、plugin登録済みの
Nuで `nu validation/benches/nu-matrix.nu --device d0` を実行する。
スレッド設定は結果に記録する。Nu標準benchの端末制御コードを除いて保存するには
`nu validation/benches/nu-matrix.nu --device d0 | ansi strip | save -f nu-matrix.nuon` を使う。
`--sizes 64,256,1024`、`--rounds 10`、`--warmup 2`、
`--chain-lengths 1,2,4,8,16`（積の回数）で条件を指定できる。サイズは正の2の冪に限定する。
隔離plugin registryを使う場合は `make bench NU_ARGS="--config-home <dir> --plugin-config <file>"`。

`make bench-gpu` は GPU 内部経路の Criterion 計測で、Nuコマンド比較とは別の補助資料。
`make bench-elements` は新しい生成・形状変更・要素演算を、同じサイズ・dtype・warmup・計測回数で native Nu と比較する。
計算のみ・最終 readback 付き・入力変換から最終 readback までを分け、reshape→transpose→reshape→scale→add→subtract→multiply の chain も測る。
生成には入力変換がないため、計算のみと readback 付きの2境界を測る。
同一NPYファイルの読み込みからCPU／GPU計算完了までの比較は、Nuで次のように実行できる。
CPU入力はRustのcodecでMessagePackへ復号し、`from msgpack | into matrix` でnative matrixへ渡す。
Pythonは不要。CPUのcodec起動・復号・Nu変換、GPUのimport・転送・演算完了待ちを含める。
`computed` と、結果をNuへ取り出す `nu_readback` を別に測り、計測外で全要素を照合する。
データは同一のC順序f64 NPY、warm page cache、scaleと行列積、3 GPU、warmup2・測定10回。

```nu
cargo build --manifest-path validation/Cargo.toml --release --example npy-codec
with-env {RAYON_NUM_THREADS: '1', MATMUL_NUM_THREADS: '1'} {
    nu validation/benches/file-compute.nu --output ./file-compute-results
}
```

出力先は新しいディレクトリを指定する。codecの場所は `--codec` で指定でき、デフォルトは
`~/.cargo/target/release/examples/npy-codec`。codecは2次元C順序f64専用の検証補助ツールで、
`decode PATH` がMessagePackの行リストを標準出力へ返し、`encode PATH` が同形式を標準入力から読む。
encodeは既存ファイルを上書きしない。NuのCPU行列は `matrix into-nu | to msgpack` で渡せる。

ndarray を直接呼ぶCPUベンチは置かない。数値の正しさの参照値には引き続き ndarray を使う。
Cargo target はデフォルトで `~/.cargo/target` を使う。

公開版のままでは f32 の非正規化数が GPU 演算でゼロになったため、使用する f32/f64 型に
`DenormPreserve` を指定する compiler の変更を加えた。元の比較テストは変更していない。
Nu の取込みと Rust の ND 転送 API は、選択した dtype の保持能力を検査してから転送する。
借用・所有の転送計測は `validation` で `cargo bench --locked --bench transfers` を使う。
所有する入力の準備は計測外とし、upload と GPU 完了待ちを測る。readback は計測前の値検証だけで行う。

## NPY 入出力の検証

公開コマンドの動作確認は `nu tests/file-io.nu --device d0`（`d1` / `i0` も指定可）。
比較測定は `nu validation/benches/file-io.nu`。3 GPU・64/256/1024角・f64・warm cacheで、
Nu値からの入力、ファイル入力、共有入力、同期込みファイル出力を各10回測る（warmup 2回）。
結果はCSVで、正しさの確認は計時外に行う。
