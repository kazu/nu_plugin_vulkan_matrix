# nu_plugin_vulkan_matrix

[Nushell][nushell] の [matrix][matrix] に近いインターフェースで、Vulkan を使った GPU 行列演算を行う Nushell プラグイン。

## 主な特徴

- Nu の `matrix` に近い操作で、行列積・要素演算・形状変更・集約などを GPU で実行できます。
- 外部 GPU では概ね 512×512 以上の行列から効果が出始めます。内蔵 GPU では、より小さい行列でも速くなる場合があります。
- データを GPU 上に保持したまま演算をつなげられます。メモリと VRAM の転送負担が計算の短縮分を上回る場合は、CPU だけで処理する方が速くなります。
- Linux kernel の page cache から VRAM への入力と、VRAM から page cache への出力に対応します。ファイル形式は NumPy の NPY です。
- 内蔵 GPU のような共有メモリ環境では、`--shared` で入力コピーを省けます。AMD Radeon 780M（Ryzen 9 8945HS 内蔵）では NPY 読込＋行列積が **72.1 → 44.0 ms** に短縮しました。

f64 の行列積の例（NPY ファイルからの読込を含む、10回平均、単位 ms）：

| 行列サイズ | 処理範囲 | CPU（Nu matrix） | RX 7800 XT | 内蔵GPU Radeon 780M | Radeon 780M 共有入力 |
|---|---|---:|---:|---:|---:|
| 256×256 | ファイル読込＋行列積 | 18.1 | 39.8 | 8.7 | 5.3 |
| 512×512 | ファイル読込＋行列積 | 66.7 | 40.6 | 17.6 | 14.1 |
| 1024×1024 | ファイル読込＋行列積 | 255.7 | 53.0 | 72.1 | 44.0 |
| 256×256 | 上記＋Nu に書き出す | 20.4 | 60.8 | 27.3 | 23.7 |
| 512×512 | 上記＋Nu に書き出す | 76.8 | 102.4 | 90.6 | 89.2 |
| 1024×1024 | 上記＋Nu に書き出す | 301.5 | 348.5 | 363.3 | 352.5 |

測定条件と全 GPU の結果は[ベンチマーク][benchmarks]、共有入力の条件は[NPY 入出力][file-io]を参照してください。

## インストール

Nu 0.116.1、Rust 1.96.1（`rust-toolchain.toml`）、対象 dtype の非正規化数を保持できる
Vulkan GPU が必要です。バックエンドは本体に同梱されています。

crates.io からインストール（初回公開後、Nushell で実行）：

```nu
cargo install nu_plugin_vulkan_matrix --locked --root $env.HOME
plugin add ~/bin/nu_plugin_vulkan_matrix
plugin use vulkan_matrix
```

GitHub のソースからインストールする場合は、最初のコマンドを次に置き換えます。

```nu
cargo install --git https://github.com/kazu/nu_plugin_vulkan_matrix --locked --root $env.HOME
```

## 使う

利用できる GPU を確認します。

Nushell:

```nu
vulkan matrix devices
```

以下の例の `d0` は、デバイス一覧に表示された GPU に合わせてください。

行列積と要素演算を GPU 上でつなぎ、最後に結果を取り出します。デフォルトは f64 です。

Nushell:

```nu
let a = [[1 2] [3 4]] | vulkan matrix from-nu --device d0
let b = [[5 6] [7 8]] | vulkan matrix from-nu --device d0
let result = $a | vulkan matrix multiply $b | vulkan matrix scale 2
$result | vulkan matrix into-nu
```

`$result` は GPU 上の行列です。`into-nu` で全要素を取り出します。
`+ - * /` は要素演算、`multiply` は行列積です。
Nu の `matrix` との受け渡しや集約などは[演算の使い方][operations]を参照してください。

Linux では、演算結果を NumPy の NPY ファイルに保存し、再び GPU へ読み込めます。

Nushell（上の例に続けて実行）:

```nu
$result | vulkan matrix to-npy ./result.npy
let restored = vulkan matrix from-npy ./result.npy --device d0
$restored | vulkan matrix into-nu
```

保存先に同名のファイルがある場合はエラーになります。
対応形式・共有入力・ドライバーの条件は[NPY 入出力][file-io]を参照してください。

## 開発

全ゲートは `make ci` で実行します（Nushell / POSIX shell 共通、GNU Make が必要）。
必要な環境、ソースの構成、各検証・計測の手順は[開発と検証][development]にまとめています。

## 構成

- `src/`: プラグインと GPU 演算。
- `tests/`: Rust / Nushell のテスト。
- `validation/`: 実機検証とベンチマーク。
- `backend/`: 本体専用の Vulkan バックエンド。[出所と変更点][backend]。
- `docs/`: 操作・ファイル入出力・開発の詳細。

## 関連

- [Nushell][nushell]：構造化データを扱うシェル。
- [matrix][matrix]：Nushell 標準の行列演算。
- [polars-dyn][polars-dyn]：DataFrame と GPU 行列演算の連携を強化予定。

## ライセンス

[MIT][license] © 2026 Kazuhisa TAKEI。CubeCL 由来のバックエンドには元の著作権表示とライセンスを保持しています。

[nushell]: https://www.nushell.sh/
[matrix]: https://www.nushell.sh/commands/docs/matrix.html
[polars-dyn]: https://github.com/kazu/nu_plugin_polars_dyn
[benchmarks]: docs/benchmarks.md
[file-io]: docs/file-io.md
[operations]: docs/operations.md
[development]: docs/development.md
[backend]: backend/README.md
[license]: LICENSE
