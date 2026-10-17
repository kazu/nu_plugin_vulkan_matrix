# NumPy ファイルの読み書き（Linux）

NPY ファイルから GPU へ行列を読み込み、演算結果を保存できます。
例は Nushell 向けです。[導入手順](../README.ja.md#インストール)を済ませてから実行してください。

```nu
let a = [[1 2 3] [4 5 6]] | vulkan matrix from-nu --device d0
$a | vulkan matrix scale 2 | vulkan matrix to-npy ./result.npy
let b = vulkan matrix from-npy ./result.npy --device d0
$b | vulkan matrix into-nu

# 内蔵 GPU でファイルのマッピングを行列として保持する
let shared = vulkan matrix from-npy ./result.npy --device i0 --shared
$shared | vulkan matrix scale 3
```

## 対応形式

形式は標準の NumPy `.npy`。C-order（行優先）、little-endian の `<f4` / `<f8`、
空でない 1 次元または 2 次元配列を扱う。入力は NPY v1/v2/v3、ヘッダーは最大 64 KiB。
出力は NPY v2 で、shape と dtype をヘッダーに保存するため、別途指定する必要はない。
標準 NPY ヘッダーの空白パディングで GPU の import alignment に合わせる。
空の先頭ページを要求する独自形式ではない。

## 転送と共有入力

`from-npy` はファイルの page cache を import し、GPU コピーで通常の GPU 行列へ取り込む。
`--shared` はそのコピーを省き、読み取り専用のファイルマッピングを行列の寿命まで保持する。
この場合はデータ開始位置と割当範囲が GPU の条件を満たす必要があり、満たさない入力はエラーになる。
このプラグインの出力は位置を揃えてある。共有中のファイルは変更・切詰めしないこと。

`to-npy` は GPU の行列からファイルの page cache へ Vulkan buffer copy を行い、完了後に
ファイルを同期する。転置された行列は GPU 上で行優先に並べ直す。
ユーザースペースでの payload のメモリコピーや `write` / `splice` は使わず、CPU が書くのはヘッダーだけ。
同じディレクトリの一時ファイルから確定し、既存ファイルは保護する。
置換する場合は `vulkan matrix to-npy ./result.npy --force` を使う。

## 必要な環境

GPU とドライバーに `VK_EXT_external_memory_host` と coherent host import が必要。
AMD の USERPTR 呼出しに対するフラグ調整は plugin executable 内で import 対象の範囲・スレッドに限定して行う。
システムライブラリの変更や `LD_PRELOAD` の指定は不要。
