# GPU 上で演算をつなげる

生成・形状変更・要素演算・行列積・集約を、GPU 上の値に対して実行できます。
各例は Nushell 向けです。[導入手順](../README.ja.md#インストール)を済ませてから実行してください。

## Nu の行列と受け渡す

device は一覧の `d0`・`d1`（離散 GPU）、`i0`（内蔵 GPU）などを明示する。
デフォルト dtype は f64。f32 は `--dtype f32` で指定する。

```nu
let source = [[1 2] [3 4]] | into matrix
let gpu = $source | matrix into-nu | vulkan matrix from-nu --device d0
$gpu         # shape・dtype・device のみを表示する
$gpu.shape   # [2, 2]
$gpu | vulkan matrix into-nu | into matrix
```

native matrix との間は `matrix into-nu` と `into matrix` を使う。

## 行列積と内積

1D/2D の dot は、同じ device・dtype の GPU 値どうしを掛ける。
入力を保持し、配列の結果も GPU に保持する。取得するときに `into-nu` を使う。
ベクトルどうしの内積は Nu float を返す。

```nu
let lhs = [[1 2] [3 4]] | vulkan matrix from-nu --device d0
let rhs = [[5 6] [7 8]] | vulkan matrix from-nu --device d0
$lhs | vulkan matrix multiply $rhs | vulkan matrix into-nu
$lhs | vulkan matrix multiply --swap --multall $rhs $lhs | vulkan matrix into-nu
let vector = [1 2] | vulkan matrix from-nu --device d0
$vector | vulkan matrix multiply $vector
```

`--swap` は最初の2入力を交換し、`--multall` は残りを左から順に掛ける。

## 生成・形状変更・要素演算

GPU 上で生成し、転送を挟まずに形状変更と要素演算をつなげる。
`+ - * /` は要素演算で、行列積は `multiply` を使う。

```nu
let a = vulkan matrix identity 3 --device d0
let b = vulkan matrix zeros 3 3 --device d0 | vulkan matrix add 2
$a | vulkan matrix scale 2 | vulkan matrix add $b | vulkan matrix into-nu
($a * $b) | vulkan matrix into-nu
let row = [1 2 3] | vulkan matrix from-nu --device d0
$b | vulkan matrix subtract --broadcast $row | vulkan matrix into-nu
$a | vulkan matrix reshape 9 1 | vulkan matrix transpose
   | vulkan matrix reshape 3 3 | vulkan matrix multiply $a | vulkan matrix into-nu
```

`transpose` は全軸を反転し、buffer を共有する。
`reshape` と `--flatten` は native matrix と同様に行優先の連続配置だけを受け付ける。
通常の転置後は非連続になるため、reshape はエラーになる。
要素演算は同じ dtype・device を要求し、`--broadcast` は右辺だけを左辺の shape に合わせる。
除算の丸めは Vulkan 実装に依存し、CPU とビット一致しない場合がある。

## 集約

`sum` と `max` は全体集約と `--axis` に対応し、`mean` は全体集約に対応する。
全体集約は Nu float を返すため、完了を待って scalar を読み戻す。
`--axis` はその軸を取り除いた GPU 値を返し、途中の読み戻しなしで次の演算につなげられる。

```nu
let a = [[1 2] [3 4]] | vulkan matrix from-nu --device d0
$a | vulkan matrix sum                         # 10.0
$a | vulkan matrix max --axis 0 | vulkan matrix into-nu  # [3.0 4.0]
$a | vulkan matrix mean                        # 2.5
$a | vulkan matrix multiply $a | vulkan matrix scale 2
   | vulkan matrix sum --axis 1 | vulkan matrix into-nu  # [34.0 74.0]
```

集約の入力・蓄積・GPU 出力は同じ dtype。f32 を f64 に昇格して蓄積することはない。
空の集約は `sum` が 0、`max` が負の無限大になり、空配列の `mean` はエラーになる。
`max` は native matrix と同様に NaN を無視し、全要素が NaN なら負の無限大を返す。
`sum` と `mean` は NaN を伝播する。無限大や丸めは浮動小数点演算に従う。
並列加算の順序と除算の丸めが異なるため、native matrix とのビット一致は保証しない。
`--axis` は 0 以上で入力の次元数未満に限る。scalar に対する軸指定はエラーになる。

## 入出力の形状と値の寿命

多次元 list、数値 scalar、2 次元 table を受け取る。table は先頭行の列順を使い、各行の列名を検査する。
`into-nu --as-records` は 2 次元に限り `col0`、`col1` などの table を返す。

```nu
let nd = [[[1 2] [3 4]] [[5 6] [7 8]]]
let gpu_nd = $nd | vulkan matrix from-nu --device d0 --dtype f32
$gpu_nd | vulkan matrix into-nu
let empty = [] | vulkan matrix from-nu --device d0 --shape [2 0 3]
$empty | vulkan matrix into-nu | vulkan matrix from-nu --device d0 --shape $empty.shape
```

`--shape` は要素数を保って reshape する。空軸より後ろの寸法は list だけでは表せないため、再取込み時に元の shape を指定する。
変数を複製しても GPU データは複製しない。最後の参照がなくなると資源を解放し、通常の plugin GC に戻る。
`plugin stop vulkan_matrix` や再起動後の旧値は利用できず、元データの再転送が必要。
