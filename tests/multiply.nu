plugin use vulkan_matrix

def check [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}

def export-native [value: any]: nothing -> any { $value | matrix into-nu }
def multiply-native [a: any, b: any]: nothing -> any { $a | matrix multiply $b }
def import-native-array [value: any]: nothing -> any {
    let vector = (($value | describe) | str starts-with list) and (($value | is-empty) or (($value | get 0 | describe) in [int float]))
    let source = if $vector { [$value] } else { $value }
    let matrix = $source | into matrix
    if $vector { $matrix | matrix reshape --flatten } else { $matrix }
}
def import-native [value: any, rows: int, cols: int]: nothing -> any {
    if $rows == 0 {
        0..<$cols | each { [] } | into matrix | matrix transpose
    } else { import-native-array $value }
}

let device = $env.VM_GPU? | default d0
let a = [[1 2 3] [4 5 6]] | vulkan matrix from-nu --device $device
let b = [[1 2] [3 4] [5 6]] | vulkan matrix from-nu --device $device
let result = $a | vulkan matrix multiply $b
check ($result.shape == [2 2]) '2D product shape'
check ($result.dtype == f64) '2D product dtype'
check ($result.device == $device) '2D product device'
check (($result | vulkan matrix into-nu) == [[22.0 28.0] [49.0 64.0]]) '2D exact product'
check (($a | vulkan matrix into-nu) == [[1.0 2.0 3.0] [4.0 5.0 6.0]]) 'left input unchanged'
check (($b | vulkan matrix into-nu) == [[1.0 2.0] [3.0 4.0] [5.0 6.0]]) 'right input unchanged'

for dtype in [f64 f32] {
    for dims in [[1 1 1] [31 33 65] [0 3 2] [2 0 3] [2 3 0]] {
        let m = $dims.0
        let k = $dims.1
        let n = $dims.2
        let ha = 0..<$m | each {|i| 0..<$k | each {|j| (($i * 7 + $j) mod 5) - 2 } }
        let hb = 0..<$k | each {|i| 0..<$n | each {|j| (($i + $j * 3) mod 5) - 2 } }
        let na = import-native $ha $m $k
        let nb = import-native $hb $k $n
        let ga = $ha | vulkan matrix from-nu --device $device --dtype $dtype --shape [$m $k]
        let gb = $hb | vulkan matrix from-nu --device $device --dtype $dtype --shape [$k $n]
        let got = $ga | vulkan matrix multiply $gb
        let expected = export-native ($na | matrix multiply $nb)
        check ($got.shape == [$m $n]) $'product shape ($dims) ($dtype)'
        check ((($got | vulkan matrix into-nu) | to msgpack) == ($expected | to msgpack)) $'native product ($dims) ($dtype)'
    }
}
let transposed = [[1 2 3] [4 5 6]] | into matrix | matrix transpose
let gt = $transposed | matrix into-nu | vulkan matrix from-nu --device $device
check ((($gt | vulkan matrix multiply $a | vulkan matrix into-nu) | to msgpack) == (($transposed | matrix multiply ([[1 2 3] [4 5 6]] | into matrix) | matrix into-nu) | to msgpack)) 'native transpose input'
let branches = 1..4 | par-each { $a | vulkan matrix multiply $b | vulkan matrix into-nu }
check ($branches | all {|v| $v == [[22.0 28.0] [49.0 64.0]]}) 'branched inputs'
check (try { $a | vulkan matrix multiply $a | ignore; false } catch { true }) 'shape mismatch'
check (try { $a | vulkan matrix multiply 1 | ignore; false } catch { true }) 'nonresident argument'
for dtype in [f64 f32] {
    for case in [
        {a: [1 2 3], b: [4 5 6], shape: [], scalar: true}
        {a: [1 2], b: [[1 2 3] [4 5 6]], shape: [3], scalar: false}
        {a: [[1 2] [3 4]], b: [5 6], shape: [2], scalar: false}
        {a: [], b: [], shape: [], scalar: true}
        {a: [[] []], b: [], shape: [2], scalar: false}
    ] {
        let ga = $case.a | vulkan matrix from-nu --device $device --dtype $dtype
        let gb = $case.b | vulkan matrix from-nu --device $device --dtype $dtype
        let got = $ga | vulkan matrix multiply $gb
        let expected_native = multiply-native (import-native-array $case.a) (import-native-array $case.b)
        let expected = if $case.scalar { $expected_native } else { export-native $expected_native | first }
        let read = if $case.scalar { $got } else { $got | vulkan matrix into-nu }
        check (($read | to msgpack) == ($expected | to msgpack)) $'1D product ($dtype) ($case.shape)'
        if $case.scalar { check (($got | describe) == float) 'dot returns Nu float' } else {
            check ($got.shape == $case.shape) 'vector result shape'
        }
    }
}
let x = [[1 2] [3 4]] | vulkan matrix from-nu --device $device
let y = [[0 1] [1 0]] | vulkan matrix from-nu --device $device
let z = [[2 0] [0 3]] | vulkan matrix from-nu --device $device
let nx = [[1 2] [3 4]] | into matrix
let ny = [[0 1] [1 0]] | into matrix
let nz = [[2 0] [0 3]] | into matrix
let chained = $x | vulkan matrix multiply --swap --multall $y $z | vulkan matrix into-nu
check (($chained | to msgpack) == (($nx | matrix multiply --swap --multall $ny $nz | matrix into-nu) | to msgpack)) 'swap first pair then multall'
let ignored = $x | vulkan matrix multiply $y 'ignored' | vulkan matrix into-nu
check (($ignored | to msgpack) == (($nx | matrix multiply $ny 'ignored' | matrix into-nu) | to msgpack)) 'extra argument ignored without multall'
let v = [1 2] | vulkan matrix from-nu --device $device
check (try { $v | vulkan matrix multiply --multall $v $v | ignore; false } catch { true }) 'scalar intermediate rank error'
let plain_chain = $x | vulkan matrix multiply --multall $y $z | vulkan matrix into-nu
check (($plain_chain | to msgpack) == (($nx | matrix multiply --multall $ny $nz | matrix into-nu) | to msgpack)) 'left associative multall'
check (($plain_chain | to msgpack) == (($x | vulkan matrix multiply $y | vulkan matrix multiply $z | vulkan matrix into-nu) | to msgpack)) 'chained resident commands'
let scalar = $v | vulkan matrix multiply --multall $x $v
let native_vector = import-native-array [1 2]
check (($scalar | to msgpack) == (($native_vector | matrix multiply --multall $nx $native_vector) | to msgpack)) 'vector matrix vector chain'
for dtype in [f64 f32] {
    let empty_vector = [] | vulkan matrix from-nu --device $device --dtype $dtype
    let empty_matrix = [] | vulkan matrix from-nu --device $device --dtype $dtype --shape [0 3]
    let zeros = $empty_vector | vulkan matrix multiply $empty_matrix
    check ($zeros.shape == [3]) 'empty vector matrix shape'
    check (($zeros | vulkan matrix into-nu) == [0.0 0.0 0.0]) 'empty vector matrix values'
    for value in [NaN inf -inf] {
        let left = [[$value]] | vulkan matrix from-nu --device $device --dtype $dtype
        let right = [[1.0]] | vulkan matrix from-nu --device $device --dtype $dtype
        let got = $left | vulkan matrix multiply $right | vulkan matrix into-nu | get 0.0
        check (($got | to nuon) == ($value | to nuon)) $'special value ($dtype)'
    }
}
for case in [
    {a: 1, b: [1]}
    {a: [[[1]]], b: [1]}
    {a: [1 2], b: [1]}
    {a: [1 2], b: [[1]]}
    {a: [[1 2]], b: [1]}
    {a: [[1]], b: [[1] [2]]}
] {
    let ga = $case.a | vulkan matrix from-nu --device $device
    let gb = $case.b | vulkan matrix from-nu --device $device
    let native_fails = try { multiply-native (import-native-array $case.a) (import-native-array $case.b) | ignore; false } catch { true }
    let gpu_fails = try { $ga | vulkan matrix multiply $gb | ignore; false } catch { true }
    check ($native_fails and $gpu_fails) 'native rank and dimension failures'
}
let f32_input = [[1 2] [3 4]] | vulkan matrix from-nu --device $device --dtype f32
check (try { $f32_input | vulkan matrix multiply $x | ignore; false } catch { true }) 'dtype mismatch'
let other_device = if $device == d1 { 'd0' } else { 'd1' }
let foreign = [[1 2] [3 4]] | vulkan matrix from-nu --device $other_device
check (try { $x | vulkan matrix multiply $foreign | ignore; false } catch { true }) 'device mismatch'
print 'multiply checks passed'
