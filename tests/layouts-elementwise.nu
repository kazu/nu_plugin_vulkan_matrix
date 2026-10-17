plugin use vulkan_matrix

def check [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}
def native-lists [value: any]: nothing -> any { $value | matrix into-nu }
def native [value: any]: nothing -> any {
    let output = native-lists $value
    if ($value.shape | length) == 1 { $output | get 0 } else { $output }
}
def cpu-array [value: any] { $value | into matrix }
def gpu [value: any]: nothing -> any { $value | vulkan matrix into-nu }
def fails [run: closure] { try { do $run | ignore; false } catch { true } }

for dtype in [f64 f32] {
    for dimensions in [[3] [2 3] [2 1 3]] {
        let first = $dimensions | first
        let rest = $dimensions | skip 1
        let a = vulkan matrix zeros $first ...$rest --device d0 --dtype $dtype
        check ($a.shape == $dimensions and $a.dtype == $dtype) 'zeros metadata'
        check ((gpu $a) == (native (matrix zeros $first ...$rest))) 'native zeros'
    }
    for n in [1 3 4] {
        let a = vulkan matrix identity $n --device d0 --dtype $dtype
        check ((gpu $a) == (native (matrix identity $n))) 'native identity'
    }
    let values = [[2.0 4.0 8.0] [16.0 32.0 64.0]]
    let a = $values | vulkan matrix from-nu --device d0 --dtype $dtype
    let n = $values | into matrix
    let b = [[2.0 4.0 2.0] [4.0 2.0 4.0]] | vulkan matrix from-nu --device d0 --dtype $dtype
    let nb = [[2.0 4.0 2.0] [4.0 2.0 4.0]] | into matrix
    check ((gpu ($a | vulkan matrix transpose)) == (native ($n | matrix transpose))) 'transpose'
    check ((gpu ($a | vulkan matrix reshape 3 2)) == (native ($n | matrix reshape 3 2))) 'reshape'
    check ((gpu ($a | vulkan matrix reshape --flatten)) == (native ($n | matrix reshape --flatten))) 'flatten'
    check ((gpu ($a | vulkan matrix add $b)) == (native ($n | matrix add $nb))) 'add matrix'
    check ((gpu ($a | vulkan matrix subtract $b)) == (native ($n | matrix subtract $nb))) 'subtract matrix'
    check ((gpu ($a | vulkan matrix add 2)) == (native ($n | matrix add 2))) 'add scalar'
    check ((gpu ($a | vulkan matrix subtract 2)) == (native ($n | matrix subtract 2))) 'subtract scalar'
    check ((gpu ($a | vulkan matrix scale 2)) == (native ($n | matrix scale 2))) 'scale'
    check ((gpu ($a + $b)) == (native ($n + $nb))) 'operator + matrix'
    check ((gpu ($a - $b)) == (native ($n - $nb))) 'operator - matrix'
    check ((gpu ($a * $b)) == (native ($n * $nb))) 'operator * elementwise'
    check ((gpu ($a / $b)) == (native ($n / $nb))) 'operator / elementwise'
    check ((gpu ($a + 2)) == (native ($n + 2))) 'operator + scalar'
    check ((gpu ($a - 2)) == (native ($n - 2))) 'operator - scalar'
    check ((gpu ($a * 2)) == (native ($n * 2))) 'operator * scalar'
    check ((gpu ($a / 2)) == (native ($n / 2))) 'operator / scalar'
    let row = [1.0 2.0 3.0] | vulkan matrix from-nu --device d0 --dtype $dtype
    let nrow = cpu-array [[1.0 2.0 3.0]] | matrix reshape 3
    check ((gpu ($a | vulkan matrix add --broadcast $row)) == (native ($n | matrix add --broadcast $nrow))) 'RHS broadcast'
    check ((gpu ($a | vulkan matrix subtract --broadcast $row)) == (native ($n | matrix subtract --broadcast $nrow))) 'RHS subtract broadcast'
    check (fails { $row | vulkan matrix add --broadcast $a }) 'broadcast direction'
    check (fails { $a | vulkan matrix add $row }) 'explicit broadcast only'
    check (fails { $a + $row }) 'operator shape mismatch'
    check (fails { $a | vulkan matrix reshape 5 }) 'reshape count'
    check (fails { $a | vulkan matrix reshape 0 6 }) 'reshape dimensions'
    check (fails { $a | vulkan matrix reshape }) 'reshape dimensions missing'
    check ((fails { $a | vulkan matrix transpose | vulkan matrix reshape 6 }) == (fails { $n | matrix transpose | matrix reshape 6 })) 'native nonstandard reshape error'
    let t = $a | vulkan matrix transpose
    let one = vulkan matrix identity 2 --device d0 --dtype $dtype
    check ((gpu ($t | vulkan matrix multiply $one)) == (native ($n | matrix transpose | matrix multiply (matrix identity 2)))) 'transposed multiply'
    let chain = $a | vulkan matrix reshape 6 1 | vulkan matrix transpose | vulkan matrix reshape 2 3 | vulkan matrix multiply (vulkan matrix identity 3 --device d0 --dtype $dtype)
    let expected = $n | matrix reshape 6 1 | matrix transpose | matrix reshape 2 3 | matrix multiply (matrix identity 3)
    check ((gpu $chain) == (native $expected)) 'transpose reshape multiply chain'
    check ($chain.dtype == $dtype) 'chain dtype'
    check ((gpu ($a | vulkan matrix reshape 2 1 3 | vulkan matrix transpose)) == (native ($n | matrix reshape 2 1 3 | matrix transpose))) 'N-D transpose'
    check ((gpu $a) == $values) 'branch input unchanged'
}

let survivor = do {
    let a = [[1.0 2.0] [3.0 4.0]] | vulkan matrix from-nu --device d0
    $a | vulkan matrix transpose
}
check ((gpu $survivor) == [[1.0 3.0] [2.0 4.0]]) 'layout survives source drop'
let a64 = [1.0 2.0] | vulkan matrix from-nu --device d0
let a32 = [1.0 2.0] | vulkan matrix from-nu --device d0 --dtype f32
check (fails { $a64 + $a32 }) 'dtype mismatch'
check (fails { $a64 | vulkan matrix add true }) 'invalid RHS'
check (fails { vulkan matrix zeros 0 --device d0 }) 'zeros dimension'
check (fails { vulkan matrix identity -1 --device d0 }) 'identity dimension'
check (fails { vulkan matrix zeros 2 }) 'required device'
check (fails { vulkan matrix identity 2 --device cpu }) 'no CPU fallback'
check (fails { vulkan matrix zeros 2 --device d0 --dtype f16 }) 'invalid dtype'
let scalar = 3.0 | vulkan matrix from-nu --device d0
check ((gpu ($scalar * 2)) == 6.0) 'scalar input'
let empty = [] | vulkan matrix from-nu --device d0 --shape [2 0 3]
check (($empty | vulkan matrix scale 2).shape == [2 0 3]) 'empty arithmetic shape'
check ((gpu ($empty | vulkan matrix reshape --flatten)) == []) 'empty flatten'
for dtype in [f64 f32] {
    let a = [-0.0 inf -inf NaN] | vulkan matrix from-nu --device d0 --dtype $dtype
    let b = gpu ($a * 1)
    check (($b.0 | to msgpack | encode hex) == CB8000000000000000) 'arithmetic negative zero'
    check ($b.1 == inf and $b.2 == -inf and ($b.3 | into string) == 'NaN') 'arithmetic Inf/NaN'
}
print 'layout and elementwise Nu tests passed'
