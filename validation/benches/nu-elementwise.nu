plugin use vulkan_matrix
use comparison.nu [compare export-native export-gpu]

def native-chain [a: any, b: any, identity: any, n: int] {
    $a | matrix reshape ($n * $n) 1 | matrix transpose | matrix reshape $n $n
    | matrix scale 2 | matrix add $b | matrix subtract $b | matrix multiply $identity
}
def gpu-chain [a: any, b: any, identity: any, n: int] {
    $a | vulkan matrix reshape ($n * $n) 1 | vulkan matrix transpose | vulkan matrix reshape $n $n
    | vulkan matrix scale 2 | vulkan matrix add $b | vulkan matrix subtract $b | vulkan matrix multiply $identity
}

def main [
    --device: string
    --sizes: string = '64,256,1024'
    --rounds: int = 10
    --warmup: int = 2
] {
    let device = $device | default ($env.VM_GPU? | default 'd0')
    let sizes = $sizes | split row ',' | each { into int }
    if ($sizes | any {|n| $n < 1}) { error make {msg: 'sizes must be positive'} }
    let results = $sizes | each {|n|
        # Dense, dyadic values make all comparisons exact, including the mixed chain.
        let a = export-native (matrix identity $n | matrix add 1)
        let b = export-native (matrix identity $n | matrix scale 2 | matrix add 2)
        let row = export-native (matrix zeros 1 $n | matrix add 1)
        let ma = $a | into matrix
        let mb = $b | into matrix
        let mr = $row | into matrix
        let mi = matrix identity $n
        let ga = $a | vulkan matrix from-nu --device $device
        let gb = $b | vulkan matrix from-nu --device $device
        let gr = $row | vulkan matrix from-nu --device $device
        let gi = vulkan matrix identity $n --device $device
        let cases = [
            {operation: zeros, native: { matrix zeros $n $n }, gpu: { vulkan matrix zeros $n $n --device $device }}
            {operation: identity, native: { matrix identity $n }, gpu: { vulkan matrix identity $n --device $device }}
            {operation: transpose, native: { $ma | matrix transpose }, gpu: { $ga | vulkan matrix transpose }, end_native: { $a | into matrix | matrix transpose }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix transpose }}
            {operation: reshape, native: { $ma | matrix reshape ($n * $n) 1 }, gpu: { $ga | vulkan matrix reshape ($n * $n) 1 }, end_native: { $a | into matrix | matrix reshape ($n * $n) 1 }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix reshape ($n * $n) 1 }}
            {operation: add, native: { $ma | matrix add $mb }, gpu: { $ga | vulkan matrix add $gb }, end_native: { $a | into matrix | matrix add ($b | into matrix) }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix add ($b | vulkan matrix from-nu --device $device) }}
            {operation: subtract, native: { $ma | matrix subtract $mb }, gpu: { $ga | vulkan matrix subtract $gb }, end_native: { $a | into matrix | matrix subtract ($b | into matrix) }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix subtract ($b | vulkan matrix from-nu --device $device) }}
            {operation: scale, native: { $ma | matrix scale 2 }, gpu: { $ga | vulkan matrix scale 2 }, end_native: { $a | into matrix | matrix scale 2 }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix scale 2 }}
            {operation: element_multiply, native: { $ma * $mb }, gpu: { $ga * $gb }, end_native: { ($a | into matrix) * ($b | into matrix) }, end_gpu: { ($a | vulkan matrix from-nu --device $device) * ($b | vulkan matrix from-nu --device $device) }}
            {operation: element_divide, native: { $ma / 2 }, gpu: { $ga / 2 }, end_native: { ($a | into matrix) / 2 }, end_gpu: { ($a | vulkan matrix from-nu --device $device) / 2 }}
            {operation: broadcast_add, native: { $ma | matrix add --broadcast $mr }, gpu: { $ga | vulkan matrix add --broadcast $gr }, end_native: { $a | into matrix | matrix add --broadcast ($row | into matrix) }, end_gpu: { $a | vulkan matrix from-nu --device $device | vulkan matrix add --broadcast ($row | vulkan matrix from-nu --device $device) }}
            {operation: mixed_chain, native: { native-chain $ma $mb $mi $n }, gpu: { gpu-chain $ga $gb $gi $n }, end_native: { native-chain ($a | into matrix) ($b | into matrix) (matrix identity $n) $n }, end_gpu: { gpu-chain ($a | vulkan matrix from-nu --device $device) ($b | vulkan matrix from-nu --device $device) (vulkan matrix identity $n --device $device) $n }}
        ]
        $cases | each {|case|
            let jobs = [
                {mode: compute_only, matrix_output: true, native: $case.native, gpu: $case.gpu}
                {mode: prepared_readback, matrix_output: false, native: { export-native (do $case.native) }, gpu: { export-gpu (do $case.gpu) }}
            ]
            let jobs = if 'end_native' in ($case | columns) {
                $jobs | append {mode: end_to_end, matrix_output: false, native: { export-native (do $case.end_native) }, gpu: { export-gpu (do $case.end_gpu) }}
            } else { $jobs }
            $jobs | each {|job|
                compare $job.native $job.gpu $rounds $warmup $job.matrix_output $case.operation
                | each {|row| $row | merge {size: $n, operation: $case.operation, mode: $job.mode} }
            } | flatten
        } | flatten
    } | flatten
    {
        nu_version: (version | get version)
        source_commit: (^git rev-parse HEAD | str trim)
        cpu: (sys cpu | first | get brand)
        device: $device
        gpu: (vulkan matrix devices | where device == $device)
        dtype: f64
        rounds: $rounds
        warmup_rounds: $warmup
        rayon_num_threads: $env.RAYON_NUM_THREADS?
        matmul_num_threads: $env.MATMUL_NUM_THREADS?
        results: $results
    } | to nuon
}
