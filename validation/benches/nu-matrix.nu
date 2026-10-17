plugin use vulkan_matrix
use comparison.nu [compare export-native export-gpu]

# Run in an isolated Nu config home with an explicitly registered plugin.
# Inputs match the Rust GPU profiling benchmark; native matrix uses f64.
def main [
    --device: string
    --sizes: string = '64,256,1024'
    --rounds: int = 10
    --warmup: int = 2
    # Number of multiplications, not number of input matrices.
    --chain-lengths: string = '1,2,4,8,16'
] {
    let device = $device | default ($env.VM_GPU? | default 'd0')
    let sizes = $sizes | split row ',' | each { into int }
    let lengths = $chain_lengths | split row ',' | each { into int }
    if $rounds < 2 or $warmup < 1 or ($sizes | any {|n| $n < 1 or (($n | bits and ($n - 1)) != 0)}) or ($lengths | any {|n| $n < 1}) {
        error make {msg: 'Use power-of-two sizes, positive chain lengths, rounds >= 2 and warmup >= 1'}
    }
    let results = $sizes | each {|n|
        let a = 0..<$n | each {|i| 0..<$n | each {|j| (($i * $n + $j) mod 17) / 16.0 } }
        let b = 0..<$n | each {|i| 0..<$n | each {|j| (($i * 7 + $j * 11) mod 17) / 16.0 } }
        # Dense reflection R = I - (2/n)J satisfies R*R = I.
        # Power-of-two sizes keep coefficients dyadic and the chain bounded.
        let reflection = 0..<$n | each {|i| 0..<$n | each {|j| (if $i == $j { 1.0 } else { 0.0 }) - (2.0 / $n) } }
        let ma = $a | into matrix
        let mb = $b | into matrix
        let mr = $reflection | into matrix
        let ga = $a | vulkan matrix from-nu --device $device --dtype f64
        let gb = $b | vulkan matrix from-nu --device $device --dtype f64
        let gr = $reflection | vulkan matrix from-nu --device $device --dtype f64
        let cases = [
            {
                mode: prepared_readback
                multiplications: 1
                native: { export-native ($ma | matrix multiply $mb) }
                gpu: { export-gpu ($ga | vulkan matrix multiply $gb) }
            }
            {
                mode: end_to_end
                multiplications: 1
                native: { export-native ($a | into matrix | matrix multiply ($b | into matrix)) }
                gpu: { export-gpu ($a | vulkan matrix from-nu --device $device --dtype f64 | vulkan matrix multiply ($b | vulkan matrix from-nu --device $device --dtype f64)) }
            }
            {
                mode: chain3_readback
                multiplications: 2
                native: { export-native ($ma | matrix multiply --multall $mb $mr) }
                gpu: { export-gpu ($ga | vulkan matrix multiply --multall $gb $gr) }
            }
        ]
        let chains = $lengths | each {|length|
            let native_tail = 1..<$length | each { $mr }
            let gpu_tail = 1..<$length | each { $gr }
            {
                mode: compute_only
                multiplications: $length
                matrix_output: true
                native: { $ma | matrix multiply --multall $mb ...$native_tail }
                gpu: { $ga | vulkan matrix multiply --multall $gb ...$gpu_tail }
            }
        }
        $cases | append $chains | each {|case|
            compare $case.native $case.gpu $rounds $warmup ($case.matrix_output? | default false)
            | each {|row| $row | merge {size: $n, mode: $case.mode, multiplications: $case.multiplications} }
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
        chain_multiplications: $lengths
        rayon_num_threads: $env.RAYON_NUM_THREADS?
        matmul_num_threads: $env.MATMUL_NUM_THREADS?
        results: $results
    } | to nuon
}
