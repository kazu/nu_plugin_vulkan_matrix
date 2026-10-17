plugin use vulkan_matrix
use comparison.nu [compare export-native export-gpu]

def main [
    --device: string
    --sizes: string = '64,256,1024'
    --rounds: int = 10
    --warmup: int = 2
] {
    let device = $device | default ($env.VM_GPU? | default 'd0')
    let results = $sizes | split row ',' | each { into int } | each {|n|
        # Exact dyadic results at the default power-of-two sizes.
        let data = export-native (matrix identity $n | matrix add 1)
        let native = $data | into matrix
        let gpu = $data | vulkan matrix from-nu --device $device
        let identity = matrix identity $n
        let gpu_identity = vulkan matrix identity $n --device $device
        let scalar_cases = [
            {operation: sum, native: { $native | matrix sum }, gpu: { $gpu | vulkan matrix sum }, end_native: { $data | into matrix | matrix sum }, end_gpu: { $data | vulkan matrix from-nu --device $device | vulkan matrix sum }}
            {operation: max, native: { $native | matrix max }, gpu: { $gpu | vulkan matrix max }, end_native: { $data | into matrix | matrix max }, end_gpu: { $data | vulkan matrix from-nu --device $device | vulkan matrix max }}
            {operation: mean, native: { $native | matrix mean }, gpu: { $gpu | vulkan matrix mean }, end_native: { $data | into matrix | matrix mean }, end_gpu: { $data | vulkan matrix from-nu --device $device | vulkan matrix mean }}
        ]
        let scalar_rows = $scalar_cases | each {|case|
            [
                {mode: resident_scalar_readback, native: $case.native, gpu: $case.gpu}
                {mode: end_to_end, native: $case.end_native, gpu: $case.end_gpu}
            ] | each {|job|
                compare $job.native $job.gpu $rounds $warmup false $case.operation
                | each {|row| $row | merge {size: $n, operation: $case.operation, mode: $job.mode} }
            } | flatten
        } | flatten
        let axis_rows = [0 1] | each {|axis|
            [
                {operation: $'sum_axis($axis)', native: { $native | matrix sum --axis $axis }, gpu: { $gpu | vulkan matrix sum --axis $axis }, end_native: { $data | into matrix | matrix sum --axis $axis }, end_gpu: { $data | vulkan matrix from-nu --device $device | vulkan matrix sum --axis $axis }}
                {operation: $'max_axis($axis)', native: { $native | matrix max --axis $axis }, gpu: { $gpu | vulkan matrix max --axis $axis }, end_native: { $data | into matrix | matrix max --axis $axis }, end_gpu: { $data | vulkan matrix from-nu --device $device | vulkan matrix max --axis $axis }}
            ] | each {|case|
                # Check values before timing metadata-only access.
                compare { export-native (do $case.native) | get 0 } { export-gpu (do $case.gpu) } 1 0 false $case.operation | ignore
                [
                    {mode: metadata, native: { (do $case.native).shape }, gpu: { (do $case.gpu).shape }}
                    {mode: all_read, native: { export-native (do $case.native) | get 0 }, gpu: { export-gpu (do $case.gpu) }}
                    {mode: end_to_end, native: { export-native (do $case.end_native) | get 0 }, gpu: { export-gpu (do $case.end_gpu) }}
                ] | each {|job|
                    compare $job.native $job.gpu $rounds $warmup false $case.operation
                    | each {|row| $row | merge {size: $n, operation: $case.operation, mode: $job.mode} }
                } | flatten
            } | flatten
        } | flatten
        let chain_rows = [
            {mode: metadata, native: { ($native | matrix multiply $identity | matrix scale 2 | matrix sum --axis 1).shape }, gpu: { ($gpu | vulkan matrix multiply $gpu_identity | vulkan matrix scale 2 | vulkan matrix sum --axis 1).shape }}
            {mode: all_read, native: { export-native ($native | matrix multiply $identity | matrix scale 2 | matrix sum --axis 1) | get 0 }, gpu: { export-gpu ($gpu | vulkan matrix multiply $gpu_identity | vulkan matrix scale 2 | vulkan matrix sum --axis 1) }}
            {mode: end_to_end, native: { export-native ($data | into matrix | matrix multiply $identity | matrix scale 2 | matrix sum --axis 1) | get 0 }, gpu: { export-gpu ($data | vulkan matrix from-nu --device $device | vulkan matrix multiply $gpu_identity | vulkan matrix scale 2 | vulkan matrix sum --axis 1) }}
        ] | each {|job|
            compare $job.native $job.gpu $rounds $warmup false 'multiply_scale_sum'
            | each {|row| $row | merge {size: $n, operation: multiply_scale_sum, mode: $job.mode} }
        } | flatten
        $scalar_rows | append $axis_rows | append $chain_rows
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
        completion: 'GPU registration waits for output buffers; scalar and all_read also download.'
        results: $results
    } | to nuon
}
