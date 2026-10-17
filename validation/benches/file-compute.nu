plugin use vulkan_matrix

def cpu-load [codec: path, file: path]: nothing -> any {
    ^$codec decode $file | from msgpack | into matrix
}

def cpu-compute [codec: path, a: path, b: path, op: string]: nothing -> any {
    let x = cpu-load $codec $a
    if $op == 'scale' { $x | matrix scale 2 } else {
        $x | matrix multiply (cpu-load $codec $b)
    }
}

def gpu-compute [a: path, b: path, op: string, device: string, shared: bool]: nothing -> any {
    let x = vulkan matrix from-npy $a --device $device --shared=$shared
    if $op == 'scale' { $x | vulkan matrix scale 2 } else {
        let y = vulkan matrix from-npy $b --device $device --shared=$shared
        $x | vulkan matrix multiply $y
    }
}

def native-rows [value: any]: nothing -> any { $value | matrix into-nu }
def gpu-rows [value: any]: nothing -> any { $value | vulkan matrix into-nu }

# Warm page cache; timings include process startup/codec/Nu conversion for CPU,
# file import/transfer and synchronized GPU compute for GPU. All use f64.
def main [
    --output: path
    --codec: path = '~/.cargo/target/release/examples/npy-codec'
    --sizes: string = '64,256,1024'
    --rounds: int = 10
    --warmup: int = 2
] {
    if $rounds < 1 or $warmup < 1 { error make {msg: 'rounds and warmup must be positive'} }
    mkdir $output
    let csv = $output | path join 'file-compute.csv'
    if ($csv | path exists) { error make {msg: 'output CSV already exists'} }
    let codec = $codec | path expand
    let modes = [{mode: cpu, device: cpu, shared: false}]
        | append (['d0' 'd1' 'i0'] | each {|device|
            [{mode: gpu_copy, device: $device, shared: false}
             {mode: gpu_shared, device: $device, shared: true}]
        } | flatten)
    mut first_row = true
    for n in ($sizes | split row ',' | each { into int }) {
        if $n < 1 { error make {msg: 'sizes must be positive'} }
        let a = $output | path join $'a-($n).npy'
        let b = $output | path join $'b-($n).npy'
        # Integer-valued dense inputs make all dot products exactly representable.
        for spec in [{path: $a, factor: 3} {path: $b, factor: 5}] {
            let rows = 0..<$n | each {|i|
                0..<$n | each {|j| ((($i * $spec.factor + $j) mod 7) - 3) | into float }
            }
            $rows | vulkan matrix from-nu --device d0 --dtype f64
                | vulkan matrix to-npy $spec.path
            if ((native-rows (cpu-load $codec $spec.path)) != $rows) {
                error make {msg: 'NPY decoding differs from generated input'}
            }
        }
        for op in ['scale' 'multiply'] {
            let expected = native-rows (cpu-compute $codec $a $b $op)
            for mode in $modes {
                if $mode.mode != 'cpu' {
                    let actual = gpu-rows (gpu-compute $a $b $op $mode.device $mode.shared)
                    if $actual != $expected { error make {msg: $'wrong result: ($n) ($op) ($mode)'} }
                }
            }
            for boundary in ['computed' 'nu_readback'] {
                print --stderr $'size=($n) op=($op) boundary=($boundary)'
                for round in 0..<($warmup + $rounds) {
                    # Rotate execution order across rounds to reduce order bias.
                    let shift = $round mod ($modes | length)
                    let ordered = ($modes | skip $shift) | append ($modes | first $shift)
                    for mode in $ordered {
                        let action = if $mode.mode == 'cpu' {
                            {|| let value = cpu-compute $codec $a $b $op
                                if $boundary == 'nu_readback' { native-rows $value } else { $value } }
                        } else {
                            {|| let value = gpu-compute $a $b $op $mode.device $mode.shared
                                if $boundary == 'nu_readback' { gpu-rows $value } else { $value } }
                        }
                        let elapsed = timeit { do $action | ignore }
                        if $round >= $warmup {
                            let row = {size: $n, operation: $op, boundary: $boundary,
                                mode: $mode.mode, device: $mode.device,
                                round: ($round - $warmup), ms: (($elapsed | into int) / 1_000_000)}
                            if $first_row { $row | to csv | save $csv; $first_row = false } else {
                                $row | to csv --noheaders | save --append $csv
                            }
                        }
                    }
                }
            }
        }
    }
    open $csv | group-by size operation boundary mode device --to-table
        | each {|group| {
            size: $group.size, operation: $group.operation, boundary: $group.boundary,
            mode: $group.mode, device: $group.device, samples: ($group.items | length),
            mean_ms: ($group.items.ms | math avg),
            min_ms: ($group.items.ms | math min), max_ms: ($group.items.ms | math max)
        }} | save ($output | path join 'summary.nuon')
}
