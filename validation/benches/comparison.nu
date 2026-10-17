use std/bench *

# Wrappers hide native matrix into-nu's table signature for its actual list output.
export def export-native [value: any]: nothing -> any { $value | matrix into-nu }
export def export-gpu [value: any]: nothing -> any { $value | vulkan matrix into-nu }

export def compare [native: closure, gpu: closure, rounds: int, warmup: int, matrix_output: bool = false, command: string = 'multiply'] {
    # Correctness readback is outside bench, including for compute-only cases.
    let expected = if $matrix_output { export-native (do $native) } else { do $native }
    let actual = if $matrix_output { export-gpu (do $gpu) } else { do $gpu }
    if $actual != $expected { error make {msg: 'native/GPU result mismatch'} }
    bench $native $gpu --rounds $rounds --warmup $warmup --verbose
    | enumerate
    | each {|row|
        {
            command: (if $row.index == 0 { $'matrix ($command)' } else { $'vulkan matrix ($command)' })
            mean_ms: (($row.item.mean | into int) / 1_000_000)
            std_ms: (($row.item.std | into int) / 1_000_000)
            samples_ms: ($row.item.times | each {|t| ($t | into int) / 1_000_000})
        }
    }
}
