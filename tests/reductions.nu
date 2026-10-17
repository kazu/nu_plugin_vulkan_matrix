plugin use vulkan_matrix
$env.config.plugin_gc = {default: {enabled: true, stop_after: 100ms}, plugins: {}}

def check [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}
def export-native [value: any]: nothing -> any { $value | matrix into-nu }
def export-gpu [value: any]: nothing -> any { $value | vulkan matrix into-nu }
def equal [actual: any, expected: any] {
    if ($expected | describe) =~ '^list' {
        if ($actual | length) != ($expected | length) { return false }
        $expected | enumerate | all {|v| equal ($actual | get $v.index) $v.item }
    } else if ($expected | describe) == float and ($expected | into string) in ['NaN' 'inf' '-inf'] {
        ($actual | into string) == ($expected | into string)
    } else { $actual == $expected }
}
def error-labels [run: closure] {
    try { do $run | ignore; [] } catch {|e| $e.details.labels.text }
}
def same-error [native: closure, gpu: closure, cause: string] {
    let expected = error-labels $native
    check (($expected | str join ' ') | str contains $cause) $'native error: ($cause)'
    check ((error-labels $gpu) == $expected) $'GPU error: ($cause)'
}
def close-mean [actual: float, expected: float, dtype: string, data: list] {
    if ($expected | into string) in ['NaN' 'inf' '-inf'] {
        return (equal $actual $expected)
    }
    let values = $data | flatten
    let n = $values | length
    let unit = if $dtype == f32 { 5.960464477539063e-8 } else { 1.1102230246251565e-16 }
    let gamma = $n * $unit / (1.0 - $n * $unit)
    let magnitude = $values | each { math abs } | math sum
    let bound = 2.0 * $gamma * $magnitude / $n + 4.0 * $unit * ($expected | math abs)
    ($actual - $expected | math abs) <= $bound
}

def main [--device: string = 'd0'] {
    do {
    for dtype in [f64 f32] {
        for data in [
            [[1.0 2.0 3.0] [4.0 5.0 6.0]]
            [[NaN NaN NaN] [NaN 2.0 -1.0] [1.0 2.0 3.0] [inf -inf 3.0]]
            [[-inf -inf]]
            []
        ] {
            let native = $data | into matrix
            let gpu = $data | vulkan matrix from-nu --device $device --dtype $dtype --shape $native.shape
            for transpose in [false true] {
                let a = if $transpose { $native | matrix transpose } else { $native }
                let g = if $transpose { $gpu | vulkan matrix transpose } else { $gpu }
                check (equal ($g | vulkan matrix sum) ($a | matrix sum)) 'sum scalar'
                check (equal ($g | vulkan matrix max) ($a | matrix max)) 'max scalar'
                if ($a.shape | any {|n| $n == 0}) {
                    same-error { $a | matrix mean } { $g | vulkan matrix mean } 'cannot compute mean of an empty matrix'
                } else {
                    let actual = $g | vulkan matrix mean
                    let expected = $a | matrix mean
                    check (close-mean $actual $expected $dtype $data) $'mean scalar ($dtype), transpose=($transpose): ($actual) vs ($expected)'
                }
                for axis in [0 1] {
                    let ns = $a | matrix sum --axis $axis
                    let gs = $g | vulkan matrix sum --axis $axis
                    let nm = $a | matrix max --axis $axis
                    let gm = $g | vulkan matrix max --axis $axis
                    check ($gs.shape == $ns.shape and $gm.shape == $nm.shape) 'axis shape'
                    check ($gs.dtype == $dtype and $gs.device == $device) 'axis metadata'
                    check (equal (export-gpu $gs) (export-native $ns | get 0)) 'axis sum'
                    check (equal (export-gpu $gm) (export-native $nm | get 0)) 'axis max'
                }
                for axis in [-1 2] {
                    same-error { $a | matrix sum --axis $axis } { $g | vulkan matrix sum --axis $axis } 'out of bounds for a 2-dimensional array'
                    same-error { $a | matrix max --axis $axis } { $g | vulkan matrix max --axis $axis } 'out of bounds for a 2-dimensional array'
                }
            }
        }
        let vector = [1 2 3] | vulkan matrix from-nu --device $device --dtype $dtype
        let scalar = $vector | vulkan matrix sum --axis 0
        check ($scalar.shape == [] and (export-gpu $scalar) == 6.0) 'axis scalar remains a GPU value'
        check (($scalar | vulkan matrix sum) == 6.0) 'scalar GPU input'
    }
    }
    do {
        let result = do {
            let a = [[1 2] [3 4]] | vulkan matrix from-nu --device $device
            let identity = vulkan matrix identity 2 --device $device
            $a | vulkan matrix multiply $identity | vulkan matrix scale 2 | vulkan matrix sum --axis 1
        }
        let cloned = $result
        sleep 300ms
        check ((plugin list | where name == vulkan_matrix | get 0.pid) != null) 'GC while reduction is held'
        check ((export-gpu $cloned) == [6.0 14.0]) 'chain result survives input drop'
    }
    for attempt in 0..20 {
        if (plugin list | where name == vulkan_matrix | get 0.pid) == null {
            print $'reduction Nu tests passed: ($device)'
            return
        }
        sleep 100ms
    }
    error make {msg: 'GC after reduction final drop'}
}
