plugin use vulkan_matrix

# Public command wall time, warm page cache, f64. Output includes file/directory fsync.
def main [--rounds: int = 10] {
    if $rounds < 1 { error make {msg: 'rounds must be positive'} }
    let dir = (($env.TMPDIR? | default '/tmp') | path join $'vulkan-npy-bench-(random uuid)')
    mkdir $dir
    let results = try {
        mut rows = []
        for device in [d0 d1 i0] {
            for n in [64 256 1024] {
                let file = ($dir | path join $'($device)-($n).npy')
                let source = vulkan matrix identity $n --device $device
                let computed = $source | vulkan matrix scale 2
                let expected = $computed | vulkan matrix into-nu
                $computed | vulkan matrix to-npy $file
                for mode in [nu-upload copy shared output] {
                    for round in 0..($rounds + 1) {
                        let elapsed = timeit {
                            match $mode {
                                nu-upload => { $expected | vulkan matrix from-nu --device $device | ignore }
                                copy => { vulkan matrix from-npy $file --device $device | ignore }
                                shared => { vulkan matrix from-npy $file --device $device --shared | ignore }
                                output => { $computed | vulkan matrix to-npy $file --force }
                            }
                        }
                        if $round >= 2 {
                            $rows = ($rows | append {device: $device, n: $n, mode: $mode, round: ($round - 2), ns: ($elapsed | into int)})
                        }
                    }
                    let actual = if $mode == nu-upload {
                        $expected | vulkan matrix from-nu --device $device | vulkan matrix into-nu
                    } else if $mode == shared {
                        vulkan matrix from-npy $file --device $device --shared | vulkan matrix into-nu
                    } else {
                        vulkan matrix from-npy $file --device $device | vulkan matrix into-nu
                    }
                    if $actual != $expected { error make {msg: $'incorrect payload: ($device)/($n)/($mode)'} }
                }
            }
        }
        $rows
    } catch {|err|
        rm -rf $dir
        error make $err
    }
    rm -rf $dir
    $results | to csv
}
