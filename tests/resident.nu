plugin use vulkan_matrix

def check [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}

def read-dynamic [value: any] { $value | vulkan matrix into-nu }

def wait-for-stop [] {
    for attempt in 0..20 {
        if (plugin list | where name == vulkan_matrix | get 0.pid) == null { return }
        sleep 100ms
    }
    error make {msg: 'GC after final drop'}
}

$env.config.plugin_gc = {default: {enabled: true, stop_after: 100ms}, plugins: {}}
do {
    let a = [1.0 2.0 3.0] | vulkan matrix from-nu --device d0
    let b = $a
    check ($b.shape == [3]) 'shape metadata'
    check ($b.dtype == 'f64') 'default dtype'
    check ($b.device == 'd0') 'explicit device'
    sleep 300ms
    check ((plugin list | where name == vulkan_matrix | get 0.pid) != null) 'GC while value is held'
    check (($b | vulkan matrix into-nu) == [1.0 2.0 3.0]) 'cloned value'
    let outputs = 1..4 | par-each { $b | vulkan matrix into-nu }
    check ($outputs | all {|v| $v == [1.0 2.0 3.0]}) 'parallel reads'
}
wait-for-stop

let old = [7.0] | vulkan matrix from-nu --device d0
plugin stop vulkan_matrix
let fresh = [9.0] | vulkan matrix from-nu --device d0
check (try { $old | vulkan matrix into-nu | ignore; false } catch { true }) 'old ID after restart'
check (($fresh | vulkan matrix into-nu) == [9.0]) 'fresh ID after restart'
let small = [0.0 -0.0 1.0e-45] | vulkan matrix from-nu --device d0 --dtype f32
check ($small.dtype == f32) 'explicit f32'
let downloaded = $small | vulkan matrix into-nu
check (($downloaded | get 1 | to msgpack | encode hex) == CB8000000000000000) 'negative zero bits'
check (($downloaded | get 2 | to msgpack | encode hex) == CB36A0000000000000) 'f32 subnormal bits'
check (try { [1] | vulkan matrix from-nu | ignore; false } catch { true }) 'required device'
check (try { [1] | vulkan matrix from-nu --device cpu | ignore; false } catch { true }) 'no CPU fallback'
check (try { [1] | vulkan matrix from-nu --device d0 --dtype f16 | ignore; false } catch { true }) 'invalid dtype'
check (try { [true] | vulkan matrix from-nu --device d0 | ignore; false } catch { true }) 'invalid numeric type'
check (try { read-dynamic [1] | ignore; false } catch { true }) 'expected GPU value'
plugin stop vulkan_matrix
do {
    let result = do {
        let a = [[1 2] [3 4]] | vulkan matrix from-nu --device d0
        let b = [[5 6] [7 8]] | vulkan matrix from-nu --device d0
        $a | vulkan matrix multiply $b
    }
    let branched = $result
    sleep 300ms
    check ((plugin list | where name == vulkan_matrix | get 0.pid) != null) 'GC while only product is held'
    check (($branched | vulkan matrix into-nu) == [[19.0 22.0] [43.0 50.0]]) 'product survives input drop'
}
wait-for-stop
print 'resident Nu tests passed'
