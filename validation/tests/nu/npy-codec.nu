plugin use vulkan_matrix
use std/assert

def main [codec: path, directory: path] {
    mkdir $directory
    let file = $directory | path join 'roundtrip.npy'
    let rows = [[1.25 -2.5 0.0] [3.0 4.0 5.5]]
    $rows | to msgpack | ^$codec encode $file
    let decoded = ^$codec decode $file | from msgpack
    assert ($decoded == $rows)
    assert (($decoded | into matrix | matrix scale 2 | matrix into-nu) == [[2.5 -5.0 0.0] [6.0 8.0 11.0]])
    let gpu = vulkan matrix from-npy $file --device d0 | vulkan matrix into-nu
    assert ($gpu == $rows)
    let duplicate = $rows | to msgpack | ^$codec encode $file | complete
    assert ($duplicate.exit_code != 0)
    for rows in [[] [[1.0] [2.0 3.0]]] {
        let result = $rows | to msgpack | ^$codec encode ($directory | path join 'invalid.npy') | complete
        assert ($result.exit_code != 0)
    }
    let short = $directory | path join 'short.npy'
    open --raw $file | bytes at 0..20 | save $short
    let result = ^$codec decode $short | complete
    assert ($result.exit_code != 0)
    let truncated = $directory | path join 'truncated.npy'
    let bytes = open --raw $file
    $bytes | bytes at 0..(($bytes | bytes length) - 2) | save $truncated
    let result = ^$codec decode $truncated | complete
    assert ($result.exit_code != 0)
    let f32file = $directory | path join 'f32.npy'
    $rows | vulkan matrix from-nu --device d0 --dtype f32 | vulkan matrix to-npy $f32file
    let result = ^$codec decode $f32file | complete
    assert ($result.exit_code != 0)
    print 'NPY codec: roundtrip, CPU math, GPU compatibility and invalid input passed'
}
