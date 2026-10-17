plugin use vulkan_matrix

def check [ok: bool, message: string] {
    if not $ok { error make {msg: $message} }
}

def main [--device: string = 'd0'] {
    let dir = (($env.TMPDIR? | default '/tmp') | path join $'vulkan-npy-(random uuid)')
    mkdir $dir
    try {
        let path = ($dir | path join 'matrix.npy')
        let a = [[1.0 2.0 3.0] [4.0 5.0 6.0]] | vulkan matrix from-nu --device $device
        $a | vulkan matrix to-npy $path
        let loaded = vulkan matrix from-npy $path --device $device
        check (($loaded | vulkan matrix into-nu) == ($a | vulkan matrix into-nu)) 'NPY roundtrip'
        let scaled = $loaded | vulkan matrix scale 2
        $scaled | vulkan matrix to-npy ($dir | path join 'scaled.npy')
        check ((vulkan matrix from-npy ($dir | path join 'scaled.npy') --device $device | vulkan matrix into-nu) == ($scaled | vulkan matrix into-nu)) 'computed GPU value output'
        let shared = vulkan matrix from-npy $path --device $device --shared
        check (($shared | vulkan matrix into-nu) == [[1.0 2.0 3.0] [4.0 5.0 6.0]]) 'shared mapping'
        let shared_scaled = $shared | vulkan matrix scale 3
        check (($shared_scaled | vulkan matrix into-nu) == [[3.0 6.0 9.0] [12.0 15.0 18.0]]) 'compute from shared mapping'
        check (($shared | vulkan matrix into-nu) == [[1.0 2.0 3.0] [4.0 5.0 6.0]]) 'shared input is immutable'
        let transposed = $a | vulkan matrix transpose
        $transposed | vulkan matrix to-npy ($dir | path join 'transpose.npy')
        check ((vulkan matrix from-npy ($dir | path join 'transpose.npy') --device $device | vulkan matrix into-nu) == [[1.0 4.0] [2.0 5.0] [3.0 6.0]]) 'strided GPU output'
        let overwrite = try { $scaled | vulkan matrix to-npy $path; false } catch { true }
        check $overwrite 'existing file must be protected'
        check ((vulkan matrix from-npy $path --device $device | vulkan matrix into-nu) == ($a | vulkan matrix into-nu)) 'failed overwrite preserves original payload'
        check (try { $a | vulkan matrix to-npy ($dir | path join 'missing' 'output.npy'); false } catch { true }) 'missing output directory is rejected'
        $scaled | vulkan matrix to-npy $path --force
        check ((vulkan matrix from-npy $path --device $device | vulkan matrix into-nu) == ($scaled | vulkan matrix into-nu)) 'explicit overwrite'
        check (($shared | vulkan matrix into-nu) == [[1.0 2.0 3.0] [4.0 5.0 6.0]]) 'shared mapping survives atomic replacement'
        let single = [1.5 2.5 3.5] | vulkan matrix from-nu --device $device --dtype f32
        $single | vulkan matrix to-npy ($dir | path join 'f32.npy')
        check ((vulkan matrix from-npy ($dir | path join 'f32.npy') --device $device --shared | vulkan matrix into-nu) == [1.5 2.5 3.5]) 'f32 vector shared roundtrip'
        # A standard v1 file with an 80-byte header, not a page-sized header.
        let dict = "{'descr':'<f8','fortran_order':False,'shape':(1,),}"
        let ordinary = (0x[93 4e 55 4d 50 59 01 00 46 00] | bytes add --end ($dict | fill -w 69 -a l | encode utf-8) | bytes add --end 0x[0a 00 00 00 00 00 00 f8 3f])
        $ordinary | save --raw ($dir | path join 'ordinary.npy')
        let relative = do { cd $dir; vulkan matrix from-npy ordinary.npy --device $device }
        check (($relative | vulkan matrix into-nu) == [1.5]) 'ordinary 80-byte NPY header and Nu cwd'
        ($ordinary | bytes at 0..86) | save --raw ($dir | path join 'truncated.npy')
        check (try { vulkan matrix from-npy ($dir | path join 'truncated.npy') --device $device; false } catch { true }) 'truncated NPY is rejected'
    } catch {|err|
        rm -rf $dir
        error make $err
    }
    rm -rf $dir
}
