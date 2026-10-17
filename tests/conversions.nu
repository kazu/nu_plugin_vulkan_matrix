plugin use vulkan_matrix

def check [condition: bool, message: string] {
    if not $condition { error make {msg: $message} }
}

let native = [[1 2] [3 4]] | into matrix
let gpu = $native | matrix into-nu | vulkan matrix from-nu --device d0
check ($gpu.shape == [2 2]) 'native matrix shape'
check (($gpu | vulkan matrix into-nu | into matrix | matrix into-nu) == ($native | matrix into-nu)) 'native matrix round trip'
let table_gpu = $native | matrix into-nu --as-records | vulkan matrix from-nu --device d0
check (($table_gpu | vulkan matrix into-nu --as-records) == ($native | matrix into-nu --as-records)) 'native records round trip'

let nd = [[[1 2] [3 4]] [[5 6] [7 8]]]
let nd_gpu = $nd | vulkan matrix from-nu --device d0 --dtype f32
check ($nd_gpu.shape == [2 2 2]) '3D shape'
check (($nd_gpu | vulkan matrix into-nu) == $nd) '3D values'
check (try { $nd_gpu | vulkan matrix into-nu --as-records | ignore; false } catch { true }) 'records require 2D'
let scalar = 5.5 | vulkan matrix from-nu --device d0
check ($scalar.shape == []) 'scalar shape'
check (($scalar | vulkan matrix into-nu) == 5.5) 'scalar value'

let empty = [] | vulkan matrix from-nu --device d0 --shape [2 0 3]
check ($empty.shape == [2 0 3]) 'empty trailing dimensions'
let restored = $empty | vulkan matrix into-nu | vulkan matrix from-nu --device d0 --shape $empty.shape
check ($restored.shape == $empty.shape) 'empty shape restored explicitly'
let reshaped = [1 2 3 4] | vulkan matrix from-nu --device d0 --shape [2 2]
check (($reshaped | vulkan matrix into-nu) == [[1 2] [3 4]]) 'explicit reshape'
let ordered = [{a: 1, b: 2} {b: 4, a: 3}] | vulkan matrix from-nu --device d0
check (($ordered | vulkan matrix into-nu) == [[1 2] [3 4]]) 'record column order'

check (try { [[1] [2 3]] | vulkan matrix from-nu --device d0 | ignore; false } catch { true }) 'ragged input'
check (try { [{a: 1} {b: 2}] | vulkan matrix from-nu --device d0 | ignore; false } catch { true }) 'mismatched columns'
check (try { [1] | vulkan matrix from-nu --device d0 --shape [2] | ignore; false } catch { true }) 'wrong element count'
check (try { [1] | vulkan matrix from-nu --device d0 --shape [-1] | ignore; false } catch { true }) 'negative dimension'
check (try { [1] | vulkan matrix from-nu --device d0 --shape [9223372036854775807 2] | ignore; false } catch { true }) 'shape overflow'
let enormous_empty = [] | vulkan matrix from-nu --device d0 --shape [9223372036854775807 0]
check (try { $enormous_empty | vulkan matrix into-nu | ignore; false } catch { true }) 'Nu list capacity error'
check (try { $enormous_empty | vulkan matrix into-nu --as-records | ignore; false } catch { true }) 'Nu records capacity error'
check (($gpu | vulkan matrix into-nu) == [[1 2] [3 4]]) 'plugin survives invalid conversions'
print 'conversion Nu tests passed'
