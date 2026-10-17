plugin use vulkan_matrix

let devices = vulkan matrix devices
if ($devices | is-empty) { error make {msg: 'no hardware Vulkan GPU found'} }
for device in $devices {
    if ($device.name | is-empty) { error make {msg: 'missing GPU name'} }
    let value = [1.0 -2.5] | vulkan matrix from-nu --device $device.device
    if $value.device != $device.device { error make {msg: 'device metadata mismatch'} }
    if ($value | vulkan matrix into-nu) != [1.0 -2.5] { error make {msg: 'device transfer mismatch'} }
}
print ($devices | to nuon)
print 'device Nu tests passed'
