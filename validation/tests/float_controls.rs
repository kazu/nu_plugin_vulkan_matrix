mod support;

use cubecl::{Compiler, VulkanCompilationOptions, WgpuCompilationOptions, cube, prelude::*};
use cubecl_runtime::runtime::Runtime;
use nu_plugin_vulkan_matrix::{AutoCompiler, AutoRepresentation, WgpuRuntime};
use std::{process::Command, sync::Arc};

#[cube(launch)]
fn write_float<T: Float>(output: &mut [T]) {
    output[0] = T::new(1.0);
}

#[test]
fn older_spirv_declares_float_controls_extension() {
    for (version, target) in [((1, 3), "vulkan1.1"), ((1, 4), "vulkan1.1spv1.4")] {
        check_generated::<f32>(version, target);
        check_generated::<f64>(version, target);
    }
}

fn check_generated<T: Float>(version: (u8, u8), target: &str) {
    let gpu = support::gpu();
    let settings = KernelSettings::new(
        *CubeDim::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    );
    let kernel = write_float::WriteFloat::<T>::new(
        settings,
        gpu.client.properties_shared(),
        Arc::new(WgpuRuntime::<AutoCompiler>::target_properties()),
        BufferCompilationArg { inplace: None },
    );
    let options = WgpuCompilationOptions {
        supports_vulkan_compiler: true,
        vulkan: VulkanCompilationOptions {
            max_spirv_version: version,
            max_vector_size: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let repr = AutoCompiler::SpirV(cubecl_spirv::SpirvCompiler)
        .compile(kernel.define(), &options)
        .unwrap();
    let AutoRepresentation::SpirV(repr) = repr else {
        panic!("expected SPIR-V compilation");
    };
    if version < (1, 4) {
        assert!(
            repr.to_string()
                .contains("OpExtension \"SPV_KHR_float_controls\"")
        );
        // The upstream 1.3 entry-point interface is independently invalid.
        return;
    }
    let path =
        std::env::temp_dir().join(format!("vulkan-matrix-controls-{}.spv", std::process::id()));
    let bytes: Vec<u8> = repr
        .assembled_module
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    std::fs::write(&path, bytes).unwrap();
    let result = Command::new("spirv-val")
        .args(["--target-env", target])
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
