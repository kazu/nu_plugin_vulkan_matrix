mod support;

use cubecl::server::ServerError;
use nu_plugin_vulkan_matrix::{AutoCompiler, WgpuServer, run_with_server};

#[test]
fn executes_on_the_clients_server() {
    let gpu = support::gpu();
    let value = run_with_server::<WgpuServer<AutoCompiler>, _>(&gpu.client, |_| 42).unwrap();
    assert_eq!(value, 42);
}

#[test]
fn rejects_a_different_backend_without_running_the_operation() {
    let gpu = support::gpu();
    let result = run_with_server::<cubecl::wgpu::WgpuServer<cubecl::wgpu::AutoCompiler>, ()>(
        &gpu.client,
        |_| panic!("mismatched operation must not run"),
    );
    assert!(matches!(result, Err(ServerError::ServiceMismatch { .. })));
}
