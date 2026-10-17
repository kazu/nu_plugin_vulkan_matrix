use cubecl_common::{
    device::DeviceService,
    device_handle::{CallResultExt, DeviceHandle},
};
use cubecl_core::{
    client::Client,
    server::{ServerError, ServerStorage},
};
use cubecl_environment::backtrace::BackTrace;

/// Runs an operation on the client's matching backend service.
/// A different server type returns `ServerError::ServiceMismatch`.
pub fn run_with_server<S: ServerStorage + DeviceService, R: Send + 'static>(
    client: &Client,
    operation: impl FnOnce(&mut S) -> R + Send + 'static,
) -> Result<R, ServerError> {
    if client.service_id().service != core::any::TypeId::of::<S>() {
        return Err(ServerError::ServiceMismatch {
            client: format!("{}", client.service_id()),
            requested: String::from(core::any::type_name::<S>()),
            backtrace: BackTrace::capture(),
        });
    }
    Ok(DeviceHandle::<S>::new(client.service_id().device)
        .submit_blocking(operation)
        .unwrap_or_resume())
}
