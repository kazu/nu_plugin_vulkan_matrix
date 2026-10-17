mod format;
mod import;
mod transfer;
#[cfg(target_os = "linux")]
mod userptr;

use crate::gpu::{Gpu, Result};
use cubecl::std::tensor::TensorHandle;
use std::{fs::File, path::Path};
use vulkan_matrix_backend::{AutoCompiler, WgpuMemory, WgpuServer, wgpu};

pub(crate) fn load(gpu: &Gpu, path: &Path, shared: bool) -> Result<TensorHandle> {
    let mut file = File::open(path)?;
    let layout = format::read(&mut file)?;
    crate::gpu::check_dtype(gpu, layout.dtype)?;
    gpu.client
        .exclusive(|| {
            (|| -> Result<TensorHandle> {
                let ctx = import::Context::new(gpu)?;
                let handle = if shared {
                    gpu.client
                        .memory_dedicated_allocation((), |()| gpu.client.empty(layout.bytes))
                } else {
                    gpu.client.empty(layout.bytes)
                };
                let resource = gpu
                    .client
                    .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())?;
                let resource = resource.resource();
                let end = (layout.offset + layout.bytes) as u64;
                if shared {
                    let size = resource.buffer.size();
                    let imported =
                        import::Imported::new(&ctx, file, size, layout.offset as u64, false, end)?;
                    let address = imported.address;
                    let raw = unsafe {
                        wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                            imported.buffer,
                            Box::new(move || drop(imported)),
                        )
                    };
                    let buffer = unsafe {
                        gpu.device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                            raw,
                            &wgpu::BufferDescriptor {
                                label: Some("read-only NPY mapping"),
                                size,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            },
                        )
                    };
                    let binding = handle.clone().binding();
                    vulkan_matrix_backend::run_with_server::<WgpuServer<AutoCompiler>, _>(
                        &gpu.client,
                        move |server| unsafe {
                            server.replace_external(binding, WgpuMemory::new(buffer, address))
                        },
                    )??;
                } else {
                    let imported = import::Imported::new(&ctx, file, end, 0, false, end)?;
                    cubecl::future::block_on(gpu.client.sync_buffers([&handle]))?;
                    let raw = unsafe { resource.buffer.as_hal::<wgpu::hal::api::Vulkan>() }
                        .ok_or("Vulkan buffer required")?;
                    transfer::copy(
                        &ctx,
                        imported.buffer,
                        layout.offset as u64,
                        unsafe { raw.raw_handle() },
                        resource.offset,
                        layout.bytes as u64,
                    )?;
                }
                Ok(TensorHandle::new_contiguous(
                    layout.shape,
                    handle,
                    layout.dtype,
                ))
            })()
            .map_err(|e| e.to_string())
        })?
        .map_err(Into::into)
}

pub(crate) fn save(gpu: &Gpu, input: &TensorHandle, path: &Path, force: bool) -> Result<()> {
    crate::gpu::check_dtype(gpu, input.dtype)?;
    gpu.client
        .exclusive(|| {
            (|| -> Result<()> {
                use std::io::Write;
                let ctx = import::Context::new(gpu)?;
                let header = format::header(input.shape(), input.dtype, ctx.alignment)?;
                let tensor = if crate::layout::reshape(input, input.shape()).is_ok() {
                    input.clone()
                } else {
                    cubecl::std::tensor::into_contiguous(
                        &gpu.client,
                        input.clone().binding(),
                        input.dtype,
                    )
                };
                let bytes = crate::layout::count_elements(tensor.shape())?
                    .checked_mul(tensor.dtype.size())
                    .ok_or("array size overflow")?;
                let end = header
                    .len()
                    .checked_add(bytes)
                    .ok_or("file size overflow")?;
                let parent = path.parent().ok_or("output path has no parent")?;
                let mut temp = tempfile::NamedTempFile::new_in(parent)?;
                temp.write_all(&header)?;
                temp.as_file().set_len(
                    end.checked_next_multiple_of(ctx.alignment)
                        .ok_or("file size overflow")? as u64,
                )?;
                let imported = import::Imported::new(
                    &ctx,
                    temp.as_file().try_clone()?,
                    end as u64,
                    0,
                    true,
                    end as u64,
                )?;
                cubecl::future::block_on(gpu.client.sync_buffers([&tensor.handle]))?;
                let resource = gpu
                    .client
                    .get_resource::<WgpuServer<AutoCompiler>>(tensor.handle.clone())?;
                let raw = unsafe {
                    resource
                        .resource()
                        .buffer
                        .as_hal::<wgpu::hal::api::Vulkan>()
                }
                .ok_or("Vulkan buffer required")?;
                transfer::copy(
                    &ctx,
                    unsafe { raw.raw_handle() },
                    resource.resource().offset,
                    imported.buffer,
                    header.len() as u64,
                    bytes as u64,
                )?;
                drop(imported);
                temp.as_file().set_len(end as u64)?;
                temp.as_file().sync_all()?;
                if force {
                    temp.persist(path)?;
                } else {
                    temp.persist_noclobber(path)?;
                }
                File::open(parent)?.sync_all()?;
                Ok(())
            })()
            .map_err(|e| e.to_string())
        })?
        .map_err(Into::into)
}
