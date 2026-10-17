use super::userptr;
use crate::gpu::{Gpu, Result};
use ash::vk;
use cubecl::wgpu::wgpu;
use std::{fs::File, num::NonZeroU64};

pub(super) struct Context {
    pub device: ash::Device,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub family: u32,
    pub alignment: usize,
}
impl Context {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let hal = unsafe { gpu.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or("Vulkan device required")?;
        let instance = hal.shared_instance().raw_instance().clone();
        let device = hal.raw_device().clone();
        if unsafe {
            instance.get_device_proc_addr(
                device.handle(),
                c"vkGetMemoryHostPointerPropertiesEXT".as_ptr(),
            )
        }
        .is_none()
        {
            return Err("VK_EXT_external_memory_host is unavailable".into());
        }
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        unsafe {
            instance.get_physical_device_properties2(
                hal.raw_physical_device(),
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut host),
            );
        }
        let alignment = usize::try_from(host.min_imported_host_pointer_alignment)?;
        if !alignment.is_power_of_two() {
            return Err("invalid host import alignment".into());
        }
        Ok(Self {
            device,
            instance,
            physical: hal.raw_physical_device(),
            queue: hal.raw_queue(),
            family: hal.queue_family_index(),
            alignment,
        })
    }
}

enum Mapping {
    Read(memmap2::Mmap),
    Write(memmap2::MmapMut),
}
impl Mapping {
    fn pointer(&self) -> *mut std::ffi::c_void {
        match self {
            Self::Read(m) => m.as_ptr().cast_mut().cast(),
            Self::Write(m) => m.as_ptr().cast_mut().cast(),
        }
    }
}
pub(super) struct Imported {
    device: ash::Device,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub address: Option<NonZeroU64>,
    _mapping: Mapping,
    _file: File,
}
impl Drop for Imported {
    fn drop(&mut self) {
        // Every raw submission is fenced; externally owned wgpu buffers defer this drop.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
impl Imported {
    pub fn new(
        ctx: &Context,
        file: File,
        size: u64,
        offset: u64,
        writable: bool,
        required_len: u64,
    ) -> Result<Self> {
        let d = &ctx.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external),
                None,
            )?
        };
        let allocated = (|| -> Result<_> {
            let req = unsafe { d.get_buffer_memory_requirements(buffer) };
            if !offset.is_multiple_of(req.alignment) {
                return Err("NPY data offset cannot be bound directly; omit --shared".into());
            }
            let len = usize::try_from(offset.checked_add(req.size).ok_or("import size overflow")?)?
                .checked_next_multiple_of(ctx.alignment)
                .ok_or("import size overflow")?;
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0 {
                return Err("cannot obtain page size".into());
            }
            let file_len = file.metadata()?.len();
            if file_len < required_len
                || len as u64
                    > file_len
                        .checked_next_multiple_of(page as u64)
                        .ok_or("file size overflow")?
            {
                return Err("file does not cover the GPU import range".into());
            }
            let mapping = if writable {
                Mapping::Write(unsafe { memmap2::MmapOptions::new().len(len).map_mut(&file)? })
            } else {
                Mapping::Read(unsafe { memmap2::MmapOptions::new().len(len).map(&file)? })
            };
            let pointer = mapping.pointer();
            if !(pointer as usize).is_multiple_of(ctx.alignment) {
                return Err("mmap address does not meet GPU alignment".into());
            }
            let host = ash::ext::external_memory_host::Device::new(&ctx.instance, d);
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            unsafe {
                (host.fp().get_memory_host_pointer_properties_ext)(
                    d.handle(),
                    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                    pointer,
                    &mut props,
                )
                .result()?;
            }
            let memory = unsafe {
                ctx.instance
                    .get_physical_device_memory_properties(ctx.physical)
            };
            let index = (0..memory.memory_type_count)
                .find(|&i| {
                    req.memory_type_bits & props.memory_type_bits & (1 << i) != 0
                        && memory.memory_types[i as usize].property_flags.contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                })
                .ok_or("no coherent host import memory type")?;
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(pointer);
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = userptr::with_import(pointer as usize, len, writable, || unsafe {
                d.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(len as u64)
                        .memory_type_index(index)
                        .push_next(&mut import)
                        .push_next(&mut flags),
                    None,
                )
            })?;
            if let Err(error) = unsafe { d.bind_buffer_memory(buffer, memory, offset) } {
                unsafe {
                    d.free_memory(memory, None);
                }
                return Err(error.into());
            }
            let address = unsafe {
                d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
            };
            Ok((memory, NonZeroU64::new(address), mapping))
        })();
        match allocated {
            Ok((memory, address, mapping)) => {
                let owned = Self {
                    device: d.clone(),
                    buffer,
                    memory,
                    address,
                    _mapping: mapping,
                    _file: file,
                };
                if owned.address.is_none() {
                    return Err("missing buffer device address".into());
                }
                Ok(owned)
            }
            Err(error) => {
                unsafe {
                    d.destroy_buffer(buffer, None);
                }
                Err(error)
            }
        }
    }
}
