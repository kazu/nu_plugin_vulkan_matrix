use super::import::Context;
use crate::gpu::Result;
use ash::vk;

// Caller holds Client::exclusive and every participating resource until this returns.
pub(super) fn copy(
    ctx: &Context,
    source: vk::Buffer,
    src_offset: u64,
    destination: vk::Buffer,
    dst_offset: u64,
    bytes: u64,
) -> Result<()> {
    let d = &ctx.device;
    unsafe {
        let pool = d.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.family),
            None,
        )?;
        let result = (|| -> Result<()> {
            let command = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            d.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    )],
                &[],
                &[],
            );
            d.cmd_copy_buffer(
                command,
                source,
                destination,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(bytes)],
            );
            d.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            d.end_command_buffer(command)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let result = d
                .queue_submit(
                    ctx.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    fence,
                )
                .and_then(|()| d.wait_for_fences(&[fence], true, u64::MAX));
            if result.is_err() {
                let _ = d.device_wait_idle();
            }
            d.destroy_fence(fence, None);
            result?;
            Ok(())
        })();
        d.destroy_command_pool(pool, None);
        result
    }
}
