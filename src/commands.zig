const vk = @import("vulkan");
const Context = @import("context.zig").Context;

pub const BeginError = vk.DeviceWrapper.AllocateCommandBuffersError ||
    vk.DeviceWrapper.BeginCommandBufferError;

pub const SubmitError = vk.DeviceWrapper.EndCommandBufferError ||
    vk.DeviceWrapper.QueueSubmit2Error ||
    vk.DeviceWrapper.QueueWaitIdleError;

pub const PoolInitError = vk.DeviceWrapper.CreateCommandPoolError;

// A command pool for the one-shot path below, owned by whoever does cold setup:
// asset upload, initial layout transitions, anything submitted outside the frame
// loop. It exists because the frame's pool must not be borrowed for this. That
// one is reset as a whole when its slot is reused, which would take any
// long-lived setup buffer with it.
//
// The family is the graphics queue's because submitOneShotAndWait submits there.
// A pool from any other family produces command buffers that queue cannot
// execute, and that is the invariant the two functions below rely on.
pub const OneShotPool = struct {
    handle: vk.CommandPool,

    pub fn init(context: *const Context) PoolInitError!OneShotPool {
        return .{
            .handle = try context.device.createCommandPool(&.{
                .queue_family_index = context.graphics_queue.family,
                // Vulkan specification, VkCommandPoolCreateFlagBits: transient
                // is a hint that the buffers are short-lived, which is what
                // one-shot means. No reset bit: that one is required by
                // vkResetCommandBuffer and by re-beginning a recorded buffer,
                // and this path allocates a fresh buffer per submission and
                // frees it. vkFreeCommandBuffers needs no pool flag.
                .flags = .{ .transient_bit = true },
            }, null),
        };
    }

    // Vulkan specification, vkDestroyCommandPool: every command buffer allocated
    // from the pool must have completed. On the one-shot path that holds once the
    // last submitOneShotAndWait has returned, because it waits.
    pub fn deinit(self: *OneShotPool, context: *const Context) void {
        context.device.destroyCommandPool(self.handle, null);
        self.* = undefined;
    }
};

pub fn beginOneShot(
    context: *const Context,
    pool: vk.CommandPool,
) BeginError!vk.CommandBuffer {
    var command_buffer: vk.CommandBuffer = undefined;
    try context.device.allocateCommandBuffers(&.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&command_buffer));
    errdefer context.device.freeCommandBuffers(pool, &.{command_buffer});

    try context.device.beginCommandBuffer(command_buffer, &.{
        .flags = .{ .one_time_submit_bit = true },
    });
    return command_buffer;
}
// This synchronous path is for cold setup. Frame and bulk-upload paths batch
// work instead of idling the complete graphics queue after each command buffer.
// On success the command buffer is freed. On error ownership remains with the
// caller; Vulkan specification, vkFreeCommandBuffers, forbids freeing it while
// it may still be pending.
pub fn submitOneShotAndWait(
    context: *const Context,
    pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
) SubmitError!void {
    try context.device.endCommandBuffer(command_buffer);

    const command_buffer_infos = [_]vk.CommandBufferSubmitInfo{.{
        .command_buffer = command_buffer,
        // Vulkan specification, VkCommandBufferSubmitInfo: deviceMask must not
        // be zero, including when no device group is enabled.
        .device_mask = 1,
    }};
    try context.device.queueSubmit2(context.graphics_queue.handle, &.{.{
        .command_buffer_info_count = command_buffer_infos.len,
        .p_command_buffer_infos = &command_buffer_infos,
    }}, .null_handle);
    try context.device.queueWaitIdle(context.graphics_queue.handle);

    context.device.freeCommandBuffers(pool, &.{command_buffer});
}
