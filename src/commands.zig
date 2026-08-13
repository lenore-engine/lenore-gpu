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

// An execution and memory dependency between two scopes, as
// synchronization2 states one: what happened before, and what may observe it.
//
// The four masks are named rather than collapsed into cases. Every dependency
// this module and its consumers record is different from every other one:
// compute writes read back through the vertex attribute fetch, compute writes
// read in a vertex shader, a transfer fill read by a compute dispatch, one
// dispatch's output read by a copy, another dispatch and a fragment shader at
// once. A set of named cases with one caller each is a rename, not an
// abstraction, and the mask a caller has to get right stays the same either way.
pub const Dependency = struct {
    src_stage: vk.PipelineStageFlags2,
    src_access: vk.AccessFlags2,
    dst_stage: vk.PipelineStageFlags2,
    dst_access: vk.AccessFlags2,
};

// The barrier a dependency is, without recording it. Separated from the call so
// that the mapping is reachable from a test: a source scope written into a
// destination member orders the opposite of what the caller asked for, and
// nothing reports it. The picture is merely wrong, one frame late.
pub fn memoryBarrier(dependency: Dependency) vk.MemoryBarrier2 {
    return .{
        .src_stage_mask = dependency.src_stage,
        .src_access_mask = dependency.src_access,
        .dst_stage_mask = dependency.dst_stage,
        .dst_access_mask = dependency.dst_access,
    };
}

// One global memory barrier, which is what a dependency over buffer contents
// wants. A buffer barrier would additionally have to name a range, and for a
// device-local resource that buys no filtering the driver acts on; the same
// reasoning is written out at `morph.barrier`.
//
// Image contents are not this function's business: they need a layout
// transition per image, which is what the `beginBarriers` and `endBarriers`
// pairs in `pass`, `post` and `shadow` are.
pub fn recordMemoryBarrier(
    context: *const Context,
    command_buffer: vk.CommandBuffer,
    dependency: Dependency,
) void {
    const barriers = [_]vk.MemoryBarrier2{memoryBarrier(dependency)};
    context.device.cmdPipelineBarrier2(command_buffer, &.{
        .memory_barrier_count = barriers.len,
        .p_memory_barriers = &barriers,
    });
}

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
