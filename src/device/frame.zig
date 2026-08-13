const std = @import("std");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;

pub const InitError = vk.DeviceWrapper.CreateCommandPoolError ||
    vk.DeviceWrapper.AllocateCommandBuffersError ||
    vk.DeviceWrapper.CreateFenceError ||
    vk.DeviceWrapper.CreateSemaphoreError;

pub const WaitError = vk.DeviceWrapper.WaitForFencesError;
pub const BeginError = vk.DeviceWrapper.ResetCommandPoolError ||
    vk.DeviceWrapper.BeginCommandBufferError;
pub const SubmitError = vk.DeviceWrapper.EndCommandBufferError ||
    vk.DeviceWrapper.ResetFencesError ||
    vk.DeviceWrapper.QueueSubmit2Error;

// What one submitted frame waits on and signals. Both semaphores belong to the
// caller: the one waited on is the frame's own image-acquired semaphore, and the
// one signalled belongs to the swapchain image being written, because that is
// what presentation waits on.
//
// The wait stage is the caller's because it depends on what was recorded. A
// frame that clears through a transfer waits at a different stage from one that
// renders into an attachment, and naming either here would over-synchronize one
// of them or under-synchronize the other.
pub const Submission = struct {
    wait: vk.Semaphore,
    wait_stage: vk.PipelineStageFlags2,
    signal: vk.Semaphore,
    signal_stage: vk.PipelineStageFlags2,
};

// Vulkan specification, vkWaitForFences and the fence parameter of
// vkQueueSubmit2: a slot's pool and acquire semaphore are reused only after its
// submission fence signals.
pub const Frame = struct {
    // A dedicated pool lets the frame loop reset this slot without disturbing
    // command buffers owned by other in-flight slots.
    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
    fence: vk.Fence,
    image_acquired: vk.Semaphore,

    pub fn init(context: *const Context) InitError!Frame {
        const command_pool = try context.device.createCommandPool(&.{
            .queue_family_index = context.graphics_queue.family,
            .flags = .{ .transient_bit = true },
        }, null);
        errdefer context.device.destroyCommandPool(command_pool, null);

        var command_buffer: vk.CommandBuffer = undefined;
        try context.device.allocateCommandBuffers(&.{
            .command_pool = command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&command_buffer));

        // The first use follows the same wait path as every reuse.
        const fence = try context.device.createFence(&.{
            .flags = .{ .signaled_bit = true },
        }, null);
        errdefer context.device.destroyFence(fence, null);

        const image_acquired = try context.device.createSemaphore(&.{}, null);
        errdefer context.device.destroySemaphore(image_acquired, null);

        return .{
            .command_pool = command_pool,
            .command_buffer = command_buffer,
            .fence = fence,
            .image_acquired = image_acquired,
        };
    }

    pub fn deinit(self: Frame, context: *const Context) void {
        context.device.destroySemaphore(self.image_acquired, null);
        context.device.destroyFence(self.fence, null);
        // Vulkan specification, vkDestroyCommandPool: this also frees every
        // command buffer allocated from the pool.
        context.device.destroyCommandPool(self.command_pool, null);
    }

    pub fn waitForGpu(self: Frame, context: *const Context) WaitError!void {
        _ = try context.device.waitForFences(&.{self.fence}, .true, std.math.maxInt(u64));
    }

    // Recycles this slot's command buffer and opens it for recording. The pool
    // is reset rather than the buffer, which is why it carries no reset flag:
    // resetting the pool releases every buffer in it at once, and this slot owns
    // exactly one.
    //
    // The caller has waited on the fence, so the work this recycles is complete.
    pub fn beginCommands(self: Frame, context: *const Context) BeginError!vk.CommandBuffer {
        try context.device.resetCommandPool(self.command_pool, .{});
        try context.device.beginCommandBuffer(self.command_buffer, &.{
            .flags = .{ .one_time_submit_bit = true },
        });
        return self.command_buffer;
    }

    // Ends recording and submits, signalling this slot's fence on completion.
    //
    // The fence is reset here rather than after waiting on it, so a frame that
    // returns early between the wait and the submission leaves it signalled and
    // the next wait does not hang.
    pub fn submit(
        self: Frame,
        context: *const Context,
        submission: Submission,
    ) SubmitError!void {
        try context.device.endCommandBuffer(self.command_buffer);

        const wait = [_]vk.SemaphoreSubmitInfo{.{
            .semaphore = submission.wait,
            .value = 0,
            .stage_mask = submission.wait_stage,
            .device_index = 0,
        }};
        const signal = [_]vk.SemaphoreSubmitInfo{.{
            .semaphore = submission.signal,
            .value = 0,
            .stage_mask = submission.signal_stage,
            .device_index = 0,
        }};
        const buffers = [_]vk.CommandBufferSubmitInfo{.{
            .command_buffer = self.command_buffer,
            // Vulkan specification, VkCommandBufferSubmitInfo: deviceMask must
            // not be zero, including when no device group is enabled.
            .device_mask = 1,
        }};

        try context.device.resetFences(&.{self.fence});
        try context.device.queueSubmit2(context.graphics_queue.handle, &.{.{
            .wait_semaphore_info_count = wait.len,
            .p_wait_semaphore_infos = &wait,
            .command_buffer_info_count = buffers.len,
            .p_command_buffer_infos = &buffers,
            .signal_semaphore_info_count = signal.len,
            .p_signal_semaphore_infos = &signal,
        }}, self.fence);
    }
};
