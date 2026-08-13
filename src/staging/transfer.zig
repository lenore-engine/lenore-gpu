const std = @import("std");
const vk = @import("vulkan");

const commands = @import("../device/commands.zig");
const Context = @import("../device/context.zig").Context;
const staging = @import("pool.zig");

const StagingPool = staging.StagingPool;

pub const BeginError = commands.BeginError;
pub const FinishError = commands.SubmitError;
pub const FlushError = FinishError || BeginError;
pub const ReserveError = staging.ReserveError || FlushError;

// Three states, because ownership of the command buffer and permission to free
// it are not the same thing.
//
// Vulkan specification, vkFreeCommandBuffers: a command buffer must not be freed
// while it may still be pending. A submission that fails after the work reached
// the queue establishes neither completion nor that it never started, so the
// buffer stays ours and stays unfreeable. Collapsing that into one boolean is
// what made an earlier version free a possibly pending buffer.
pub const CommandState = enum {
    // Ours, not submitted, safe to free.
    recording,
    // Submitted with an unknown outcome. Freeing it, and destroying anything its
    // commands name, is forbidden until completion or device loss is
    // established, and nothing here can establish either.
    pending,
    // Submitted and waited for. Already freed by the submission path.
    consumed,
};

// One upload job: the command buffer copies are recorded into, and the staging
// space they read from.
//
// The two are one type because a reservation can end a command buffer. When the
// pool is exhausted the only way to reclaim it is to submit what has been
// recorded and wait, which retires the buffer and starts another, so a caller
// holding a vk.CommandBuffer across a reservation would record into a freed one.
// Every consumer therefore takes a *Transfer and reads commandBuffer() at the
// point it records, never before.
//
// A staging pool feeds one transfer at a time. Recycling is by the whole pool,
// so a second transfer sharing it would have its blocks reclaimed underneath a
// submission it is still recording.
pub const Transfer = struct {
    context: *const Context,
    staging: *StagingPool,
    pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
    // Read by the rollback paths of whatever owns this transfer: a pending
    // submission forbids destroying the resources its commands name.
    state: CommandState,
    // How many times the staging pool had to be reclaimed mid-job. Zero means
    // the job fitted; every one after that is a full stall of the graphics
    // queue, and it is the number that says whether the ceiling is too low.
    flushes: u32,

    pub fn begin(
        context: *const Context,
        pool: vk.CommandPool,
        staging_pool: *StagingPool,
    ) BeginError!Transfer {
        return .{
            .context = context,
            .staging = staging_pool,
            .pool = pool,
            .command_buffer = try commands.beginOneShot(context, pool),
            .state = .recording,
            .flushes = 0,
        };
    }

    // The buffer to record into, which changes whenever a reservation had to
    // flush. Call it at the point of recording.
    pub fn commandBuffer(self: *const Transfer) vk.CommandBuffer {
        return self.command_buffer;
    }

    // True when a submission that may still be reading what was recorded has
    // failed, so nothing its commands name may be destroyed. Every rollback
    // along an upload path asks this; what each of them leaks, and so what it
    // reports, is its own.
    pub fn abandoned(self: *const Transfer) bool {
        return self.state == .pending;
    }

    // Reserves as much of the request as the pool can give, reclaiming it first
    // if that is what it takes. The chunk can be shorter than the request; a
    // caller that cannot split passes its own size as the granularity.
    pub fn reserve(self: *Transfer, request: staging.Request) ReserveError!staging.Reservation {
        switch (try self.staging.reserve(request)) {
            .reserved => |reservation| return reservation,
            .flush_required => {},
        }

        try self.flush();

        switch (try self.staging.reserve(request)) {
            .reserved => |reservation| return reservation,
            // Every block is empty now and the policy has already rejected a
            // granule wider than one, so the pool has room for a chunk unless
            // the request never fitted. Reporting that rather than looping is
            // what keeps a retry from being infinite.
            .flush_required => return error.LargerThanBlock,
        }
    }

    // Submits what has been recorded, waits for it, and starts a new command
    // buffer over a reclaimed pool.
    //
    // The state moves before the submission, not after. A failure inside it may
    // leave the buffer pending, and a rollback that assumed otherwise would free
    // it and destroy what its commands name.
    pub fn flush(self: *Transfer) FlushError!void {
        try self.submitAndWait();
        self.command_buffer = try commands.beginOneShot(self.context, self.pool);
        self.state = .recording;
        self.flushes += 1;
    }

    // Submits the last of the work and waits for it. No command buffer is begun
    // after it, which is the whole difference from flush, and deinit is what
    // releases the bookkeeping.
    pub fn finish(self: *Transfer) FinishError!void {
        try self.submitAndWait();
    }

    // Frees the command buffer if it was never submitted. What the commands name
    // outlives this call either way, because a pending submission may still be
    // reading it and nothing here can prove otherwise.
    pub fn deinit(self: *Transfer) void {
        switch (self.state) {
            .recording => self.context.device.freeCommandBuffers(
                self.pool,
                &.{self.command_buffer},
            ),
            .consumed => {},
            .pending => std.log.err(
                "transfer abandoned after a failed submission: one command " ++
                    "buffer is leaked because the submission may still be pending",
                .{},
            ),
        }
        self.* = undefined;
    }

    // This waits on the whole graphics queue and not on this submission:
    // commands.submitOneShotAndWait ends in vkQueueWaitIdle. The two are the
    // same wait whenever nothing else holds work on the queue, which is what a
    // caller that uploads between frames rather than during them gets.
    //
    // Narrowing it means a fence per submission and a submission tag per block,
    // so that a flush waits for the blocks it is reclaiming instead of for
    // everything in flight. That bookkeeping earns its place only for a caller
    // that uploads while frame work is still pending.
    fn submitAndWait(self: *Transfer) commands.SubmitError!void {
        std.debug.assert(self.state == .recording);

        self.state = .pending;
        try commands.submitOneShotAndWait(self.context, self.pool, self.command_buffer);
        self.state = .consumed;
        // The submission has been waited for, so every block it read is free.
        // This is the one place that knows both, and it is why the pool reports
        // exhaustion instead of resolving it.
        self.command_buffer = .null_handle;
        self.staging.recycle();
    }
};
