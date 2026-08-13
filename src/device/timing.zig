// Where a frame's device time went, measured on the device.
//
// The host's phase timings can only say how long it waited, which is the length
// of the longer of the two sides and not a decomposition of either. A scene
// where the wait is most of the frame is one this could say nothing about beyond
// naming the device as the limit.
//
// Two timestamps per pass, written into a pool with a run of slots per frame in
// flight. A frame's slots are read after its fence, which is the same point the
// loop already waits at, so nothing here adds a synchronisation of its own.

const std = @import("std");
const vk = @import("vulkan");

const Context = @import("context.zig").Context;

// The passes a frame is divided into. The names are the recorder's, and a pass
// that records nothing this frame reports zero rather than going missing: a
// reader comparing two frames wants the same rows in both.
pub const Pass = enum {
    shadow,
    main,
    bloom,
    post,

    pub const count = @typeInfo(Pass).@"enum".fields.len;
};

pub const Edge = enum { begin, end };

pub const slots_per_frame = Pass.count * 2;

// Which query a frame's pass boundary writes into.
//
// Split out and public because it is the whole of the layout: a wrong index here
// reads another pass's timestamp, and the result is a plausible number rather
// than an error. Nothing in it needs a device.
pub fn slot(frame_index: usize, pass: Pass, edge: Edge) u32 {
    // Widened before the arithmetic. The tag of a four-member enum is two bits
    // wide, and doubling it in that width overflows at the third pass.
    const ordinal: usize = @intFromEnum(pass);
    const within = ordinal * 2 + @intFromEnum(edge);
    return @intCast(frame_index * slots_per_frame + within);
}

// Nanoseconds between two timestamps of the same pool.
//
// The counter is only `valid_bits` wide, and the specification leaves the bits
// above it undefined, so both readings are masked before they are subtracted.
// Wrapping subtraction then gives the right interval across the wrap, which a
// signed difference would report as an enormous negative one.
//
// Vulkan specification, vkGetPhysicalDeviceQueueFamilyProperties:
// timestampValidBits is the number of meaningful bits, between 36 and 64 for a
// queue that supports timestamps at all. The period is nanoseconds per tick from
// VkPhysicalDeviceLimits::timestampPeriod.
pub fn durationNs(begin: u64, end: u64, valid_bits: u32, period_ns: f32) u64 {
    std.debug.assert(valid_bits > 0 and valid_bits <= 64);
    const mask: u64 = if (valid_bits == 64)
        std.math.maxInt(u64)
    else
        (@as(u64, 1) << @intCast(valid_bits)) - 1;

    const ticks = (end & mask) -% (begin & mask) & mask;
    const nanoseconds = @as(f64, @floatFromInt(ticks)) * period_ns;
    // A device whose period puts the interval past what the return type holds is
    // reporting something no frame produced, and a saturating conversion says
    // that better than a wrap would.
    if (!(nanoseconds >= 0)) return 0;
    if (nanoseconds >= @as(f64, @floatFromInt(std.math.maxInt(u64)))) return std.math.maxInt(u64);
    return @intFromFloat(nanoseconds);
}

// One frame's decomposition. Every pass is present; a pass the frame did not
// record reads zero.
pub const Frame = struct {
    pass_ns: [Pass.count]u64 = @splat(0),

    pub fn get(self: Frame, pass: Pass) u64 {
        return self.pass_ns[@intFromEnum(pass)];
    }

    pub fn total(self: Frame) u64 {
        var sum: u64 = 0;
        for (self.pass_ns) |value| sum += value;
        return sum;
    }
};

pub const InitError = vk.DeviceWrapper.CreateQueryPoolError;

// Whether this device can be asked at all, and at what resolution.
//
// Both come from the physical device rather than from a guess: a queue family
// may report zero valid bits, which is the specification's way of saying that
// writing a timestamp on it is not supported.
pub const Support = struct {
    valid_bits: u32,
    period_ns: f32,

    pub fn available(self: Support) bool {
        return self.valid_bits > 0 and self.period_ns > 0;
    }
};

pub const GpuTimer = struct {
    pool: vk.QueryPool,
    support: Support,
    frames: usize,
    // Whether a frame's slots hold a submitted frame's results. Read back before
    // the first submission would ask the driver for a query nothing wrote, which
    // is a wait with no writer.
    written: [max_frames]bool,

    // The frame rings this engine builds are two deep, and a pool sized for a
    // count the caller invents is a pool the layout arithmetic cannot check.
    pub const max_frames = 4;

    pub fn init(context: *const Context, support: Support, frames: usize) InitError!GpuTimer {
        std.debug.assert(frames > 0 and frames <= max_frames);
        const pool = try context.device.createQueryPool(&.{
            .query_type = .timestamp,
            .query_count = @intCast(frames * slots_per_frame),
        }, null);
        return .{
            .pool = pool,
            .support = support,
            .frames = frames,
            .written = @splat(false),
        };
    }

    pub fn deinit(self: *GpuTimer, context: *const Context) void {
        context.device.destroyQueryPool(self.pool, null);
        self.* = undefined;
    }

    // Clears this frame's run of slots. Vulkan specification, vkCmdWriteTimestamp2:
    // a query must be reset before it is written, and the reset has to be outside
    // a render pass instance, so this is recorded at the top of the frame.
    pub fn reset(self: *const GpuTimer, command_buffer: vk.CommandBuffer, context: *const Context, frame_index: usize) void {
        context.device.cmdResetQueryPool(
            command_buffer,
            self.pool,
            slot(frame_index, .shadow, .begin),
            slots_per_frame,
        );
    }

    // One boundary. `all_commands_bit` on both edges rather than a narrower
    // stage: what is wanted is when the pass finished, not when one stage of it
    // reached a given point, and a narrower stage would time a prefix of the work.
    pub fn write(
        self: *const GpuTimer,
        command_buffer: vk.CommandBuffer,
        context: *const Context,
        frame_index: usize,
        pass: Pass,
        edge: Edge,
    ) void {
        context.device.cmdWriteTimestamp2(
            command_buffer,
            .{ .all_commands_bit = true },
            self.pool,
            slot(frame_index, pass, edge),
        );
    }

    // Marks a frame's slots as carrying results. Called where the frame is
    // submitted, so a read of a frame that was recorded and never submitted
    // cannot wait on a query the device will never write.
    pub fn markSubmitted(self: *GpuTimer, frame_index: usize) void {
        self.written[frame_index] = true;
    }

    // This frame's decomposition, or null when it holds nothing yet.
    //
    // The caller guarantees the frame's fence has been waited on. `.wait` is not
    // passed for that reason: the results are already there, and asking the
    // driver to wait would hide a caller that read too early behind a stall.
    pub fn read(self: *const GpuTimer, context: *const Context, frame_index: usize) ?Frame {
        if (!self.support.available()) return null;
        if (!self.written[frame_index]) return null;

        var ticks: [slots_per_frame]u64 = @splat(0);
        const result = context.device.getQueryPoolResults(
            self.pool,
            slot(frame_index, .shadow, .begin),
            slots_per_frame,
            @sizeOf(@TypeOf(ticks)),
            &ticks,
            @sizeOf(u64),
            .{ .@"64_bit" = true },
        ) catch return null;
        // `not_ready` is a frame whose queries were reset and never written,
        // which is what a pass that recorded nothing leaves behind.
        if (result != .success) return null;

        var frame: Frame = .{};
        for (0..Pass.count) |index| {
            frame.pass_ns[index] = durationNs(
                ticks[index * 2],
                ticks[index * 2 + 1],
                self.support.valid_bits,
                self.support.period_ns,
            );
        }
        return frame;
    }
};
