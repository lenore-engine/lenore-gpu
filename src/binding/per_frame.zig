const std = @import("std");
const vk = @import("vulkan");
const Context = @import("../device/context.zig").Context;
const buffer = @import("../object/buffer.zig");
const descriptors = @import("descriptors.zig");
const memory = @import("../memory/allocator.zig");

pub const InitError = error{
    ZeroFrames,
    ZeroCount,

    // The ring exists to be written through its mapping, so memory the host
    // cannot see is a configuration error rather than a fall back to staging.
    NotHostVisible,

    // Buffer memory requirements are the device's and the payload's alignment
    // is the compiler's. Nothing relates the two, so the mapping is measured
    // against the type once here instead of being assumed by `slice`.
    MisalignedMapping,
} || buffer.InitError;

// How one ring is laid out in its buffer. Separated from `init` so the
// arithmetic can be exercised without a device, the same split `buffer.zig` and
// the staging arena make: everything here that can be wrong is arithmetic.
pub const Layout = struct {
    stride: vk.DeviceSize,
    total: vk.DeviceSize,
};

pub const LayoutError = error{
    ZeroFrames,
    ZeroCount,
    SizeOverflow,
};

// `alignment` is a power of two. Both device limits it is derived from carry
// `limittype="min,pot"` in vk.xml, and `@alignOf` is one by definition, so the
// larger of them is too.
pub fn layout(
    frames: usize,
    count: usize,
    element_size: vk.DeviceSize,
    alignment: vk.DeviceSize,
) LayoutError!Layout {
    if (frames == 0) return error.ZeroFrames;
    if (count == 0) return error.ZeroCount;

    const payload = std.math.mul(vk.DeviceSize, @intCast(count), element_size) catch
        return error.SizeOverflow;
    // alignForward would wrap on a payload within `alignment` of the maximum.
    const stride = std.math.add(vk.DeviceSize, payload, alignment - 1) catch
        return error.SizeOverflow;
    const aligned = stride & ~(alignment - 1);
    const total = std.math.mul(vk.DeviceSize, aligned, @intCast(frames)) catch
        return error.SizeOverflow;

    return .{ .stride = aligned, .total = total };
}

// Vulkan specification, VkPhysicalDeviceLimits: a dynamic uniform buffer offset
// is a multiple of minUniformBufferOffsetAlignment and a dynamic storage buffer
// offset a multiple of minStorageBufferOffsetAlignment. A buffer declaring both
// usages has to satisfy both, and both are powers of two, so the larger does.
// The payload's own alignment joins them, because a slot is read as a `T`.
pub fn offsetAlignment(
    usage: vk.BufferUsageFlags,
    limits: vk.PhysicalDeviceLimits,
    element_alignment: vk.DeviceSize,
) vk.DeviceSize {
    var alignment: vk.DeviceSize = element_alignment;
    if (usage.uniform_buffer_bit)
        alignment = @max(alignment, limits.min_uniform_buffer_offset_alignment);
    if (usage.storage_buffer_bit)
        alignment = @max(alignment, limits.min_storage_buffer_offset_alignment);
    return alignment;
}

// A ring of `frames` copies of `count` elements of `T`, held as sub-ranges of
// one buffer and written through one persistent mapping.
//
// One buffer rather than one per frame. The frames differ only by an offset,
// so a single descriptor of type `uniform_buffer_dynamic` or
// `storage_buffer_dynamic` serves all of them and the offset is supplied at
// bind time. Per-frame buffers instead force a descriptor set per frame, which
// is what turns one binding into a pool, a layout and N sets at every site that
// wants per-frame data.
//
// The safety invariant, and the only one this type has: a slot is written by
// the host after that slot's submission fence has signalled, and before the
// same slot is submitted again. Nothing else writes these bytes and nothing
// reads them back, so no barrier belongs here. A caller writing a slot the
// device is still reading corrupts the frame in flight, and no check in this
// file can observe it.
//
// The memory needs no flush: `MemoryAllocator.chooseMemoryType` accepts an
// upload memory type only when it carries both `host_visible_bit` and
// `host_coherent_bit`, so a written slot is visible to the device as it stands.
pub fn PerFrame(comptime T: type) type {
    return struct {
        const Self = @This();

        storage: buffer.Buffer,
        count: usize,
        frame_count: usize,

        // Distance between one frame's first byte and the next frame's. The
        // payload is padded up to it so that every slot begins at an offset the
        // device accepts as a dynamic offset.
        stride: vk.DeviceSize,

        pub fn init(
            context: *const Context,
            memory_allocator: *memory.MemoryAllocator,
            frames: usize,
            count: usize,
            usage: vk.BufferUsageFlags,
        ) (InitError || LayoutError)!Self {
            const alignment = offsetAlignment(usage, context.properties.limits, @alignOf(T));
            const plan = try layout(frames, count, @sizeOf(T), alignment);

            var storage = try buffer.Buffer.init(context, memory_allocator, plan.total, usage, .upload);
            errdefer storage.deinit();

            // Every slot starts a multiple of the stride from the base, and the
            // stride carries `@alignOf(T)` because `offsetAlignment` folded it
            // in, so the base carrying it is what makes all of them aligned.
            const mapping = storage.allocation.mappedBytes() orelse return error.NotHostVisible;
            if (!std.mem.isAligned(@intFromPtr(mapping.ptr), @alignOf(T)))
                return error.MisalignedMapping;

            return .{
                .storage = storage,
                .count = count,
                .frame_count = frames,
                .stride = plan.stride,
            };
        }

        // Vulkan specification, vkDestroyBuffer and vkFreeMemory: every
        // submission reading any slot must have completed.
        pub fn deinit(self: *Self) void {
            self.storage.deinit();
            self.* = undefined;
        }

        // The slot's elements, to be written under the invariant above.
        //
        // Neither the unwrap nor the alignment cast can fail: `init` returns an
        // error unless the buffer is mapped and both the base and the stride
        // carry `@alignOf(T)`, and nothing afterwards remaps it.
        pub fn slice(self: *const Self, frame: usize) []T {
            std.debug.assert(frame < self.frame_count);
            const mapping = self.storage.allocation.mappedBytes().?;
            const base = mapping[@intCast(frame * self.stride)..];
            const elements: [*]T = @ptrCast(@alignCast(base.ptr));
            return elements[0..self.count];
        }

        pub fn handle(self: *const Self) vk.Buffer {
            return self.storage.handle;
        }

        // The buffer itself, for the calls that take one rather than a handle.
        // A ring whose usage names a transfer source is copied from by
        // `Image.recordCopyFrom`, and that is what it asks for.
        pub fn storageBuffer(self: *const Self) *const buffer.Buffer {
            return &self.storage;
        }

        // What a bound descriptor adds to reach `frame`. A multiple of the
        // device's offset alignment by construction of `stride`.
        pub fn dynamicOffset(self: *const Self, frame: usize) u32 {
            std.debug.assert(frame < self.frame_count);
            return @intCast(frame * self.stride);
        }

        // One descriptor for the whole ring: it covers a single frame's payload
        // at offset zero, and the frame is chosen by the dynamic offset above.
        // The range is the payload rather than the stride, so the last frame's
        // window ends exactly at the end of the buffer.
        pub fn descriptor(self: *const Self) vk.DescriptorBufferInfo {
            return .{
                .buffer = self.storage.handle,
                .offset = 0,
                .range = @intCast(self.count * @sizeOf(T)),
            };
        }

        // The same window, in the form `Sets.writeBuffers` takes. It exists
        // because a ring is the one binding that is not a whole buffer, and a
        // caller working that out for itself would be reproducing the sentence
        // above about the payload and the stride.
        //
        // Borrows this ring: the result is valid for the call it is passed to
        // and not past it.
        pub fn source(self: *const Self) descriptors.BufferSource {
            return .{
                .buffer = &self.storage,
                .offset = 0,
                .range = @intCast(self.count * @sizeOf(T)),
            };
        }
    };
}
