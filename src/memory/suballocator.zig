const std = @import("std");

const Allocator = std.mem.Allocator;

// Byte ranges keep the width of vk.DeviceSize, which the generated bindings
// declare as u64, so conversion at the Vulkan boundary cannot truncate an offset
// or an allocation size.
const ByteOffset = u64;
const ByteSize = u64;

// Placement can split one free range into two, and a later release can require
// one additional range. Reserving both entries makes every release infallible.
const free_range_metadata_slack: usize = 2;
const initial_free_range_capacity: usize = 1 + free_range_metadata_slack;
const initial_active_capacity: usize = 1;

pub const InitError = Allocator.Error || error{InvalidSize};
pub const AllocateError = Allocator.Error || error{SizeOverflow};
pub const FreeError = error{InvalidAllocation};

const Range = struct {
    offset: ByteOffset,
    size: ByteSize,

    fn end(self: Range) ByteOffset {
        return self.offset + self.size;
    }
};

const ActiveAllocation = struct {
    id: u64,
    range: Range,
};

pub const Suballocator = struct {
    free_ranges: std.ArrayList(Range),
    active_allocations: std.ArrayList(ActiveAllocation),

    pub fn init(allocator: Allocator, size: ByteSize) InitError!Suballocator {
        if (size == 0) return error.InvalidSize;

        var free_ranges = try std.ArrayList(Range).initCapacity(
            allocator,
            initial_free_range_capacity,
        );
        errdefer free_ranges.deinit(allocator);
        free_ranges.appendAssumeCapacity(.{ .offset = 0, .size = size });

        const active_allocations = try std.ArrayList(ActiveAllocation).initCapacity(
            allocator,
            initial_active_capacity,
        );
        return .{
            .free_ranges = free_ranges,
            .active_allocations = active_allocations,
        };
    }

    pub fn deinit(self: *Suballocator, allocator: Allocator) void {
        self.active_allocations.deinit(allocator);
        self.free_ranges.deinit(allocator);
        self.* = undefined;
    }

    pub fn hasAllocations(self: *const Suballocator) bool {
        return self.active_allocations.items.len != 0;
    }

    pub fn allocate(
        self: *Suballocator,
        allocator: Allocator,
        size: ByteSize,
        alignment: ByteSize,
        allocation_id: u64,
    ) AllocateError!?ByteOffset {
        if (size == 0 or alignment == 0 or !std.math.isPowerOfTwo(alignment))
            return error.SizeOverflow;

        for (self.free_ranges.items, 0..) |range, index| {
            const offset = alignForward(range.offset, alignment) catch continue;
            const allocation_end = std.math.add(u64, offset, size) catch continue;
            if (allocation_end > range.end()) continue;

            // Reserve one future free-list entry for every live allocation.
            // Releasing memory must remain infallible in any release order.
            const live_margin = std.math.add(
                usize,
                self.active_allocations.items.len,
                free_range_metadata_slack,
            ) catch return error.SizeOverflow;
            const metadata_capacity = std.math.add(
                usize,
                self.free_ranges.items.len,
                live_margin,
            ) catch return error.SizeOverflow;
            try self.free_ranges.ensureTotalCapacity(allocator, metadata_capacity);
            try self.active_allocations.ensureUnusedCapacity(allocator, 1);

            const prefix_size = offset - range.offset;
            const suffix_size = range.end() - allocation_end;
            if (prefix_size == 0 and suffix_size == 0) {
                _ = self.free_ranges.orderedRemove(index);
            } else if (prefix_size == 0) {
                self.free_ranges.items[index] = .{
                    .offset = allocation_end,
                    .size = suffix_size,
                };
            } else {
                self.free_ranges.items[index].size = prefix_size;
                if (suffix_size != 0) self.free_ranges.insertAssumeCapacity(
                    index + 1,
                    .{ .offset = allocation_end, .size = suffix_size },
                );
            }
            self.active_allocations.appendAssumeCapacity(.{
                .id = allocation_id,
                .range = .{ .offset = offset, .size = size },
            });
            return offset;
        }
        return null;
    }

    pub fn free(
        self: *Suballocator,
        allocation_id: u64,
        offset: ByteOffset,
        size: ByteSize,
    ) FreeError!void {
        var active_index: ?usize = null;
        for (self.active_allocations.items, 0..) |active, index| {
            if (active.id == allocation_id and
                active.range.offset == offset and
                active.range.size == size)
            {
                active_index = index;
                break;
            }
        }
        const remove_index = active_index orelse return error.InvalidAllocation;
        const released = self.active_allocations.items[remove_index].range;

        var insert_index: usize = 0;
        while (insert_index < self.free_ranges.items.len and
            self.free_ranges.items[insert_index].offset < released.offset) : (insert_index += 1)
        {}

        const previous = if (insert_index > 0)
            &self.free_ranges.items[insert_index - 1]
        else
            null;
        const next = if (insert_index < self.free_ranges.items.len)
            &self.free_ranges.items[insert_index]
        else
            null;
        if (previous) |left|
            if (left.end() > released.offset) return error.InvalidAllocation;
        if (next) |right|
            if (released.end() > right.offset) return error.InvalidAllocation;

        const joins_previous = if (previous) |left|
            left.end() == released.offset
        else
            false;
        const joins_next = if (next) |right|
            released.end() == right.offset
        else
            false;
        if (joins_previous and joins_next) {
            self.free_ranges.items[insert_index - 1].size =
                next.?.end() - previous.?.offset;
            _ = self.free_ranges.orderedRemove(insert_index);
        } else if (joins_previous) {
            self.free_ranges.items[insert_index - 1].size += released.size;
        } else if (joins_next) {
            self.free_ranges.items[insert_index] = .{
                .offset = released.offset,
                .size = released.size + next.?.size,
            };
        } else {
            self.free_ranges.insertAssumeCapacity(insert_index, released);
        }
        _ = self.active_allocations.orderedRemove(remove_index);
    }
};

fn alignForward(
    value: ByteOffset,
    alignment: ByteSize,
) error{SizeOverflow}!ByteOffset {
    const sum = std.math.add(ByteOffset, value, alignment - 1) catch
        return error.SizeOverflow;
    return sum & ~(alignment - 1);
}
