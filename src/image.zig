const std = @import("std");
const vk = @import("vulkan");
const Buffer = @import("buffer.zig").Buffer;
const Context = @import("context.zig").Context;
const memory = @import("memory/allocator.zig");

// Every image here is single-sampled, optimal-tiling, single-layer and 2D. That
// is what the resource path uses and what the allocator's image pool is
// granularity-separated for, and it is also the shape of a render target.
// Multisampled and layered attachments are not this type.
const array_layers: u32 = 1;
const base_array_layer: u32 = 0;
const base_mip_level: u32 = 0;
const image_origin: vk.Offset3D = .{ .x = 0, .y = 0, .z = 0 };
const single_depth: u32 = 1;

// What an image holds, which decides its view's aspect and which format feature
// its usage is checked against. A named pair rather than a free `aspect` field:
// a view carrying both the colour and the depth aspect is not a state worth
// being able to express.
pub const Kind = enum {
    // Colour data, whether it is sampled, copied into, or rendered to.
    colour,

    // A depth attachment. Nothing here creates a combined depth-stencil view,
    // so the aspect is depth alone.
    depth,

    pub fn aspect(self: Kind) vk.ImageAspectFlags {
        return switch (self) {
            .colour => .{ .color_bit = true },
            .depth => .{ .depth_bit = true },
        };
    }
};

// A mip chain of 16 levels describes extents up to 32768 texels. Creation
// rejects anything deeper, which is what lets the copy path build its Vulkan
// regions in a stack array and stay allocation-free.
const max_mip_levels: u32 = 16;

// Context enables no feature-gated image usage, and init verifies each of these
// against the format's optimal tiling features.
const supported_usage = vk.ImageUsageFlags{
    .transfer_src_bit = true,
    .transfer_dst_bit = true,
    .sampled_bit = true,
    .color_attachment_bit = true,
    .depth_stencil_attachment_bit = true,
};

pub const InitError = error{
    AllocatorDeviceMismatch,
    EmptyUsage,
    ExtentLimitExceeded,
    InvalidExtent,
    InvalidMipLevels,
    UnsupportedFormatUsage,
    UnsupportedUsage,
} || vk.DeviceWrapper.CreateImageError ||
    vk.DeviceWrapper.CreateImageViewError ||
    memory.ImageAllocationError;

pub const CopyError = error{
    DifferentDevice,
    MipLevelOutOfRange,
    MissingTransferDestinationUsage,
    MissingTransferSourceUsage,
    NoRegions,
    RegionExtentMismatch,
    TooManyRegions,
};

pub const Config = struct {
    width: u32,
    height: u32,
    format: vk.Format,
    mip_levels: u32 = 1,
    usage: vk.ImageUsageFlags,
    kind: Kind = .colour,
};

// One mip level of a buffer-to-image copy. The fields a caller can get wrong are
// the ones that vary; aspect, layer, origin and tight packing are fixed by what
// this type is, so they are not restated at every call site.
//
// Vulkan specification, vkCmdCopyBufferToImage: bufferOffset must be a multiple
// of 4 and of the texel block size. The staging arena takes the alignment as an
// argument for exactly this reason.
pub const MipCopy = struct {
    buffer_offset: vk.DeviceSize,
    mip_level: u32,
    width: u32,
    height: u32,
};

// An explicit image memory dependency. No layout pair is translated into masks
// here: an inference table is where the masks silently stop matching the work
// being synchronized, and a wrong mask is invisible until it is a corrupted
// texture on one driver.
pub const LayoutTransition = struct {
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    src_stage_mask: vk.PipelineStageFlags2,
    src_access_mask: vk.AccessFlags2,
    dst_stage_mask: vk.PipelineStageFlags2,
    dst_access_mask: vk.AccessFlags2,

    // Receiving an upload into a fresh image. The source masks are empty because
    // there is no earlier access to wait for: the undefined layout discards
    // whatever the memory held.
    pub const to_transfer_destination: LayoutTransition = .{
        .old_layout = .undefined,
        .new_layout = .transfer_dst_optimal,
        .src_stage_mask = .{},
        .src_access_mask = .{},
        .dst_stage_mask = .{ .copy_bit = true },
        .dst_access_mask = .{ .transfer_write_bit = true },
    };

    // Handing an uploaded image to shaders. The stage is the caller's because it
    // is a property of the pipeline that will sample it, not of the image;
    // naming fragment here would silently under-synchronize a compute or vertex
    // read.
    pub fn toShaderRead(stage_mask: vk.PipelineStageFlags2) LayoutTransition {
        return .{
            .old_layout = .transfer_dst_optimal,
            .new_layout = .shader_read_only_optimal,
            .src_stage_mask = .{ .copy_bit = true },
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_stage_mask = stage_mask,
            .dst_access_mask = .{ .shader_sampled_read_bit = true },
        };
    }
};

pub const Image = struct {
    context: *const Context,
    memory_allocator: *memory.MemoryAllocator,
    handle: vk.Image,
    view: vk.ImageView,
    allocation: memory.Allocation,
    width: u32,
    height: u32,
    format: vk.Format,
    mip_levels: u32,
    usage: vk.ImageUsageFlags,
    kind: Kind,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        config: Config,
    ) InitError!Image {
        if (config.width == 0 or config.height == 0) return error.InvalidExtent;
        if (memory_allocator.context.device.handle != context.device.handle)
            return error.AllocatorDeviceMismatch;
        if (std.meta.eql(config.usage, vk.ImageUsageFlags{})) return error.EmptyUsage;
        if (!usageSupported(config.usage)) return error.UnsupportedUsage;

        const limit = context.properties.limits.max_image_dimension_2d;
        if (config.width > limit or config.height > limit)
            return error.ExtentLimitExceeded;

        // Vulkan specification, VkImageCreateInfo: mipLevels must be at least
        // one and must not exceed the number of levels the extent can halve
        // into, which is floor(log2(max(width, height))) + 1.
        if (config.mip_levels == 0 or config.mip_levels > max_mip_levels)
            return error.InvalidMipLevels;
        const largest_side = @max(config.width, config.height);
        const levels_for_extent = @as(u32, std.math.log2_int(u32, largest_side)) + 1;
        if (config.mip_levels > levels_for_extent) return error.InvalidMipLevels;

        try verifyFormatSupport(context, config.format, config.usage);

        const handle = try context.device.createImage(&.{
            .image_type = .@"2d",
            .format = config.format,
            .extent = .{
                .width = config.width,
                .height = config.height,
                .depth = single_depth,
            },
            .mip_levels = config.mip_levels,
            .array_layers = array_layers,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = config.usage,
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);
        errdefer context.device.destroyImage(handle, null);

        const allocation = try memory_allocator.allocateOptimalImage(handle);
        errdefer memory_allocator.free(allocation) catch |err| switch (err) {
            error.InvalidAllocation => @panic("allocator rejected its own image allocation"),
        };

        const view = try context.device.createImageView(&.{
            .image = handle,
            .view_type = .@"2d",
            .format = config.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = config.kind.aspect(),
                .base_mip_level = base_mip_level,
                .level_count = config.mip_levels,
                .base_array_layer = base_array_layer,
                .layer_count = array_layers,
            },
        }, null);

        return .{
            .context = context,
            .memory_allocator = memory_allocator,
            .handle = handle,
            .view = view,
            .allocation = allocation,
            .width = config.width,
            .height = config.height,
            .format = config.format,
            .mip_levels = config.mip_levels,
            .usage = config.usage,
            .kind = config.kind,
        };
    }

    // Vulkan specification, vkDestroyImage and vkDestroyImageView: submitted work
    // using either must have completed. The view goes first because it names the
    // image.
    pub fn deinit(self: *Image) void {
        self.context.device.destroyImageView(self.view, null);
        self.context.device.destroyImage(self.handle, null);
        self.memory_allocator.free(self.allocation) catch |err| switch (err) {
            error.InvalidAllocation => @panic("image owns an invalid memory allocation"),
        };
        self.* = undefined;
    }

    // The barrier covers every mip level and the single layer, because a partial
    // transition has no user here and a range that does not match the copy is a
    // silent hazard.
    pub fn recordLayoutTransition(
        self: *const Image,
        command_buffer: vk.CommandBuffer,
        transition: LayoutTransition,
    ) void {
        const barriers = [_]vk.ImageMemoryBarrier2{.{
            .src_stage_mask = transition.src_stage_mask,
            .src_access_mask = transition.src_access_mask,
            .dst_stage_mask = transition.dst_stage_mask,
            .dst_access_mask = transition.dst_access_mask,
            .old_layout = transition.old_layout,
            .new_layout = transition.new_layout,
            // No queue-family ownership transfer: images are created with
            // exclusive sharing and stay on one family.
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = self.handle,
            .subresource_range = .{
                .aspect_mask = self.kind.aspect(),
                .base_mip_level = base_mip_level,
                .level_count = self.mip_levels,
                .base_array_layer = base_array_layer,
                .layer_count = array_layers,
            },
        }};
        self.context.device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = barriers.len,
            .p_image_memory_barriers = &barriers,
        });
    }

    // Records one vkCmdCopyBufferToImage covering the given mip levels. The image
    // must already be in transfer_dst_optimal, which the caller establishes with
    // a transition; this records no barrier of its own because batching several
    // uploads behind one barrier is the point of recording rather than
    // submitting.
    //
    // regions is borrowed for the duration of the call. Vulkan copies the
    // structures during recording, so it need not outlive it.
    pub fn recordCopyFrom(
        self: *const Image,
        source: *const Buffer,
        command_buffer: vk.CommandBuffer,
        regions: []const MipCopy,
    ) CopyError!void {
        if (self.context.device.handle != source.context.device.handle)
            return error.DifferentDevice;
        if (!source.usage.transfer_src_bit) return error.MissingTransferSourceUsage;
        if (!self.usage.transfer_dst_bit) return error.MissingTransferDestinationUsage;
        if (regions.len == 0) return error.NoRegions;
        // Creation caps the level count, so one region per level fits the stack
        // array and this path allocates nothing.
        if (regions.len > max_mip_levels) return error.TooManyRegions;

        var scratch: [max_mip_levels]vk.BufferImageCopy = undefined;
        for (regions, scratch[0..regions.len]) |region, *out| {
            if (region.mip_level >= self.mip_levels) return error.MipLevelOutOfRange;
            if (region.width != mipExtent(self.width, region.mip_level) or
                region.height != mipExtent(self.height, region.mip_level))
                return error.RegionExtentMismatch;

            out.* = .{
                .buffer_offset = region.buffer_offset,
                // Zero means tightly packed to the image extent. Vulkan
                // specification, VkBufferImageCopy: bufferRowLength and
                // bufferImageHeight of zero make the buffer rows match
                // imageExtent, which is how every uploaded mip is laid out here.
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_subresource = .{
                    .aspect_mask = self.kind.aspect(),
                    .mip_level = region.mip_level,
                    .base_array_layer = base_array_layer,
                    .layer_count = array_layers,
                },
                .image_offset = image_origin,
                .image_extent = .{
                    .width = region.width,
                    .height = region.height,
                    .depth = single_depth,
                },
            };
        }

        self.context.device.cmdCopyBufferToImage(
            command_buffer,
            source.handle,
            self.handle,
            .transfer_dst_optimal,
            scratch[0..regions.len],
        );
    }
};

// Vulkan specification, Image Mip Level Sizing: each level halves the previous
// one and clamps at one texel.
pub fn mipExtent(base: u32, level: u32) u32 {
    return @max(1, base >> @intCast(level));
}

// One check per supported usage bit, rather than a table mapping usage to
// features. Adding a usage to `supported_usage` without its feature check is
// then a visible omission instead of a silent pass.
//
// Takes the features rather than a device, so which usage needs which feature
// is decided where it can be exercised without one.
// Whether this module creates an image with that usage at all, as distinct from
// whether a given format can serve it. Feature-gated and multisample-only bits
// are the ones that fail here.
pub fn usageSupported(usage: vk.ImageUsageFlags) bool {
    return supported_usage.contains(usage);
}

pub fn formatSupports(usage: vk.ImageUsageFlags, features: vk.FormatFeatureFlags) bool {
    if (usage.transfer_src_bit and !features.transfer_src_bit) return false;
    if (usage.transfer_dst_bit and !features.transfer_dst_bit) return false;
    if (usage.sampled_bit and !features.sampled_image_bit) return false;
    if (usage.color_attachment_bit and !features.color_attachment_bit) return false;
    if (usage.depth_stencil_attachment_bit and !features.depth_stencil_attachment_bit) return false;
    return true;
}

fn verifyFormatSupport(
    context: *const Context,
    format: vk.Format,
    usage: vk.ImageUsageFlags,
) error{UnsupportedFormatUsage}!void {
    const properties = context.instance.getPhysicalDeviceFormatProperties(
        context.physical_device,
        format,
    );
    if (!formatSupports(usage, properties.optimal_tiling_features))
        return error.UnsupportedFormatUsage;
}
