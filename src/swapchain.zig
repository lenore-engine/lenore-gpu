const std = @import("std");
const platform = @import("lenore-platform");
const vk = @import("vulkan");
const Context = @import("context.zig").Context;

const Allocator = std.mem.Allocator;
const log = std.log.scoped(.vulkan);

pub const PresentModePreference = enum {
    fifo,
    mailbox,
    immediate,
};

pub const InitError = error{
    InvalidSurfaceDimensions,
    NoAvailablePresentModes,
    NoAvailableSurfaceFormats,
    NoSupportedCompositeAlpha,
    UnsupportedSurfaceUsage,
} || Allocator.Error ||
    vk.InstanceWrapper.GetPhysicalDeviceSurfaceCapabilitiesKHRError ||
    vk.InstanceWrapper.GetPhysicalDeviceSurfaceFormatsAllocKHRError ||
    vk.InstanceWrapper.GetPhysicalDeviceSurfacePresentModesAllocKHRError ||
    vk.DeviceWrapper.CreateSwapchainKHRError ||
    vk.DeviceWrapper.GetSwapchainImagesAllocKHRError ||
    vk.DeviceWrapper.CreateImageViewError ||
    vk.DeviceWrapper.CreateSemaphoreError;

pub const AcquireError = error{
    ImageUnavailable,
    UnexpectedAcquireResult,
} || vk.DeviceWrapper.AcquireNextImageKHRError;

pub const PresentError = error{
    InvalidImageIndex,
    UnexpectedPresentResult,
} || vk.DeviceWrapper.QueuePresentKHRError;

pub const ImageIndexError = error{InvalidImageIndex};

pub const Swapchain = struct {
    pub const PresentState = enum {
        optimal,
        suboptimal,
    };

    pub const AcquireResult = struct {
        image_index: u32,
        state: PresentState,
    };

    context: *const Context,
    allocator: Allocator,
    surface_format: vk.SurfaceFormatKHR,
    // Presentation policy belongs to the application. Retaining it here makes
    // recreation reproduce the same choice without another source of truth.
    present_preference: PresentModePreference,
    present_mode: vk.PresentModeKHR,
    extent: vk.Extent2D,
    handle: vk.SwapchainKHR,
    images: []SwapImage,

    pub fn init(
        context: *const Context,
        allocator: Allocator,
        requested_extent: platform.Extent2D,
        preference: PresentModePreference,
    ) InitError!Swapchain {
        return initRecycle(
            context,
            allocator,
            requested_extent,
            .null_handle,
            preference,
        );
    }

    fn initRecycle(
        context: *const Context,
        allocator: Allocator,
        requested_extent: platform.Extent2D,
        old_handle: vk.SwapchainKHR,
        preference: PresentModePreference,
    ) InitError!Swapchain {
        const capabilities = try context.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(
            context.physical_device,
            context.surface,
        );
        const extent = actualExtent(capabilities, requested_extent);
        if (extent.width == 0 or extent.height == 0)
            return error.InvalidSurfaceDimensions;

        const surface_format = try findSurfaceFormat(context, allocator);
        const present_mode = try findPresentMode(context, allocator, preference);
        const composite_alpha = try findCompositeAlpha(capabilities);
        log.info("present mode: {t}", .{present_mode});

        const image_usage = vk.ImageUsageFlags{
            .color_attachment_bit = true,
            .transfer_dst_bit = true,
        };
        if (!capabilities.supported_usage_flags.contains(image_usage))
            return error.UnsupportedSurfaceUsage;

        var image_count = capabilities.min_image_count +| 1;
        if (capabilities.max_image_count > 0)
            image_count = @min(image_count, capabilities.max_image_count);

        const queue_families = [_]u32{
            context.graphics_queue.family,
            context.present_queue.family,
        };
        const concurrent = context.graphics_queue.family != context.present_queue.family;
        const handle = try context.device.createSwapchainKHR(&.{
            .surface = context.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = image_usage,
            .image_sharing_mode = if (concurrent) .concurrent else .exclusive,
            .queue_family_index_count = if (concurrent) queue_families.len else 0,
            .p_queue_family_indices = if (concurrent) &queue_families else null,
            .pre_transform = capabilities.current_transform,
            .composite_alpha = composite_alpha,
            .present_mode = present_mode,
            .clipped = .true,
            .old_swapchain = old_handle,
        }, null);
        errdefer context.device.destroySwapchainKHR(handle, null);

        const images = try initSwapchainImages(
            context,
            handle,
            surface_format.format,
            allocator,
        );
        errdefer deinitSwapchainImages(context, allocator, images);

        return .{
            .context = context,
            .allocator = allocator,
            .surface_format = surface_format,
            .present_preference = preference,
            .present_mode = present_mode,
            .extent = extent,
            .handle = handle,
            .images = images,
        };
    }

    // Vulkan specification, vkDestroySwapchainKHR: pending operations that
    // access the swapchain must complete before destruction.
    pub fn deinit(self: Swapchain) void {
        deinitSwapchainImages(self.context, self.allocator, self.images);
        self.context.device.destroySwapchainKHR(self.handle, null);
    }

    // The same vkDestroySwapchainKHR lifetime requirement applies here. The
    // caller must complete all work referring to the current images first.
    pub fn recreate(
        self: *Swapchain,
        requested_extent: platform.Extent2D,
    ) InitError!void {
        const replacement = try initRecycle(
            self.context,
            self.allocator,
            requested_extent,
            self.handle,
            self.present_preference,
        );

        // Ownership changes only after the replacement and all of its image
        // state exist, so a failed recreation leaves the current object intact.
        const old = self.*;
        self.* = replacement;
        old.deinit();
    }

    pub fn acquireNextImage(
        self: *const Swapchain,
        image_acquired: vk.Semaphore,
    ) AcquireError!AcquireResult {
        const result = try self.context.device.acquireNextImageKHR(
            self.handle,
            std.math.maxInt(u64),
            image_acquired,
            .null_handle,
        );

        return .{
            .image_index = result.image_index,
            .state = switch (result.result) {
                .success => .optimal,
                .suboptimal_khr => .suboptimal,
                .not_ready, .timeout => return error.ImageUnavailable,
                else => return error.UnexpectedAcquireResult,
            },
        };
    }

    // Khronos Vulkan Guide, "Swapchain Semaphore Reuse": render-finished
    // semaphores are indexed by swapchain image because reacquisition proves
    // the presentation engine has finished waiting on that image's semaphore.
    pub fn present(
        self: *const Swapchain,
        image_index: u32,
    ) PresentError!PresentState {
        if (image_index >= self.images.len) return error.InvalidImageIndex;

        const wait_semaphores = [_]vk.Semaphore{
            self.images[image_index].render_finished,
        };
        const swapchains = [_]vk.SwapchainKHR{self.handle};
        const image_indices = [_]u32{image_index};
        const result = try self.context.device.queuePresentKHR(
            self.context.present_queue.handle,
            &.{
                .wait_semaphore_count = wait_semaphores.len,
                .p_wait_semaphores = &wait_semaphores,
                .swapchain_count = swapchains.len,
                .p_swapchains = &swapchains,
                .p_image_indices = &image_indices,
            },
        );

        return switch (result) {
            .success => .optimal,
            .suboptimal_khr => .suboptimal,
            else => error.UnexpectedPresentResult,
        };
    }

    // Records a clear of one swapchain image and leaves it ready to present.
    //
    // Three commands, and the two barriers are what make it correct rather than
    // the clear itself. The image has to reach the layout a transfer write
    // requires and then the one presentation requires.
    //
    // The first barrier transitions from undefined, which is a choice rather
    // than a description: an acquired image holds whatever layout it was left
    // in, present_src_khr for one that has been presented before. Vulkan
    // specification, Image Layouts: a transition out of undefined discards the
    // image's contents, which is exactly right when the next command overwrites
    // every texel, and it costs nothing to say so.
    //
    // Its source masks are empty because no earlier command in this frame
    // touched the image. The second barrier waits for the clear and hands the
    // image to the presentation engine, whose access no pipeline stage names,
    // which is why its destination masks are empty in turn.
    pub fn recordClear(
        self: *const Swapchain,
        command_buffer: vk.CommandBuffer,
        image_index: u32,
        colour: [4]f32,
    ) ImageIndexError!void {
        if (image_index >= self.images.len) return error.InvalidImageIndex;
        const image = self.images[image_index].image;
        const whole_image: vk.ImageSubresourceRange = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };

        self.recordBarrier(command_buffer, .{
            .src_stage_mask = .{},
            .src_access_mask = .{},
            .dst_stage_mask = .{ .clear_bit = true },
            .dst_access_mask = .{ .transfer_write_bit = true },
            .old_layout = .undefined,
            .new_layout = .transfer_dst_optimal,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresource_range = whole_image,
        });

        self.context.device.cmdClearColorImage(
            command_buffer,
            image,
            .transfer_dst_optimal,
            &.{ .float_32 = colour },
            &.{whole_image},
        );

        self.recordBarrier(command_buffer, .{
            .src_stage_mask = .{ .clear_bit = true },
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_stage_mask = .{},
            .dst_access_mask = .{},
            .old_layout = .transfer_dst_optimal,
            .new_layout = .present_src_khr,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresource_range = whole_image,
        });
    }

    fn recordBarrier(
        self: *const Swapchain,
        command_buffer: vk.CommandBuffer,
        barrier: vk.ImageMemoryBarrier2,
    ) void {
        const barriers = [_]vk.ImageMemoryBarrier2{barrier};
        self.context.device.cmdPipelineBarrier2(command_buffer, &.{
            .image_memory_barrier_count = barriers.len,
            .p_image_memory_barriers = &barriers,
        });
    }

    pub fn currentExtent(self: *const Swapchain) platform.Extent2D {
        return .{ .width = self.extent.width, .height = self.extent.height };
    }

    // Whether the swapchain still matches the surface. A driver is not obliged
    // to report a resize as suboptimal, so the extent is the reliable signal and
    // that flag is the additional one.
    pub fn matchesExtent(self: *const Swapchain, extent: platform.Extent2D) bool {
        return self.extent.width == extent.width and self.extent.height == extent.height;
    }

    pub fn renderFinishedSemaphore(
        self: *const Swapchain,
        image_index: u32,
    ) ImageIndexError!vk.Semaphore {
        if (image_index >= self.images.len) return error.InvalidImageIndex;
        return self.images[image_index].render_finished;
    }
};

pub const SwapImage = struct {
    image: vk.Image,
    view: vk.ImageView,
    render_finished: vk.Semaphore,

    fn init(
        context: *const Context,
        image: vk.Image,
        format: vk.Format,
    ) (vk.DeviceWrapper.CreateImageViewError ||
        vk.DeviceWrapper.CreateSemaphoreError)!SwapImage {
        const view = try context.device.createImageView(&.{
            .image = image,
            .view_type = .@"2d",
            .format = format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer context.device.destroyImageView(view, null);

        const render_finished = try context.device.createSemaphore(&.{}, null);
        errdefer context.device.destroySemaphore(render_finished, null);

        return .{
            .image = image,
            .view = view,
            .render_finished = render_finished,
        };
    }

    fn deinit(self: SwapImage, context: *const Context) void {
        context.device.destroySemaphore(self.render_finished, null);
        context.device.destroyImageView(self.view, null);
    }
};

fn initSwapchainImages(
    context: *const Context,
    swapchain: vk.SwapchainKHR,
    format: vk.Format,
    allocator: Allocator,
) (Allocator.Error || vk.DeviceWrapper.GetSwapchainImagesAllocKHRError ||
    vk.DeviceWrapper.CreateImageViewError ||
    vk.DeviceWrapper.CreateSemaphoreError)![]SwapImage {
    const images = try context.device.getSwapchainImagesAllocKHR(swapchain, allocator);
    defer allocator.free(images);

    const swapchain_images = try allocator.alloc(SwapImage, images.len);
    errdefer allocator.free(swapchain_images);

    var initialized: usize = 0;
    errdefer for (swapchain_images[0..initialized]) |image| image.deinit(context);
    for (images) |image| {
        swapchain_images[initialized] = try .init(context, image, format);
        initialized += 1;
    }
    return swapchain_images;
}

fn deinitSwapchainImages(
    context: *const Context,
    allocator: Allocator,
    images: []SwapImage,
) void {
    for (images) |image| image.deinit(context);
    allocator.free(images);
}

fn findSurfaceFormat(
    context: *const Context,
    allocator: Allocator,
) (Allocator.Error ||
    vk.InstanceWrapper.GetPhysicalDeviceSurfaceFormatsAllocKHRError ||
    error{NoAvailableSurfaceFormats})!vk.SurfaceFormatKHR {
    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };
    const formats = try context.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(
        context.physical_device,
        context.surface,
        allocator,
    );
    defer allocator.free(formats);

    for (formats) |format|
        if (std.meta.eql(format, preferred)) return preferred;
    if (formats.len == 0) return error.NoAvailableSurfaceFormats;
    return formats[0];
}

fn findPresentMode(
    context: *const Context,
    allocator: Allocator,
    preference: PresentModePreference,
) (Allocator.Error ||
    vk.InstanceWrapper.GetPhysicalDeviceSurfacePresentModesAllocKHRError ||
    error{NoAvailablePresentModes})!vk.PresentModeKHR {
    const modes = try context.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(
        context.physical_device,
        context.surface,
        allocator,
    );
    defer allocator.free(modes);

    const preferred: []const vk.PresentModeKHR = switch (preference) {
        .fifo => &.{.fifo_khr},
        .mailbox => &.{ .mailbox_khr, .fifo_khr },
        .immediate => &.{ .immediate_khr, .mailbox_khr, .fifo_khr },
    };
    for (preferred) |candidate|
        if (std.mem.indexOfScalar(vk.PresentModeKHR, modes, candidate) != null)
            return candidate;

    // Vulkan specification, VkPresentModeKHR: FIFO support is required. A
    // missing match therefore reports a broken surface contract.
    return error.NoAvailablePresentModes;
}

fn findCompositeAlpha(
    capabilities: vk.SurfaceCapabilitiesKHR,
) error{NoSupportedCompositeAlpha}!vk.CompositeAlphaFlagsKHR {
    const preferred = [_]vk.CompositeAlphaFlagsKHR{
        .{ .opaque_bit_khr = true },
        .{ .pre_multiplied_bit_khr = true },
        .{ .post_multiplied_bit_khr = true },
        .{ .inherit_bit_khr = true },
    };
    for (preferred) |candidate|
        if (capabilities.supported_composite_alpha.contains(candidate))
            return candidate;
    return error.NoSupportedCompositeAlpha;
}

fn actualExtent(
    capabilities: vk.SurfaceCapabilitiesKHR,
    requested: platform.Extent2D,
) vk.Extent2D {
    if (capabilities.current_extent.width != std.math.maxInt(u32))
        return capabilities.current_extent;
    return .{
        .width = std.math.clamp(
            requested.width,
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        .height = std.math.clamp(
            requested.height,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    };
}
