const std = @import("std");
const build_options = @import("build_options");
const platform = @import("lenore-platform");
const vk = @import("vulkan");

const Loader = @import("loader.zig").Loader;
const timing = @import("timing.zig");

const Allocator = std.mem.Allocator;
const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;
const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;
const log = std.log.scoped(.vulkan);

const validation_layer: [*:0]const u8 = "VK_LAYER_KHRONOS_validation";
const required_device_extensions = [_][*:0]const u8{
    vk.extensions.khr_swapchain.name,
};

// Every window system the platform can hand back, because the union it returns
// declares them all and this switch has to be exhaustive on any target.
const SurfaceError = InstanceWrapper.CreateWaylandSurfaceKHRError ||
    InstanceWrapper.CreateWin32SurfaceKHRError;
const LayerQueryError = BaseWrapper.EnumerateInstanceLayerPropertiesAllocError;
const DeviceQueryError = Allocator.Error ||
    InstanceWrapper.EnumeratePhysicalDevicesAllocError ||
    InstanceWrapper.EnumerateDeviceExtensionPropertiesAllocError ||
    InstanceWrapper.GetPhysicalDeviceSurfaceSupportKHRError ||
    InstanceWrapper.GetPhysicalDeviceSurfaceFormatsKHRError ||
    InstanceWrapper.GetPhysicalDeviceSurfacePresentModesKHRError;
const DevicePickError = DeviceQueryError || error{NoSuitableDevice};

pub const InitError = error{
    MissingValidationLayer,
    NoSuitableDevice,
} || Loader.Error || Allocator.Error || BaseWrapper.CreateInstanceError ||
    LayerQueryError || SurfaceError || DeviceQueryError ||
    InstanceWrapper.CreateDebugUtilsMessengerEXTError || InstanceWrapper.CreateDeviceError;

pub const MemoryTypeError = error{NoSuitableMemoryType};

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(device: Device, family: u32) Queue {
        return .{
            .handle = device.getDeviceQueue(family, 0),
            .family = family,
        };
    }
};

pub const QueueAllocation = struct {
    // Also the family every dispatch is recorded on. vk.xml gives vkCmdDispatch
    // `queues="VK_QUEUE_COMPUTE_BIT"`, and the morph prepass records into the
    // same command buffer as the draws it feeds, so the two capabilities have to
    // meet on one family rather than merely both be present on the device.
    graphics_family: u32,
    present_family: u32,
};

// What one queue family offers, as the choice below reads it.
pub const QueueSupport = struct {
    graphics: bool,
    compute: bool,
    present: bool,
    // How wide a timestamp written on this family is. Vulkan specification,
    // VkQueueFamilyProperties: timestampValidBits is zero where timestamps are
    // not supported at all, and between 36 and 64 where they are. Not part of
    // choosing a family, and carried here because this is where the properties
    // are read.
    timestamp_valid_bits: u32 = 0,
};

// Which families a device is taken with, or null when it offers no usable pair.
//
// Split from the query for the reason the other validators in this module are:
// the rule is a scan over flags, and a device and a surface stand between it and
// a test. Every path through it has been wrong at some point in some engine: the
// shared family, the split pair, and the graphics family that cannot dispatch.
//
// One family serving both is preferred, because a shared family needs no
// ownership transfer between the draw and the present.
pub fn chooseQueues(families: []const QueueSupport) ?QueueAllocation {
    var graphics_family: ?u32 = null;
    var present_family: ?u32 = null;

    for (families, 0..) |support, index| {
        const family: u32 = @intCast(index);
        const usable_graphics = support.graphics and support.compute;
        if (usable_graphics and support.present)
            return .{ .graphics_family = family, .present_family = family };
        if (graphics_family == null and usable_graphics) graphics_family = family;
        if (present_family == null and support.present) present_family = family;
    }

    return .{
        .graphics_family = graphics_family orelse return null,
        .present_family = present_family orelse return null,
    };
}

const DeviceExtensions = struct {
    memory_budget: bool,
    // VK_KHR_pipeline_executable_properties. Named for what it delivers rather
    // than for the extension: what comes back is per-shader register counts and
    // code size. `pipeline_statistics` below is the unrelated core feature that
    // counts primitives and invocations through a query pool.
    shader_statistics: bool,
};

const DeviceCandidate = struct {
    physical_device: vk.PhysicalDevice,
    properties: vk.PhysicalDeviceProperties,
    extensions: DeviceExtensions,
    queues: QueueAllocation,
    // Of the graphics family, which is the only one anything is recorded on.
    timestamp_valid_bits: u32,
    pipeline_statistics: bool,
    max_buffer_size: vk.DeviceSize,
};

pub const Context = struct {
    pub const CommandBuffer = vk.CommandBufferProxy;

    allocator: Allocator,
    loader: Loader,
    base_wrapper: BaseWrapper,
    instance: Instance,
    debug_messenger: ?vk.DebugUtilsMessengerEXT,
    surface: vk.SurfaceKHR,
    physical_device: vk.PhysicalDevice,
    properties: vk.PhysicalDeviceProperties,
    memory_properties: vk.PhysicalDeviceMemoryProperties,
    max_sampler_anisotropy: f32,
    max_buffer_size: vk.DeviceSize,
    device: Device,
    graphics_queue: Queue,
    present_queue: Queue,
    pipeline_statistics_enabled: bool,
    memory_budget_enabled: bool,
    // Whether a pipeline can be asked what the driver compiled it into. The
    // answer is only available for a pipeline created with the capture flag,
    // which is what `pipeline.Config.capture_statistics` asks for, so this
    // says the query exists and not that any pipeline carries an answer.
    shader_statistics_enabled: bool,
    // What a timestamp written on the graphics queue is worth, as the device
    // reports it. Zero is a queue that cannot carry one.
    timestamp_valid_bits: u32,

    // Nanoseconds per timestamp tick, and what a device pass duration is scaled
    // by. Vulkan specification, VkPhysicalDeviceLimits::timestampPeriod.
    pub fn timestampSupport(self: *const Context) timing.Support {
        return .{
            .valid_bits = self.timestamp_valid_bits,
            .period_ns = self.properties.limits.timestamp_period,
        };
    }

    // Initialization may return the Context by value: every proxy references a
    // separately allocated wrapper, and no callback retains the Context address.
    pub fn init(
        allocator: Allocator,
        application_name: [:0]const u8,
        native_handles: platform.NativeHandles,
    ) InitError!Context {
        var loader = try Loader.open();
        errdefer loader.close();
        const base_wrapper = BaseWrapper.load(try loader.getInstanceProcAddr());

        var extension_names: std.ArrayList([*:0]const u8) = .empty;
        defer extension_names.deinit(allocator);
        try appendSurfaceExtensions(&extension_names, allocator, native_handles);
        if (build_options.enable_validation)
            try extension_names.append(allocator, vk.extensions.ext_debug_utils.name);

        var enabled_layers: []const [*:0]const u8 = &.{};
        if (build_options.enable_validation) {
            if (!try hasInstanceLayer(&base_wrapper, allocator, std.mem.span(validation_layer))) {
                log.err("Debug builds require {s}", .{validation_layer});
                return error.MissingValidationLayer;
            }
            enabled_layers = &.{validation_layer};
        }

        const raw_instance = try base_wrapper.createInstance(&.{
            .p_application_info = &.{
                .p_application_name = application_name,
                .application_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
                .p_engine_name = "Lenore",
                .engine_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
                .api_version = @bitCast(vk.API_VERSION_1_3),
            },
            .enabled_layer_count = @intCast(enabled_layers.len),
            .pp_enabled_layer_names = enabled_layers.ptr,
            .enabled_extension_count = @intCast(extension_names.items.len),
            .pp_enabled_extension_names = extension_names.items.ptr,
        }, null);

        const loaded_instance_wrapper = InstanceWrapper.load(
            raw_instance,
            base_wrapper.dispatch.vkGetInstanceProcAddr.?,
        );
        const instance_wrapper = allocator.create(InstanceWrapper) catch |err| {
            loaded_instance_wrapper.destroyInstance(raw_instance, null);
            return err;
        };
        instance_wrapper.* = loaded_instance_wrapper;
        const instance = Instance.init(raw_instance, instance_wrapper);
        errdefer {
            instance.destroyInstance(null);
            allocator.destroy(instance_wrapper);
        }

        const debug_messenger: ?vk.DebugUtilsMessengerEXT = if (build_options.enable_validation)
            try instance.createDebugUtilsMessengerEXT(&.{
                .message_severity = .{
                    .warning_bit_ext = true,
                    .error_bit_ext = true,
                },
                .message_type = .{
                    .general_bit_ext = true,
                    .validation_bit_ext = true,
                    .performance_bit_ext = true,
                },
                .pfn_user_callback = &debugCallback,
            }, null)
        else
            null;
        errdefer if (debug_messenger) |messenger|
            instance.destroyDebugUtilsMessengerEXT(messenger, null);

        const surface = try createSurface(instance, native_handles);
        errdefer instance.destroySurfaceKHR(surface, null);

        const candidate = try pickIntegratedDevice(instance, allocator, surface);
        const raw_device = try createDevice(instance, candidate);
        const loaded_device_wrapper = DeviceWrapper.load(
            raw_device,
            instance.wrapper.dispatch.vkGetDeviceProcAddr.?,
        );
        const device_wrapper = allocator.create(DeviceWrapper) catch |err| {
            loaded_device_wrapper.destroyDevice(raw_device, null);
            return err;
        };
        device_wrapper.* = loaded_device_wrapper;
        const device = Device.init(raw_device, device_wrapper);
        errdefer {
            device.destroyDevice(null);
            allocator.destroy(device_wrapper);
        }

        return .{
            .allocator = allocator,
            .loader = loader,
            .base_wrapper = base_wrapper,
            .instance = instance,
            .debug_messenger = debug_messenger,
            .surface = surface,
            .physical_device = candidate.physical_device,
            .properties = candidate.properties,
            .memory_properties = instance.getPhysicalDeviceMemoryProperties(candidate.physical_device),
            .max_sampler_anisotropy = candidate.properties.limits.max_sampler_anisotropy,
            .max_buffer_size = candidate.max_buffer_size,
            .device = device,
            .graphics_queue = .init(device, candidate.queues.graphics_family),
            .present_queue = .init(device, candidate.queues.present_family),
            .pipeline_statistics_enabled = candidate.pipeline_statistics,
            .memory_budget_enabled = candidate.extensions.memory_budget,
            .shader_statistics_enabled = candidate.extensions.shader_statistics,
            .timestamp_valid_bits = candidate.timestamp_valid_bits,
        };
    }

    pub fn deinit(self: *Context) void {
        self.device.destroyDevice(null);
        self.allocator.destroy(self.device.wrapper);
        self.instance.destroySurfaceKHR(self.surface, null);
        if (self.debug_messenger) |messenger|
            self.instance.destroyDebugUtilsMessengerEXT(messenger, null);
        self.instance.destroyInstance(null);
        self.allocator.destroy(self.instance.wrapper);
        self.loader.close();
        self.* = undefined;
    }

    // Waits for every queue to drain.
    //
    // A fence proves a submission finished; it says nothing about the
    // presentation that was queued after it, which still holds the swapchain
    // image and the semaphore it waited on. Destroying either therefore needs
    // this, not a fence. Vulkan specification, vkDestroySwapchainKHR and
    // vkDestroySemaphore both require every use to have completed.
    //
    // It is a cold path by nature: recreation and teardown. Anything per-frame
    // that reaches for it is synchronizing the wrong way.
    pub fn waitIdle(self: *const Context) vk.DeviceWrapper.DeviceWaitIdleError!void {
        try self.device.deviceWaitIdle();
    }

    pub fn deviceName(self: *const Context) []const u8 {
        return std.mem.sliceTo(&self.properties.device_name, 0);
    }

    pub fn findMemoryTypeIndex(
        self: *const Context,
        memory_type_bits: u32,
        required: vk.MemoryPropertyFlags,
    ) MemoryTypeError!u32 {
        for (self.memory_properties.memory_types[0..self.memory_properties.memory_type_count], 0..) |memory_type, index| {
            const supported = memory_type_bits & (@as(u32, 1) << @intCast(index)) != 0;
            if (supported and memory_type.property_flags.contains(required))
                return @intCast(index);
        }
        return error.NoSuitableMemoryType;
    }
};

fn appendSurfaceExtensions(
    names: *std.ArrayList([*:0]const u8),
    allocator: Allocator,
    handles: platform.NativeHandles,
) Allocator.Error!void {
    try names.append(allocator, vk.extensions.khr_surface.name);
    switch (handles) {
        .wayland => try names.append(allocator, vk.extensions.khr_wayland_surface.name),
        // The extra underscore is not a typo. vulkan-zig generates
        // VK_KHR_win32_surface under this spelling; the string it carries is
        // the registry's.
        .win32 => try names.append(allocator, vk.extensions.khr_win_32_surface.name),
    }
}

fn createSurface(instance: Instance, handles: platform.NativeHandles) SurfaceError!vk.SurfaceKHR {
    return switch (handles) {
        .wayland => |wayland| instance.createWaylandSurfaceKHR(&.{
            .display = @ptrCast(wayland.display),
            .surface = @ptrCast(wayland.surface),
        }, null),
        .win32 => |win32| instance.createWin32SurfaceKHR(&.{
            .hinstance = @ptrCast(win32.hinstance),
            .hwnd = @ptrCast(win32.hwnd),
        }, null),
    };
}

fn hasInstanceLayer(
    base_wrapper: *const BaseWrapper,
    allocator: Allocator,
    name: []const u8,
) LayerQueryError!bool {
    const layers = try base_wrapper.enumerateInstanceLayerPropertiesAlloc(allocator);
    defer allocator.free(layers);
    for (layers) |layer|
        if (std.mem.eql(u8, std.mem.sliceTo(&layer.layer_name, 0), name)) return true;
    return false;
}

fn pickIntegratedDevice(
    instance: Instance,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) DevicePickError!DeviceCandidate {
    const physical_devices = try instance.enumeratePhysicalDevicesAlloc(allocator);
    defer allocator.free(physical_devices);

    for (physical_devices) |physical_device| {
        const properties = instance.getPhysicalDeviceProperties(physical_device);
        if (properties.device_type != .integrated_gpu) continue;
        if (try inspectDevice(instance, physical_device, properties, allocator, surface)) |candidate|
            return candidate;
    }
    return error.NoSuitableDevice;
}

fn inspectDevice(
    instance: Instance,
    physical_device: vk.PhysicalDevice,
    properties: vk.PhysicalDeviceProperties,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) DeviceQueryError!?DeviceCandidate {
    const extensions = try inspectDeviceExtensions(
        instance,
        physical_device,
        allocator,
    ) orelse return null;
    if (!try supportsSurface(instance, physical_device, surface)) return null;
    const pipeline_statistics = supportsRequiredFeatures(instance, physical_device) orelse return null;
    const queues = try allocateQueues(instance, physical_device, allocator, surface) orelse return null;
    if (!supportsVertexFormats(instance, physical_device)) return null;

    return .{
        .physical_device = physical_device,
        .properties = properties,
        .queues = queues.allocation,
        .timestamp_valid_bits = queues.timestamp_valid_bits,
        .extensions = extensions,
        .pipeline_statistics = pipeline_statistics,
        .max_buffer_size = queryMaxBufferSize(instance, physical_device),
    };
}

fn allocateQueues(
    instance: Instance,
    physical_device: vk.PhysicalDevice,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) (Allocator.Error || InstanceWrapper.GetPhysicalDeviceSurfaceSupportKHRError)!?ChosenQueues {
    const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, allocator);
    defer allocator.free(families);

    const support = try allocator.alloc(QueueSupport, families.len);
    defer allocator.free(support);

    for (families, support, 0..) |properties, *entry, index| {
        entry.* = .{
            .graphics = properties.queue_flags.graphics_bit,
            .compute = properties.queue_flags.compute_bit,
            .present = try instance.getPhysicalDeviceSurfaceSupportKHR(
                physical_device,
                @intCast(index),
                surface,
            ) == .true,
            .timestamp_valid_bits = properties.timestamp_valid_bits,
        };
    }
    const allocation = chooseQueues(support) orelse return null;
    return .{
        .allocation = allocation,
        // Of the family everything is recorded on. Read here rather than
        // re-enumerated later: the properties are in hand exactly once.
        .timestamp_valid_bits = support[allocation.graphics_family].timestamp_valid_bits,
    };
}

// The families chosen, together with what a timestamp on the graphics one is
// worth. Two answers from one enumeration of the same properties.
const ChosenQueues = struct {
    allocation: QueueAllocation,
    timestamp_valid_bits: u32,
};

fn supportsSurface(
    instance: Instance,
    physical_device: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
) (InstanceWrapper.GetPhysicalDeviceSurfaceFormatsKHRError ||
    InstanceWrapper.GetPhysicalDeviceSurfacePresentModesKHRError)!bool {
    var format_count: u32 = 0;
    _ = try instance.getPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, null);
    var present_mode_count: u32 = 0;
    _ = try instance.getPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, null);
    return format_count > 0 and present_mode_count > 0;
}

fn inspectDeviceExtensions(
    instance: Instance,
    physical_device: vk.PhysicalDevice,
    allocator: Allocator,
) InstanceWrapper.EnumerateDeviceExtensionPropertiesAllocError!?DeviceExtensions {
    const available = try instance.enumerateDeviceExtensionPropertiesAlloc(
        physical_device,
        null,
        allocator,
    );
    defer allocator.free(available);

    for (required_device_extensions) |required|
        if (!hasDeviceExtension(available, required)) return null;

    return .{
        .memory_budget = hasDeviceExtension(
            available,
            vk.extensions.ext_memory_budget.name,
        ),
        .shader_statistics = hasDeviceExtension(
            available,
            vk.extensions.khr_pipeline_executable_properties.name,
        ),
    };
}

fn hasDeviceExtension(
    available: []const vk.ExtensionProperties,
    name: [*:0]const u8,
) bool {
    for (available) |extension|
        if (std.mem.eql(
            u8,
            std.mem.span(name),
            std.mem.sliceTo(&extension.extension_name, 0),
        )) return true;
    return false;
}

fn supportsRequiredFeatures(instance: Instance, physical_device: vk.PhysicalDevice) ?bool {
    const properties = instance.getPhysicalDeviceProperties(physical_device);
    if (properties.api_version < @as(u32, @bitCast(vk.API_VERSION_1_3))) return null;

    var vulkan_13 = vk.PhysicalDeviceVulkan13Features{};
    var vulkan_11 = vk.PhysicalDeviceVulkan11Features{ .p_next = @ptrCast(&vulkan_13) };
    var features = vk.PhysicalDeviceFeatures2{
        .p_next = @ptrCast(&vulkan_11),
        .features = .{},
    };
    instance.getPhysicalDeviceFeatures2(physical_device, &features);

    const core = features.features;
    if (core.sampler_anisotropy != .true or
        core.sample_rate_shading != .true or
        core.texture_compression_bc != .true or
        core.shader_storage_image_extended_formats != .true or
        // Subpixel text has one coverage per colour channel, and a single
        // alpha cannot scale the destination by three different amounts. The
        // second source output carries them, which is what this feature admits.
        core.dual_src_blend != .true or
        vulkan_13.dynamic_rendering != .true or
        vulkan_13.synchronization_2 != .true or
        // The masked alpha path discards, and Slang lowers `discard` to
        // OpDemoteToHelperInvocation rather than OpKill. A demoted invocation
        // stays alive as a helper, which is what keeps the implicit-derivative
        // texture samples around it defined. vk.xml lists this feature as what
        // enables the DemoteToHelperInvocation capability.
        vulkan_13.shader_demote_to_helper_invocation != .true or
        // An instanced draw reads SV_InstanceID, which Slang emits as the
        // InstanceIndex builtin less the BaseInstance one, and BaseInstance is
        // what this feature admits.
        vulkan_11.shader_draw_parameters != .true)
    {
        return null;
    }
    return core.pipeline_statistics_query == .true;
}

fn queryMaxBufferSize(
    instance: Instance,
    physical_device: vk.PhysicalDevice,
) vk.DeviceSize {
    var maintenance_4 = vk.PhysicalDeviceMaintenance4Properties{
        .max_buffer_size = undefined,
    };
    var properties = vk.PhysicalDeviceProperties2{
        .p_next = @ptrCast(&maintenance_4),
        .properties = undefined,
    };
    instance.getPhysicalDeviceProperties2(physical_device, &properties);
    return maintenance_4.max_buffer_size;
}

fn supportsVertexFormats(instance: Instance, physical_device: vk.PhysicalDevice) bool {
    const required = [_]vk.Format{
        .a2b10g10r10_snorm_pack32,
        .r16g16_sfloat,
        .r16g16b16a16_unorm,
        .r16g16b16a16_uint,
    };
    for (required) |format| {
        const properties = instance.getPhysicalDeviceFormatProperties(physical_device, format);
        if (!properties.buffer_features.vertex_buffer_bit) return false;
    }
    return true;
}

fn createDevice(instance: Instance, candidate: DeviceCandidate) InstanceWrapper.CreateDeviceError!vk.Device {
    const priority = [_]f32{1};
    const queue_infos = [_]vk.DeviceQueueCreateInfo{
        .{
            .queue_family_index = candidate.queues.graphics_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
        .{
            .queue_family_index = candidate.queues.present_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
    };
    const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.present_family) 1 else 2;

    var vulkan_13 = vk.PhysicalDeviceVulkan13Features{
        .dynamic_rendering = .true,
        .synchronization_2 = .true,
        .shader_demote_to_helper_invocation = .true,
    };
    var vulkan_11 = vk.PhysicalDeviceVulkan11Features{
        .p_next = @ptrCast(&vulkan_13),
        .shader_draw_parameters = .true,
    };
    const features = vk.PhysicalDeviceFeatures2{
        .p_next = @ptrCast(&vulkan_11),
        .features = .{
            .sampler_anisotropy = .true,
            .sample_rate_shading = .true,
            .texture_compression_bc = .true,
            .shader_storage_image_extended_formats = .true,
            .dual_src_blend = .true,
            .pipeline_statistics_query = if (candidate.pipeline_statistics) .true else .false,
        },
    };
    var extension_names = [_][*:0]const u8{
        vk.extensions.khr_swapchain.name,
        undefined,
        undefined,
    };
    var extension_count: u32 = required_device_extensions.len;
    if (candidate.extensions.memory_budget) {
        extension_names[extension_count] = vk.extensions.ext_memory_budget.name;
        extension_count += 1;
    }
    if (candidate.extensions.shader_statistics) {
        extension_names[extension_count] =
            vk.extensions.khr_pipeline_executable_properties.name;
        extension_count += 1;
    }

    // Appended to the tail of the chain rather than to its head: the head is
    // where the 1.1 and 1.3 feature structures already are, and taking their
    // place would ask for neither.
    //
    // Chained only when the extension is enabled beside it. A feature structure
    // for an extension the device was not given is not a request the driver has
    // to understand.
    var executable_properties = vk.PhysicalDevicePipelineExecutablePropertiesFeaturesKHR{
        .pipeline_executable_info = .true,
    };
    if (candidate.extensions.shader_statistics)
        vulkan_13.p_next = @ptrCast(&executable_properties);

    return instance.createDevice(candidate.physical_device, &.{
        .p_next = @ptrCast(&features),
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &queue_infos,
        .enabled_extension_count = extension_count,
        .pp_enabled_extension_names = &extension_names,
    }, null);
}

fn debugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(.c) vk.Bool32 {
    const kind = if (message_type.validation_bit_ext)
        "validation"
    else if (message_type.performance_bit_ext)
        "performance"
    else
        "general";
    const message = if (callback_data) |data|
        data.p_message orelse "missing callback message"
    else
        "missing callback data";

    if (severity.error_bit_ext) {
        _ = validation_errors.fetchAdd(1, .monotonic);
        log.err("[{s}] {s}", .{ kind, message });
    } else {
        log.warn("[{s}] {s}", .{ kind, message });
    }
    return .false;
}

// Errors the validation layer has reported since the process started.
//
// A count rather than a log line, because a run that means to treat a
// validation error as a failure cannot do that by reading its own output. The
// callback is called from whichever thread made the offending call, so the
// counter is atomic. Zero when validation is not enabled, which is every
// release build.
var validation_errors: std.atomic.Value(u32) = .init(0);

pub fn validationErrorCount() u32 {
    return validation_errors.load(.monotonic);
}
