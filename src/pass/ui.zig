const std = @import("std");
const vk = @import("vulkan");
const res = @import("lenore-resources");

const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const memory = @import("../memory/allocator.zig");
const pass = @import("scene.zig");
const per_frame = @import("../binding/per_frame.zig");
const pipeline = @import("../binding/pipeline.zig");
const pool = @import("../store/pool.zig");

const Allocator = std.mem.Allocator;

// The overlay: a two-dimensional draw list composited onto a finished picture.
//
// It is recorded inside the rendering the post pass opens, after the tone
// operator has written the presentable image and before the barrier that hands
// it to presentation. That placement is the whole reason this pass exists
// separately from anything in the main pass: what it draws is in display
// colour, not in scene radiance, so it must not be tone mapped or bloomed.
//
// The list it draws is `lenore-resources` data. Whoever produced it is not
// named here and cannot be: a draw list is neutral, and the module that writes
// one names the same declarations this does.

// What a UI shader has to supply. One vertex stage and one fragment stage, and
// no variants: the fragment stage multiplies the vertex colour by the sampled
// texel unconditionally, so a solid fill is the white image and a tinted glyph
// is a coverage image, with no branch and no second pipeline between them.
pub const Shader = struct {
    spirv: []const u32,
    vertex_entry: [*:0]const u8,
    fragment_entry: [*:0]const u8,
};

// The block the vertex stage reads.
//
// The reciprocal rather than the extent, so the shader turns a pixel position
// into clip space with a multiply and no divide per vertex. Recomputing it per
// frame costs two divides on the host and removes one from every vertex.
pub const PushConstants = extern struct {
    inverse_extent: [2]f32,

    comptime {
        std.debug.assert(@sizeOf(PushConstants) == 8);
    }
};

// The vertex stage alone. The fragment stage reads the descriptor and the
// interpolated colour and needs nothing from here, so a range naming it would
// declare a block it never reads.
pub const push_constant_range: vk.PushConstantRange = .{
    .stage_flags = .{ .vertex_bit = true },
    .offset = 0,
    .size = @sizeOf(PushConstants),
};

// One image, sampled by the fragment stage.
//
// A set per registered image rather than one array binding over all of them.
// The count a bindless set declares is fixed at layout creation, so an array
// would put the registry's capacity into the pipeline layout, and the set would
// have to be rewritten whenever any slot changed. A set per image is written
// once when the image is registered and bound per draw, which is the frequency
// the draws actually change it at.
pub const bindings = [_]descriptors.Binding{
    .{ .slot = 0, .name = "image", .kind = .combined_image_sampler, .stages = .{ .fragment_bit = true } },
};

pub const Sets = descriptors.Sets(&bindings);

// The vertex `lenore-resources` declares, described for the driver.
//
// The offsets are read from the declaration rather than written again, so the
// two cannot disagree. The formats are the one thing that must be stated: the
// declaration says `[2]f32` and `[4]f16`, and which Vulkan format corresponds
// to each is a choice this module makes.
pub const vertex_input: pipeline.VertexInput = blk: {
    var input: pipeline.VertexInput = .none;
    input.bindings[0] = .{
        .binding = 0,
        .stride = @sizeOf(res.Vertex2D),
        .input_rate = .vertex,
    };
    input.binding_count = 1;
    input.attributes[0] = .{
        .location = 0,
        .binding = 0,
        .format = .r32g32_sfloat,
        .offset = @offsetOf(res.Vertex2D, "position"),
    };
    input.attributes[1] = .{
        .location = 1,
        .binding = 0,
        .format = .r32g32_sfloat,
        .offset = @offsetOf(res.Vertex2D, "uv"),
    };
    input.attributes[2] = .{
        .location = 2,
        .binding = 0,
        .format = .r16g16b16a16_sfloat,
        .offset = @offsetOf(res.Vertex2D, "colour"),
    };
    input.attribute_count = 3;
    break :blk input;
};

// The index width the draw list is written in, as the bind command names it.
// Derived from the declaration so that widening `res.DrawIndex` reaches the
// bind rather than leaving it describing the old width.
pub const index_type: vk.IndexType = switch (res.DrawIndex) {
    u16 => .uint16,
    u32 => .uint32,
    else => @compileError("no index type for " ++ @typeName(res.DrawIndex)),
};

// The layout a registered image is expected to be in. Stated once here because
// every descriptor this pass writes says it, and a descriptor whose layout is
// not the one the image is actually in is a validation error at draw time and
// nothing sooner.
pub const sampled_layout: vk.ImageLayout = .shader_read_only_optimal;

// What the pass holds per registered image.
//
// Not the image. The registry never creates or destroys one: it stores the two
// handles a descriptor is written from and the owner of the image keeps it
// alive for as long as it is registered. That keeps "who destroys this" a
// property of whoever made it, which for the white texel is the texture cache
// and for a glyph atlas will be whoever builds one.
const Registered = struct {
    view: vk.ImageView,
    sampler: vk.Sampler,
};

const Pool = pool.ResourcePool(Registered);

pub const RegistryInitError = descriptors.InitError;

pub const AddError = error{
    // One descriptor set exists per slot and they are allocated once, so the
    // capacity is a hard bound rather than something to grow into.
    //
    // The pool's own exhaustion arrives here under this name as well. Both mean
    // there is no slot to be had, and a caller can act on neither differently,
    // so a second name would only be a second branch nothing takes.
    RegistryFull,
} || Allocator.Error;

// Handles and descriptor sets for the images a draw list may name.
//
// A set is allocated for every slot at construction and stays attached to that
// slot for the registry's life. That is what `ResourcePool.slotIndex` is for:
// the pool decides which slot an image takes and reuses freed ones, and the set
// array is indexed by the same number rather than by a second free list
// tracking the same reuse.
pub const Registry = struct {
    slots: Pool,
    sets: Sets,
    capacity: u32,

    pub fn init(
        context: *const Context,
        allocator: Allocator,
        capacity: u32,
    ) RegistryInitError!Registry {
        var sets = try Sets.init(context, allocator, capacity);
        errdefer sets.deinit(context, allocator);

        return .{ .slots = .empty, .sets = sets, .capacity = capacity };
    }

    // Vulkan specification, vkDestroyDescriptorPool: the sets it allocated must
    // not be in use by a pending submission, which the caller establishes. No
    // image is touched, because none is owned.
    pub fn deinit(self: *Registry, context: *const Context, allocator: Allocator) void {
        self.slots.deinit(allocator);
        self.sets.deinit(context, allocator);
        self.* = undefined;
    }

    // Registers a view and a sampler and writes the slot's descriptor.
    //
    // Vulkan specification, vkUpdateDescriptorSets: the set must not be in use
    // by submitted work that has not completed. This is therefore a cold path,
    // and registering an image while a frame that could name it is in flight is
    // the caller's problem rather than this one's.
    pub fn add(
        self: *Registry,
        context: *const Context,
        allocator: Allocator,
        view: vk.ImageView,
        sampler: vk.Sampler,
    ) AddError!res.ImageHandle {
        if (self.slots.count() >= self.capacity) return error.RegistryFull;

        const handle = self.slots.add(allocator, .{ .view = view, .sampler = sampler }) catch |err| switch (err) {
            error.PoolExhausted => return error.RegistryFull,
            else => |remaining| return remaining,
        };
        // Live by construction, and below the capacity because the count was:
        // the pool only creates a slot when its free list is empty, so the
        // largest index it has ever handed out is one less than the largest
        // number of entries alive at once.
        const slot = self.slots.slotIndex(handle).?;
        self.sets.writeImages(context, slot, .{
            .image = descriptors.ImageSource{
                .view = view,
                .sampler = sampler,
                .layout = sampled_layout,
            },
        });
        return handleOf(handle);
    }

    // Releases a slot. The image is the caller's to destroy, and only after
    // every frame that could still be sampling it has completed.
    pub fn remove(self: *Registry, handle: res.ImageHandle) void {
        _ = self.slots.remove(poolHandleOf(handle));
    }

    // The set a draw naming this image binds, or null if the handle names
    // nothing live. A stale handle is a draw that is skipped rather than one
    // that samples whatever took the slot.
    pub fn descriptorSet(self: *const Registry, handle: res.ImageHandle) ?vk.DescriptorSet {
        const slot = self.slots.slotIndex(poolHandleOf(handle)) orelse return null;
        return self.sets.set(slot);
    }

    pub fn count(self: *const Registry) u32 {
        return self.slots.count();
    }

    // The two handle types are both a u64 with an index in the low half, a
    // generation in the high half and zero reserved, so the crossing is the
    // representation and not a lookup. They are distinct types so that a draw
    // list, which resolves nothing, cannot be handed a pool handle by accident.
    fn handleOf(handle: Pool.Handle) res.ImageHandle {
        return @enumFromInt(@intFromEnum(handle));
    }

    fn poolHandleOf(handle: res.ImageHandle) Pool.Handle {
        return @enumFromInt(@intFromEnum(handle));
    }
};

// How much of a frame the pass can hold.
//
// Provisional, and derived rather than guessed: a quad is four vertices and six
// indices, so the vertex and index counts are one budget of two thousand quads
// expressed twice. Two thousand is above any panel and below a dense editor,
// and the number that decides it is how many glyphs a frame shows. Nothing has
// measured that yet, so these are what an application overrides rather than
// what it is expected to keep.
pub const Capacity = struct {
    vertices: usize = 8192,
    indices: usize = 12288,
    // A draw per image and clip change. A panelled frame produces a handful;
    // this is room for a UI that changes clip far more often than that.
    commands: usize = 256,
    // Nesting, not rectangles. Sixteen is deeper than a UI that a person can
    // follow.
    clip_depth: usize = 16,
    images: u32 = 64,
};

pub const InitError = error{
    ZeroFrames,
} || RegistryInitError ||
    per_frame.InitError ||
    per_frame.LayoutError ||
    pipeline.CreateError ||
    Allocator.Error;

pub const UiPass = struct {
    context: *const Context,
    allocator: Allocator,

    registry: Registry,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    handle: vk.Pipeline,

    // One ring each, written by the producer and read by the device. The
    // invariant is `PerFrame`'s: a slot is written after that slot's submission
    // fence has signalled and before the same slot is submitted again, which is
    // where in the frame the list has to be built.
    vertices: per_frame.PerFrame(res.Vertex2D),
    indices: per_frame.PerFrame(res.DrawIndex),

    // Ordinary host memory, and one copy rather than one per frame. Neither is
    // read by the device: the commands drive this pass's own loop and the clips
    // become scissor rectangles, both within the frame that wrote them.
    commands: []res.DrawCommand,
    clips: []res.Rect,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        frames: usize,
        capacity: Capacity,
        shader: Shader,
        colour_format: vk.Format,
    ) InitError!UiPass {
        if (frames == 0) return error.ZeroFrames;

        var registry = try Registry.init(context, allocator, capacity.images);
        errdefer registry.deinit(context, allocator);

        var vertices = try per_frame.PerFrame(res.Vertex2D).init(
            context,
            memory_allocator,
            frames,
            capacity.vertices,
            .{ .vertex_buffer_bit = true },
        );
        errdefer vertices.deinit();

        var indices = try per_frame.PerFrame(res.DrawIndex).init(
            context,
            memory_allocator,
            frames,
            capacity.indices,
            .{ .index_buffer_bit = true },
        );
        errdefer indices.deinit();

        const commands = try allocator.alloc(res.DrawCommand, capacity.commands);
        errdefer allocator.free(commands);
        const clips = try allocator.alloc(res.Rect, capacity.clip_depth);
        errdefer allocator.free(clips);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{registry.sets.layout},
            .push_constants = &.{push_constant_range},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const handle = try pipeline.create(context, .{
            .mode = .overlay,
            .vertex_input = vertex_input,
            // Nothing here has a back to face away: the list is built in one
            // winding and a culled overlay would drop half of it for a sign.
            .culling = .{ .fixed = .{} },
            // No depth attachment. The rendering this is recorded into declares
            // none, and the overlay mode carries no depth behaviour to declare.
            .formats = .{ .colour = colour_format },
            .layout = layout,
            .stages = .{
                .vertex = .{ .module = module, .entry_point = shader.vertex_entry },
                .fragment = .{ .module = module, .entry_point = shader.fragment_entry },
            },
        });

        return .{
            .context = context,
            .allocator = allocator,
            .registry = registry,
            .module = module,
            .layout = layout,
            .handle = handle,
            .vertices = vertices,
            .indices = indices,
            .commands = commands,
            .clips = clips,
        };
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed, which the caller establishes.
    pub fn deinit(self: *UiPass) void {
        self.context.device.destroyPipeline(self.handle, null);
        self.context.device.destroyPipelineLayout(self.layout, null);
        self.context.device.destroyShaderModule(self.module, null);
        self.indices.deinit();
        self.vertices.deinit();
        self.allocator.free(self.clips);
        self.allocator.free(self.commands);
        self.registry.deinit(self.context, self.allocator);
        self.* = undefined;
    }

    // Where this frame's list is written.
    //
    // The vertex and index slices are a mapping of memory the device reads.
    // They are written and never read back; the command and clip arrays are
    // ordinary memory and the producer does re-read those.
    pub fn storage(self: *const UiPass, frame_index: usize) res.DrawListStorage {
        return .{
            .vertices = self.vertices.slice(frame_index),
            .indices = self.indices.slice(frame_index),
            .commands = self.commands,
            .clips = self.clips,
        };
    }

    // Records the list into a rendering the caller has already opened.
    //
    // An empty list records nothing at all, not even a pipeline bind: a frame
    // with no UI must leave the state the pass before it established exactly as
    // it was.
    pub fn record(
        self: *const UiPass,
        command_buffer: vk.CommandBuffer,
        frame_index: usize,
        draws: []const res.DrawCommand,
        extent: vk.Extent2D,
    ) void {
        if (draws.len == 0) return;

        const device = self.context.device;
        device.cmdBindPipeline(command_buffer, .graphics, self.handle);
        device.cmdSetViewport(command_buffer, 0, &.{pass.viewport(extent)});

        const vertex_offset: vk.DeviceSize = self.vertices.dynamicOffset(frame_index);
        device.cmdBindVertexBuffers(command_buffer, 0, &.{self.vertices.handle()}, &.{vertex_offset});
        device.cmdBindIndexBuffer(
            command_buffer,
            self.indices.handle(),
            self.indices.dynamicOffset(frame_index),
            index_type,
        );

        const constants: PushConstants = .{ .inverse_extent = inverseExtent(extent) };
        device.cmdPushConstants(
            command_buffer,
            self.layout,
            push_constant_range.stage_flags,
            push_constant_range.offset,
            push_constant_range.size,
            @ptrCast(&constants),
        );

        // Rebinding a set that is already bound is a command the driver still
        // has to process, and a UI's draws mostly share one image.
        var bound: vk.DescriptorSet = .null_handle;
        for (draws) |draw| {
            const scissor = scissorFor(draw.clip, extent) orelse continue;
            // A handle the registry no longer knows is a draw that is skipped.
            // The alternative is sampling whatever took the slot, which is a
            // picture rather than a diagnostic.
            const set = self.registry.descriptorSet(draw.image) orelse continue;

            if (set != bound) {
                device.cmdBindDescriptorSets(command_buffer, .graphics, self.layout, 0, &.{set}, &.{});
                bound = set;
            }
            device.cmdSetScissor(command_buffer, 0, &.{scissor});
            // No vertex base and no instance base. An index addresses the
            // whole of the frame's vertex array, so a draw's indices are
            // absolute and there is nothing for the fetch to add to them.
            device.cmdDrawIndexed(
                command_buffer,
                draw.index_count,
                1,
                draw.first_index,
                0,
                0,
            );
        }
    }
};

// The reciprocal of the target, which is what the vertex stage multiplies by.
//
// A zero extent cannot reach here: a swapchain is not created at one and the
// frame loop skips a minimised window, so the divide has a non-zero divisor by
// construction rather than by a check.
pub fn inverseExtent(extent: vk.Extent2D) [2]f32 {
    return .{
        1.0 / @as(f32, @floatFromInt(extent.width)),
        1.0 / @as(f32, @floatFromInt(extent.height)),
    };
}

// A clip rectangle as the scissor that enforces it, or null if it covers no
// pixel.
//
// Split from the recording because everything that can be wrong here is
// arithmetic, and a device is what would otherwise stand between that and a
// test.
//
// Minima round down and maxima round up, the same conservative rule the
// producer converts a filled rectangle with: a scissor that rounded inward
// would clip the edge pixel of the geometry it is meant to admit.
//
// The clamp is what makes the conversion total. A clip is only ever an
// intersection of rectangles the producer validated, but the target it is
// clamped against can have shrunk since, and `VkRect2D` cannot express a
// negative origin. A non-finite value cannot survive either: `@max` and `@min`
// return the operand that is not NaN, so a NaN edge takes the bound it was
// compared against.
pub fn scissorFor(clip: res.Rect, extent: vk.Extent2D) ?vk.Rect2D {
    const width: f32 = @floatFromInt(extent.width);
    const height: f32 = @floatFromInt(extent.height);

    const left = @max(@floor(clip.x), 0);
    const top = @max(@floor(clip.y), 0);
    const right = @min(@ceil(clip.x + clip.width), width);
    const bottom = @min(@ceil(clip.y + clip.height), height);
    if (right <= left or bottom <= top) return null;

    // Every value is inside [0, extent] by the clamps above, so the conversions
    // are in range by construction rather than by the cast's own checking.
    return .{
        .offset = .{ .x = @intFromFloat(left), .y = @intFromFloat(top) },
        .extent = .{
            .width = @intFromFloat(right - left),
            .height = @intFromFloat(bottom - top),
        },
    };
}
