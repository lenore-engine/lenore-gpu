const std = @import("std");
const vk = @import("vulkan");
const buffer_module = @import("../object/buffer.zig");
const commands = @import("../device/commands.zig");
const Context = @import("../device/context.zig").Context;
const descriptors = @import("../binding/descriptors.zig");
const memory = @import("../memory/allocator.zig");
const pipeline = @import("../binding/pipeline.zig");
const scene = @import("scene.zig");

const Allocator = std.mem.Allocator;
const Buffer = buffer_module.Buffer;

// What the frame carries, measured off the scene target: one mean luminance per
// cell of a fixed grid.
//
// It exists so that an application can set its own exposure from what the frame
// actually holds instead of from a constant chosen once. The engine measures and
// nothing here decides anything about exposure: which part of a frame a camera
// meters, how fast it follows and what it is allowed to reach are the
// application's answers, and a pass that reduced this to one number would have
// taken the first of them.
//
// **It reads the scene target and not the presented image.** That target holds
// radiance before the post operator's exposure, so what a driver adapts from
// does not depend on the exposure it chose last frame. Metering after the
// operator would close the loop through the driver's own output, and an
// adapting exposure would settle on whatever brightness its own lag produced.
//
// The cost does not follow the window. The grid is a fixed count of taps, so a
// frame at 4.1 megapixels is sampled exactly as often as one at 1.9, and what
// grows with a larger window is only how far apart the taps land. On a device
// that shares its memory bandwidth with the processor that is the property worth
// having: an average over every pixel would read the whole target once a frame
// to answer a question a few thousand samples already answer.

// Cells across the frame, and taps across a cell. Both are the supplied
// shader's as much as this file's: `tap_side` is its local size, which Vulkan
// takes from the shader's own execution mode and which is therefore never
// communicated to it, and `cell_side` is the dispatch it is given. A shader
// declaring a different local size reduces a fraction of its cell, and the
// device reports nothing. Whatever supplies the words holds the two together.
pub const cell_side: u32 = 8;
pub const tap_side: u32 = 16;
pub const cell_count: u32 = cell_side * cell_side;

// The grid as a driver reads it, row-major from the top left of the frame.
pub const Cells = [cell_count]f32;

pub const bindings = [_]descriptors.Binding{
    // The frame, as the main pass left it.
    .{ .slot = 0, .name = "scene", .kind = .combined_image_sampler, .stages = .{ .compute_bit = true } },
    // Where the cells are written: this frame slot's own buffer, host visible so
    // that the reduction lands where the host reads it and no copy stands
    // between the two.
    .{ .slot = 1, .name = "cells", .kind = .storage_buffer, .stages = .{ .compute_bit = true } },
};

const Sets = descriptors.Sets(&bindings);

// What a metering shader has to supply for this pass to be built from it. One
// compute entry point, and the whole interface it may read is `bindings`; there
// is no push block, because everything the dispatch needs is either its own
// identity or fixed above.
pub const Shader = struct {
    spirv: []const u32,
    compute_entry: [*:0]const u8,
};

pub const InitError = Allocator.Error ||
    descriptors.InitError ||
    buffer_module.InitError ||
    pipeline.CreateError ||
    vk.DeviceWrapper.CreateSamplerError;

// Makes the main pass's colour writes readable by this dispatch.
//
// The scene pass ends with a barrier of its own, but its destination is the
// fragment stage: the chain and the post pass are what it was written for. A
// compute read is outside that scope and needs its own dependency on the same
// writes. No layout transition comes with it, since the image is already in the
// layout both readers sample it in.
fn sourceDependency() commands.Dependency {
    return .{
        .src_stage = .{ .color_attachment_output_bit = true },
        .src_access = .{ .color_attachment_write_bit = true },
        .dst_stage = .{ .compute_shader_bit = true },
        .dst_access = .{ .shader_read_bit = true },
    };
}

// Makes the cells the dispatch wrote readable by the host.
//
// The memory is coherent, so nothing has to be invalidated, but the host is
// still a separate access scope and the write has to be made available to it.
// Stated as a barrier rather than left to the submission, because a dependency
// that is written down is one a reader can check.
fn hostDependency() commands.Dependency {
    return .{
        .src_stage = .{ .compute_shader_bit = true },
        .src_access = .{ .shader_storage_write_bit = true },
        .dst_stage = .{ .host_bit = true },
        .dst_access = .{ .host_read_bit = true },
    };
}

pub const MeterPass = struct {
    context: *const Context,
    allocator: Allocator,
    frames: usize,

    sampler: vk.Sampler,
    module: vk.ShaderModule,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    sets: Sets,

    // One per frame in flight, each written by that frame's dispatch and read by
    // the host once that frame's fence has signalled. One buffer for all of them
    // would be a frame's cells overwritten while the host was reading them.
    cells: []Buffer,
    // Whether a dispatch has ever written that slot. A buffer's contents before
    // its first one are undefined, and zero would be a reading rather than the
    // absence of one: a driver cannot tell a black frame from a slot nothing has
    // measured unless this says so.
    written: []bool,

    pub fn init(
        context: *const Context,
        memory_allocator: *memory.MemoryAllocator,
        allocator: Allocator,
        frames: usize,
        source_view: vk.ImageView,
        shader: Shader,
    ) InitError!MeterPass {
        const sampler = try createSampler(context);
        errdefer context.device.destroySampler(sampler, null);

        const module = try pipeline.createModule(context, shader.spirv);
        errdefer context.device.destroyShaderModule(module, null);

        var sets = try Sets.init(context, allocator, @intCast(frames));
        errdefer sets.deinit(context, allocator);

        const layout = try pipeline.createLayout(context, .{
            .descriptor_sets = &.{sets.layout},
            .push_constants = &.{},
        });
        errdefer context.device.destroyPipelineLayout(layout, null);

        const built = try pipeline.createCompute(context, .{
            .layout = layout,
            .stage = .{ .module = module, .entry_point = shader.compute_entry },
        });
        errdefer context.device.destroyPipeline(built, null);

        const written = try allocator.alloc(bool, frames);
        errdefer allocator.free(written);
        @memset(written, false);

        const cells = try allocator.alloc(Buffer, frames);
        errdefer allocator.free(cells);
        // Every buffer made before the one that fails is destroyed, so a device
        // out of readback memory leaves nothing allocated rather than a
        // half-built pass.
        var made: usize = 0;
        errdefer for (cells[0..made]) |*owned| owned.deinit();
        while (made < frames) : (made += 1) {
            cells[made] = try Buffer.init(
                context,
                memory_allocator,
                @sizeOf(Cells),
                .{ .storage_buffer_bit = true },
                .readback,
            );
        }

        const self: MeterPass = .{
            .context = context,
            .allocator = allocator,
            .frames = frames,
            .sampler = sampler,
            .module = module,
            .layout = layout,
            .pipeline = built,
            .sets = sets,
            .cells = cells,
            .written = written,
        };
        self.writeSets(source_view);
        return self;
    }

    // Vulkan specification, vkDestroyPipeline and the rest: every submission
    // naming any of these must have completed. The caller drains the device.
    pub fn deinit(self: *MeterPass) void {
        const device = self.context.device;
        for (self.cells) |*owned| owned.deinit();
        self.allocator.free(self.cells);
        self.allocator.free(self.written);
        self.sets.deinit(self.context, self.allocator);
        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.layout, null);
        device.destroyShaderModule(self.module, null);
        device.destroySampler(self.sampler, null);
        self.* = undefined;
    }

    // Point the sets at a new target. A resize replaces the image the frame is
    // drawn into, and the sets hold its view; everything else survives one, so a
    // resize costs no allocation here at all.
    //
    // Vulkan specification, vkUpdateDescriptorSets: a set may not be updated
    // while it is in use by a submission that has not completed. The caller
    // drains the device before resizing, as it does for the chain.
    pub fn recreate(self: *MeterPass, source_view: vk.ImageView) void {
        self.writeSets(source_view);
    }

    fn writeSets(self: *const MeterPass, source_view: vk.ImageView) void {
        for (self.cells, 0..) |*owned, frame| {
            self.sets.writeImages(self.context, frame, .{
                .scene = descriptors.ImageSource{
                    .view = source_view,
                    .sampler = self.sampler,
                    .layout = scene.sampled_layout,
                },
            });
            self.sets.writeBuffers(self.context, frame, .{ .cells = owned });
        }
    }

    // One dispatch, recorded after the main pass has ended and before anything
    // else reads the target.
    //
    // The frame index is the slot whose buffer this writes and whose fence the
    // host waits on before reading it. It is the caller's own ring index: this
    // pass has no clock and does not count frames.
    pub fn record(self: *MeterPass, command_buffer: vk.CommandBuffer, frame: usize) void {
        std.debug.assert(frame < self.frames);

        const device = self.context.device;
        commands.recordMemoryBarrier(self.context, command_buffer, sourceDependency());

        device.cmdBindPipeline(command_buffer, .compute, self.pipeline);
        device.cmdBindDescriptorSets(
            command_buffer,
            .compute,
            self.layout,
            0,
            &.{self.sets.set(frame)},
            &.{},
        );
        // One group a cell. The taps inside it are the shader's local size and
        // are not named here.
        device.cmdDispatch(command_buffer, cell_side, cell_side, 1);

        commands.recordMemoryBarrier(self.context, command_buffer, hostDependency());
        // Recorded rather than submitted, which is the earlier of the two and so
        // the safe side to be wrong on: a slot marked here and never submitted
        // is read as a stale measurement, where the other way round a slot that
        // holds a real one would read as nothing.
        self.written[frame] = true;
    }

    // What that frame's dispatch measured.
    //
    // Vulkan specification, Memory Mapping: the caller synchronizes this read
    // against the submission that wrote it.
    //
    // Null is a slot no dispatch has written yet, or a buffer whose memory is
    // not mapped. A readback allocation always is mapped, so the second stands
    // for a device that reported no host visible memory type.
    //
    // A copy and not a view of the mapping: the next frame using this slot
    // overwrites those bytes, and a driver holding a slice would not be able to
    // tell when.
    pub fn read(self: *const MeterPass, frame: usize) ?Cells {
        std.debug.assert(frame < self.frames);

        if (!self.written[frame]) return null;
        const bytes = self.cells[frame].mapped() orelse return null;
        var cells: Cells = undefined;
        @memcpy(std.mem.asBytes(&cells), bytes[0..@sizeOf(Cells)]);
        return cells;
    }
};

// Linear and clamped, which is what a tap wants: the filter averages the four
// texels a tap falls between, so the grid samples a frame rather than a lattice
// of single pixels, and the clamp keeps a tap on the edge of the frame from
// reading a border colour that is in no picture.
fn createSampler(context: *const Context) vk.DeviceWrapper.CreateSamplerError!vk.Sampler {
    return context.device.createSampler(&.{
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .clamp_to_edge,
        .address_mode_w = .clamp_to_edge,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 1,
        .compare_enable = .false,
        .compare_op = .always,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
    }, null);
}
