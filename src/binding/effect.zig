const std = @import("std");
const vk = @import("vulkan");

const Context = @import("../device/context.zig").Context;
const pipeline = @import("pipeline.zig");
const res = @import("lenore-resources");

// One consumer's shader modules, pipeline layouts and pipelines, declared as a
// table and torn down as a unit.
//
// What it removes is a ladder every consumer of this module writes by hand:
// create a module and `errdefer` its destroy, create a layout and `errdefer`
// that, create each pipeline with an `errdefer` of its own, and then a `deinit`
// destroying all of it in the reverse order. Seven consumers had a copy, from
// three lines in the smallest to thirty in `atoms`, and the reverse order is
// stated once here rather than seven times.
//
// The shape is read off `atoms`, which is what rules out anything fixed: two
// modules, two layouts, seven compute pipelines and three graphics passes, each
// pass with its own blend mode. A type with one module and one layout would fit
// four consumers and force the fifth to keep its ladder.
//
// Comptime carries what is comptime — which modules, layouts and pipelines
// exist, what each is called, which module and layout each pipeline is built
// from, and every piece of graphics state that is a property of the shader
// rather than of the window. Runtime carries the two things that cannot be
// known earlier: the words a module is created from, and the attachment formats,
// which follow the swapchain.

// A pipeline's stage, and the state that belongs to it rather than to the frame.
pub const Stage = union(enum) {
    compute: [*:0]const u8,
    graphics: Graphics,
};

pub const Graphics = struct {
    vertex: [*:0]const u8,
    fragment: [*:0]const u8,
    mode: pipeline.Mode,
    culling: pipeline.Culling,
    // Null for a vertex stage that reads no buffer, which every fullscreen
    // triangle is. See `pipeline.Config`.
    streams: ?res.VertexStreams = null,
    depth_bias: ?pipeline.DepthBias = null,
};

// One entry of the table. The module and the layout are named rather than
// indexed, so that a table read top to bottom says which shader each pipeline
// comes from.
//
// A table entry spells this type: `.disk = Spec{ ... }`. It has to, and the
// reason is worth knowing before trying to remove it. A field of a comptime
// struct has its own anonymous type by the time anything reads it and is no
// longer a literal, so `.disk = .{ ... }` would not coerce here and neither
// would the `Stage` inside it. Naming the type once at the top of an entry
// gives every field below a known destination, and the nested `.stage` and
// `.graphics` literals then coerce on their own.
pub const Spec = struct {
    module: []const u8,
    layout: []const u8,
    stage: Stage,
};

fn indexOfName(comptime names: anytype, comptime sought: []const u8) ?usize {
    inline for (names, 0..) |name, index| {
        if (std.mem.eql(u8, name, sought)) return index;
    }
    return null;
}

// `declaration` is a struct of three fields:
//
//   .modules   a tuple of names, one per shader module
//   .layouts   a tuple of names, one per pipeline layout
//   .pipelines a struct of `Spec`, keyed by what each pipeline is called
//
// Every name a pipeline gives is resolved here, so a pipeline built from a
// module the table does not declare is a compile error rather than a handle
// that was never created. That check is what pays for the metaprogramming.
pub fn ShaderEffect(comptime declaration: anytype) type {
    const module_names = declaration.modules;
    const layout_names = declaration.layouts;
    const Pipelines = @TypeOf(declaration.pipelines);
    const pipeline_fields = @typeInfo(Pipelines).@"struct".fields;

    comptime {
        if (pipeline_fields.len == 0) @compileError("an effect with no pipelines builds nothing");
        // A tuple is not an array, so the pairs are walked by index rather than
        // by slicing the tail.
        for (module_names, 0..) |name, index| {
            for (module_names, 0..) |other, other_index| {
                if (other_index > index and std.mem.eql(u8, name, other))
                    @compileError("two modules share the name " ++ name);
            }
        }
        for (layout_names, 0..) |name, index| {
            for (layout_names, 0..) |other, other_index| {
                if (other_index > index and std.mem.eql(u8, name, other))
                    @compileError("two layouts share the name " ++ name);
            }
        }
        for (pipeline_fields) |field| {
            const spec: Spec = @field(declaration.pipelines, field.name);
            if (indexOfName(module_names, spec.module) == null)
                @compileError(field.name ++ " names no declared module: " ++ spec.module);
            if (indexOfName(layout_names, spec.layout) == null)
                @compileError(field.name ++ " names no declared layout: " ++ spec.layout);
        }
    }

    return struct {
        const Self = @This();

        // What a caller says instead of an index. `std.meta.FieldEnum` rather
        // than an enum built here: the same call exists in 0.16 and in
        // 0.17-dev, while the type-info shape a hand-rolled `@Type` would
        // depend on does not survive between them.
        pub const Name = std.meta.FieldEnum(Pipelines);

        modules: [module_names.len]vk.ShaderModule,
        layouts: [layout_names.len]vk.PipelineLayout,
        pipelines: [pipeline_fields.len]vk.Pipeline,

        pub fn get(self: *const Self, comptime name: Name) vk.Pipeline {
            return self.pipelines[@intFromEnum(name)];
        }

        // The layout a pipeline was built with, which is what a push constant
        // or a descriptor bind needs. Taken by pipeline name rather than by
        // layout name, because the caller has the first in hand and the two
        // agreeing is the table's business rather than the call site's.
        pub fn layoutFor(self: *const Self, comptime name: Name) vk.PipelineLayout {
            const spec: Spec = @field(declaration.pipelines, @tagName(name));
            return self.layouts[comptime indexOfName(layout_names, spec.layout).?];
        }

        // `config` carries `.modules` and `.layouts` keyed by the names above,
        // and one `.formats` for every graphics pipeline in the table. Formats
        // are shared because they are the attachments the frame is drawn into,
        // which is a property of the target and not of any one pipeline.
        //
        // A layout entry names `pipeline.LayoutConfig`, for the same reason a
        // table entry names `Spec`: a field is no longer a literal by the time
        // this reads it. One rule for both halves rather than a type name in one
        // place and an unwrapping in the other.
        pub fn init(context: *const Context, config: anytype) pipeline.CreateError!Self {
            var self: Self = .{
                .modules = undefined,
                .layouts = undefined,
                .pipelines = undefined,
            };

            var modules_built: usize = 0;
            errdefer for (self.modules[0..modules_built]) |handle|
                context.device.destroyShaderModule(handle, null);
            inline for (module_names, 0..) |name, index| {
                self.modules[index] = try pipeline.createModule(context, @field(config.modules, name));
                modules_built += 1;
            }

            var layouts_built: usize = 0;
            errdefer for (self.layouts[0..layouts_built]) |handle|
                context.device.destroyPipelineLayout(handle, null);
            inline for (layout_names, 0..) |name, index| {
                self.layouts[index] = try pipeline.createLayout(context, @field(config.layouts, name));
                layouts_built += 1;
            }

            var pipelines_built: usize = 0;
            errdefer for (self.pipelines[0..pipelines_built]) |handle|
                context.device.destroyPipeline(handle, null);
            inline for (pipeline_fields, 0..) |field, index| {
                const spec: Spec = @field(declaration.pipelines, field.name);
                const module = self.modules[comptime indexOfName(module_names, spec.module).?];
                const layout = self.layouts[comptime indexOfName(layout_names, spec.layout).?];

                self.pipelines[index] = switch (spec.stage) {
                    .compute => |entry| try pipeline.createCompute(context, .{
                        .layout = layout,
                        .stage = .{ .module = module, .entry_point = entry },
                    }),
                    .graphics => |graphics| try pipeline.create(context, .{
                        .mode = graphics.mode,
                        .streams = graphics.streams,
                        .culling = graphics.culling,
                        .formats = config.formats,
                        .layout = layout,
                        .depth_bias = graphics.depth_bias,
                        .stages = .{
                            .vertex = .{ .module = module, .entry_point = graphics.vertex },
                            .fragment = .{ .module = module, .entry_point = graphics.fragment },
                        },
                    }),
                };
                pipelines_built += 1;
            }

            return self;
        }

        // Vulkan specification, vkDestroyPipeline, vkDestroyPipelineLayout and
        // vkDestroyShaderModule: every submission naming any of these must have
        // completed, which the caller establishes.
        //
        // The order is stated here and nowhere else, which is the point of
        // owning the three together: pipelines, then the layouts they were
        // built against, then the modules those stages came from. A generated
        // teardown that inherited its order from field order would be a
        // dependency held by an accident of declaration.
        pub fn deinit(self: *Self, context: *const Context) void {
            for (self.pipelines) |handle| context.device.destroyPipeline(handle, null);
            for (self.layouts) |handle| context.device.destroyPipelineLayout(handle, null);
            for (self.modules) |handle| context.device.destroyShaderModule(handle, null);
            self.* = undefined;
        }
    };
}
