const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_validation", optimize == .Debug);

    const platform = b.dependency("lenore_platform", .{ .target = target, .optimize = optimize });
    const resource = b.dependency("lenore_resources", .{ .target = target, .optimize = optimize });
    const vulkan = b.dependency("vulkan_zig", .{
        .registry = b.path("vk/vk.xml"),
    }).module("vulkan-zig");
    const zmath = b.dependency("zmath", .{}).module("root");

    const suballocator = b.createModule(.{
        .root_source_file = b.path("src/memory/suballocator.zig"),
        .target = target,
        .optimize = optimize,
    });
    const placement = b.createModule(.{
        .root_source_file = b.path("src/staging/placement.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mod = b.addModule("lenore-gpu", .{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "lenore-platform", .module = platform.module("lenore-platform") },
            .{ .name = "lenore-resources", .module = resource.module("lenore-resources") },
            .{ .name = "memory-suballocator", .module = suballocator },
            .{ .name = "staging-placement", .module = placement },
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "zmath", .module = zmath },
        },
        .target = target,
        .optimize = optimize,
    });

    addShaders(b, mod);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = testRoot(b, "tests"),
            .imports = &.{
                .{ .name = "lenore-gpu", .module = mod },
                .{ .name = "lenore-resources", .module = resource.module("lenore-resources") },
                .{ .name = "memory-suballocator", .module = suballocator },
                .{ .name = "staging-placement", .module = placement },
                .{ .name = "vulkan", .module = vulkan },
                .{ .name = "zmath", .module = zmath },
            },
            .target = target,
            .optimize = optimize,
        }),
    });
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    const module_tests = b.addTest(.{ .root_module = mod });
    test_step.dependOn(&b.addRunArtifact(module_tests).step);
}

// Vulkan 1.3 consumes SPIR-V 1.6, which is what the profile names.
const spirv_profile = "spirv_1_6";

// Every `assets/shaders/*.slang` becomes one SPIR-V module carrying all of that
// file's entry points, imported under the file's stem. `shaders.zig` embeds
// them; nothing else names a path.
//
// One module per file rather than per stage, because a Slang file marked up
// with `[shader(...)]` attributes emits every entry point into one binary and
// keeps their names, and the pipeline selects a stage by name.
//
// `-matrix-layout-row-major` is what makes a zmath `Mat` arrive as itself.
//
// The flag names the source language's convention, not the SPIR-V decoration:
// measured, it produces `ColMajor`, and the default produces `RowMajor`. Which
// one is right follows from the decoration together with the operation, and
// only from both.
//
// Slang compiles `mul(vector, matrix)` to `OpMatrixTimesVector` either way, so
// the result is the SPIR-V matrix times a column vector: component r is the sum
// over c of column c's r-th component times v[c]. SPIR-V specification, 3.20
// Decoration: `ColMajor` means components within a column are contiguous, so
// SPIR-V's column c is the c-th four floats in memory, which is zmath's row c.
// The sum is then over rows, which is what zmath's `mul(v, m)` computes.
//
// With the default the same sum runs over zmath's columns instead, and every
// matrix reaches the shader transposed.
fn addShaders(b: *std.Build, module: *std.Build.Module) void {
    for (filesIn(b, "assets/shaders", ".slang")) |name| {
        const stem = name[0 .. name.len - ".slang".len];
        const command = b.addSystemCommand(&.{
            "slangc",
            "-target",
            "spirv",
            "-profile",
            spirv_profile,
            "-matrix-layout-row-major",
            // A module with one entry point has it renamed to `main` without
            // this; several keep their names. Measured on 2026-08-08, and the
            // reflection JSON reports the source name either way, so nothing
            // reading that can see the rename. `tests/reflection.zig` scans the
            // emitted words instead.
            "-fvk-use-entrypoint-name",
        });
        command.addFileArg(b.path(b.fmt("assets/shaders/{s}", .{name})));
        command.addArg("-o");
        const spirv = command.addOutputFileArg(b.fmt("{s}.spv", .{stem}));

        // The compiler's own account of the layout it produced. It is imported
        // beside the words so a test can hold the hand-written mirror against
        // it, which is the only thing that keeps the two from drifting. Nothing
        // is generated from it: the Zig side stays authored.
        command.addArg("-reflection-json");
        const reflection = command.addOutputFileArg(b.fmt("{s}.json", .{stem}));

        module.addAnonymousImport(stem, .{ .root_source_file = spirv });
        module.addAnonymousImport(b.fmt("{s}_reflection", .{stem}), .{ .root_source_file = reflection });
    }
}

fn zigFilesIn(b: *std.Build, dir_path: []const u8) [][]const u8 {
    return filesIn(b, dir_path, ".zig");
}

fn filesIn(b: *std.Build, dir_path: []const u8, extension: []const u8) [][]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return names.items,
        else => std.debug.panic("cannot open {s}/: {t}", .{ dir_path, err }),
    };
    defer dir.close(io);

    var iterator = dir.iterate();
    while (iterator.next(io) catch @panic("cannot list the directory")) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, extension)) continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }

    std.mem.sort([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b_: []const u8) bool {
            return std.mem.order(u8, a, b_) == .lt;
        }
    }.lessThan);
    return names.items;
}

fn testRoot(b: *std.Build, dir_path: []const u8) std.Build.LazyPath {
    var source: std.ArrayList(u8) = .empty;
    source.appendSlice(b.allocator, "// Generated by build.zig from the test directory. Do not edit.\ntest {\n") catch @panic("OOM");
    for (zigFilesIn(b, dir_path)) |name|
        source.print(b.allocator, "    _ = @import(\"{s}/{s}\");\n", .{ dir_path, name }) catch @panic("OOM");
    source.appendSlice(b.allocator, "}\n") catch @panic("OOM");

    const generated = b.addWriteFiles();
    _ = generated.addCopyDirectory(b.path(dir_path), dir_path, .{});
    return generated.add("test_root.zig", source.items);
}
