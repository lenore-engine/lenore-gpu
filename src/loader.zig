const builtin = @import("builtin");
const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.vulkan);

// The Vulkan loader, opened by name at run time so that the loader answering is
// the one installed on the machine. One symbol is taken from it,
// vkGetInstanceProcAddr, and every other entry point is fetched through that.
//
// std.DynLib carries the POSIX side and cannot carry the other: in 0.16 its
// platform switch has no Windows arm, and std has no LoadLibraryW to build one
// from either, std/os/windows/kernel32.zig binding CreateProcessW alone. Hence
// the three kernel32 declarations at the bottom of this file.
pub const Loader = struct {
    // What the caller can act on is that there is no usable loader; which
    // dlopen failure produced that is a diagnostic and is logged, not returned.
    pub const Error = error{ LoaderUnavailable, MissingLoaderSymbol };

    handle: Handle,

    const Handle = switch (builtin.os.tag) {
        .windows => std.os.windows.HMODULE,
        else => std.DynLib,
    };

    // GLFW 3.4 documents these as the loader's standard names, under
    // glfwInitVulkanLoader in GLFW/glfw3.h: "vulkan-1.dll" on Windows and
    // "libvulkan.so.1" on Linux and the other Unix-likes. Its own loader opens
    // them by exactly these names.
    const library_name = switch (builtin.os.tag) {
        .windows => "vulkan-1.dll",
        else => "libvulkan.so.1",
    };

    const entry_point = "vkGetInstanceProcAddr";

    pub fn open() Error!Loader {
        switch (builtin.os.tag) {
            .windows => {
                const wide = comptime std.unicode.utf8ToUtf16LeStringLiteral(library_name);
                const module = LoadLibraryW(wide) orelse {
                    log.err("cannot load {s}", .{library_name});
                    return error.LoaderUnavailable;
                };
                return .{ .handle = module };
            },
            else => {
                const library = std.DynLib.open(library_name) catch |err| {
                    log.err("cannot load {s}: {t}", .{ library_name, err });
                    return error.LoaderUnavailable;
                };
                return .{ .handle = library };
            },
        }
    }

    pub fn close(self: *Loader) void {
        switch (builtin.os.tag) {
            // Teardown, so there is nothing to do with a failure to report.
            .windows => _ = FreeLibrary(self.handle),
            else => self.handle.close(),
        }
    }

    pub fn getInstanceProcAddr(self: *Loader) Error!vk.PfnGetInstanceProcAddr {
        switch (builtin.os.tag) {
            .windows => {
                const address = GetProcAddress(self.handle, entry_point) orelse
                    return error.MissingLoaderSymbol;
                return @ptrCast(address);
            },
            else => return self.handle.lookup(vk.PfnGetInstanceProcAddr, entry_point) orelse
                error.MissingLoaderSymbol,
        }
    }
};

const HMODULE = std.os.windows.HMODULE;

extern "kernel32" fn LoadLibraryW(file_name: [*:0]const u16) callconv(.winapi) ?HMODULE;
extern "kernel32" fn FreeLibrary(module: HMODULE) callconv(.winapi) std.os.windows.BOOL;
extern "kernel32" fn GetProcAddress(module: HMODULE, proc_name: [*:0]const u8) callconv(.winapi) ?std.os.windows.FARPROC;
