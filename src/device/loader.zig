const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");

const log = std.log.scoped(.vulkan);

// The Vulkan loader, opened by name at run time so that the loader answering is
// the one installed on the machine. One symbol is taken from it,
// vkGetInstanceProcAddr, and every other entry point is fetched through that.
pub const Loader = struct {
    // What the caller can act on is that there is no usable loader; which
    // OS failure produced that is a diagnostic and is logged, not returned.
    pub const Error = error{ LoaderUnavailable, MissingLoaderSymbol };

    handle: Library,

    const entry_point = "vkGetInstanceProcAddr";

    pub fn open() Error!Loader {
        return .{ .handle = try Library.open() };
    }

    pub fn close(self: *Loader) void {
        self.handle.close();
    }

    pub fn getInstanceProcAddr(self: *Loader) Error!vk.PfnGetInstanceProcAddr {
        return self.handle.lookup(vk.PfnGetInstanceProcAddr, entry_point) orelse
            error.MissingLoaderSymbol;
    }
};

// Chosen at comptime, so the arm a target does not use is never analysed.
// std.DynLib has no Windows implementation (lib/std/dynamic_library.zig,
// Zig 0.17.0, where every target outside Linux, Darwin and the BSDs is a
// compile error), so Windows goes to the loader in ntdll, which is what std
// itself calls to open a DLL.
const Library = switch (builtin.target.os.tag) {
    .linux => Posix,
    .windows => Windows,
    else => @compileError("no Vulkan loader for " ++ @tagName(builtin.target.os.tag)),
};

const Posix = struct {
    inner: std.DynLib,

    // The loader's soname (`readelf -d` on the installed library: SONAME
    // libvulkan.so.1). The versioned name rather than `libvulkan.so`, which is
    // a symlink development packages install and a machine that only runs
    // Vulkan programs need not have.
    const name = "libvulkan.so.1";

    fn open() Loader.Error!Posix {
        const inner = std.DynLib.open(name) catch |err| {
            log.err("cannot load {s}: {t}", .{ name, err });
            return error.LoaderUnavailable;
        };
        return .{ .inner = inner };
    }

    fn close(self: *Posix) void {
        self.inner.close();
    }

    fn lookup(self: *Posix, comptime T: type, symbol: [:0]const u8) ?T {
        return self.inner.lookup(T, symbol);
    }
};

const Windows = struct {
    const windows = std.os.windows;
    const ntdll = windows.ntdll;

    module: *anyopaque,

    // The name the Khronos loader installs into System32, and the one its
    // documentation tells applications to load. Bare, so the default search
    // order applies: a loader shipped beside the executable wins over the
    // system's, which is how a Vulkan SDK's debugging loader is substituted.
    const name = std.unicode.utf8ToUtf16LeStringLiteral("vulkan-1.dll");

    fn open() Loader.Error!Windows {
        var module: *anyopaque = undefined;
        // A null path is the default search order, and null characteristics
        // are an ordinary DLL rather than a resource-only mapping.
        const status = ntdll.LdrLoadDll(null, null, &.init(name), &module);
        if (status != .SUCCESS) {
            log.err("cannot load vulkan-1.dll: NTSTATUS 0x{x}", .{@backingInt(status)});
            return error.LoaderUnavailable;
        }
        return .{ .module = module };
    }

    fn close(self: *Windows) void {
        // Fails only for a handle that is not a loaded module, which this one
        // is until here. Nothing to recover at teardown, so it is logged.
        const status = ntdll.LdrUnloadDll(self.module);
        if (status != .SUCCESS)
            log.warn("cannot unload vulkan-1.dll: NTSTATUS 0x{x}", .{@backingInt(status)});
    }

    fn lookup(self: *Windows, comptime T: type, symbol: [:0]const u8) ?T {
        var address: *anyopaque = undefined;
        // Ordinal 0 asks for the export by name.
        const status = ntdll.LdrGetProcedureAddress(self.module, &.init(symbol), 0, &address);
        if (status != .SUCCESS) return null;
        return @ptrCast(address);
    }
};
