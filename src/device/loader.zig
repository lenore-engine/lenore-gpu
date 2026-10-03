const std = @import("std");
const vk = @import("vulkan");

const log = std.log.scoped(.vulkan);

// The Vulkan loader, opened by name at run time so that the loader answering is
// the one installed on the machine. One symbol is taken from it,
// vkGetInstanceProcAddr, and every other entry point is fetched through that.
pub const Loader = struct {
    // What the caller can act on is that there is no usable loader; which
    // dlopen failure produced that is a diagnostic and is logged, not returned.
    pub const Error = error{ LoaderUnavailable, MissingLoaderSymbol };

    handle: std.DynLib,

    // The loader's soname (`readelf -d` on the installed library: SONAME
    // libvulkan.so.1). The versioned name rather than `libvulkan.so`, which is
    // a symlink development packages install and a machine that only runs
    // Vulkan programs need not have.
    const library_name = "libvulkan.so.1";

    const entry_point = "vkGetInstanceProcAddr";

    pub fn open() Error!Loader {
        const library = std.DynLib.open(library_name) catch |err| {
            log.err("cannot load {s}: {t}", .{ library_name, err });
            return error.LoaderUnavailable;
        };
        return .{ .handle = library };
    }

    pub fn close(self: *Loader) void {
        self.handle.close();
    }

    pub fn getInstanceProcAddr(self: *Loader) Error!vk.PfnGetInstanceProcAddr {
        return self.handle.lookup(vk.PfnGetInstanceProcAddr, entry_point) orelse
            error.MissingLoaderSymbol;
    }
};
