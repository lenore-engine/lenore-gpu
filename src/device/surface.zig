const platform = @import("lenore-platform");
const vk = @import("vulkan");

const Context = @import("context.zig").Context;

const InstanceWrapper = vk.InstanceWrapper;

// One error set per window system the platform's union declares, because the
// switch below is exhaustive over it.
const CreateError = InstanceWrapper.CreateWaylandSurfaceKHRError;

pub const InitError = error{
    // The device was chosen against the display and this surface still cannot
    // be presented to from the family that presents. Nothing here repairs it:
    // the device is already created and every other window is running on it.
    PresentationUnsupported,
} || CreateError || InstanceWrapper.GetPhysicalDeviceSurfaceSupportKHRError;

// One window's Vulkan surface.
//
// Its own type because its lifetime is neither the context's nor the
// swapchain's. A resize destroys the swapchain and keeps this; closing a window
// destroys this and keeps the context, which the rest of the process is still
// drawing on.
//
// The context is a parameter rather than a field. It is wanted at creation and
// at destruction and nowhere in between, and a pointer stored for two calls is
// a pointer that must stay valid for everything between them.
pub const Surface = struct {
    handle: vk.SurfaceKHR,

    pub fn init(context: *const Context, handles: platform.NativeHandles) InitError!Surface {
        const handle = try create(context, handles);
        errdefer context.instance.destroySurfaceKHR(handle, null);

        // Vulkan specification, VUID-VkSwapchainCreateInfoKHR-surface-01270: a
        // swapchain's surface must be one the device supports as determined by
        // this call. The device was picked on the display's presentation
        // support, which is one answer for every surface on that display, so
        // this confirms rather than chooses. One cold call per window.
        const supported = try context.instance.getPhysicalDeviceSurfaceSupportKHR(
            context.physical_device,
            context.present_queue.family,
            handle,
        );
        if (supported != .true) return error.PresentationUnsupported;

        return .{ .handle = handle };
    }

    pub fn deinit(self: Surface, context: *const Context) void {
        context.instance.destroySurfaceKHR(self.handle, null);
    }
};

fn create(context: *const Context, handles: platform.NativeHandles) CreateError!vk.SurfaceKHR {
    return switch (handles) {
        .wayland => |wayland| context.instance.createWaylandSurfaceKHR(&.{
            .display = @ptrCast(wayland.display),
            .surface = @ptrCast(wayland.surface),
        }, null),
    };
}
