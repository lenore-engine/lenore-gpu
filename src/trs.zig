const zm = @import("zmath");

// Composition of a translation, rotation and scale into one matrix, in the row
// vector convention zmath uses: a point is transformed as v * M, so the factors
// read left to right in application order and scale therefore comes first.
//
// zmath/src/root.zig states that convention in its header and declares Mat as
// [4]F32x4, four rows stored consecutively.
//
// This is the only place that decides that order. Joint transforms, scene node
// transforms and asset import all have to agree on it, and a second
// implementation that disagrees shows up as animation and baked geometry drifting
// apart rather than as anything failing.
pub fn composeTransform(translation: zm.Vec, rotation: zm.Quat, scale: zm.Vec) zm.Mat {
    const scaling = zm.scaling(scale[0], scale[1], scale[2]);
    const rotating = zm.quatToMat(rotation);
    const translating = zm.translation(translation[0], translation[1], translation[2]);
    return zm.mul(zm.mul(scaling, rotating), translating);
}
