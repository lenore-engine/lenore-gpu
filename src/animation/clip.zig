const std = @import("std");
const zm = @import("zmath");

const Allocator = std.mem.Allocator;

pub const Interpolation = enum {
    linear,
    step,
    // glTF 2.0 defines this as a Hermite spline with in and out tangents. The
    // tangents are not carried here and the sampling below blends linearly
    // instead, so a clip authored with it plays smoothly but not exactly as
    // authored. Rejecting such an asset would be worse: the mode is common.
    cubicspline,
};

// The property an animation channel drives, and the tag of the track that
// drives it.
pub const TargetPath = enum { translation, rotation, scale, weights };

pub fn Keyframe(comptime T: type) type {
    return struct {
        time: f32,
        value: T,
    };
}

// Morph weights are per target and per keyframe, laid out with the target index
// varying fastest: keyframe k's weights are values[k * width ..][0..width].
pub const WeightTrack = struct {
    times: []f32,
    values: []f32,
    width: u32,
};

// A track carries exactly the data its path needs, so a channel cannot name one
// path and carry another's keys. The tag is the only thing that says which of
// the four a track is, because zmath declares Vec and Quat as the same type and
// the payloads are therefore indistinguishable.
pub const Track = union(TargetPath) {
    translation: []Keyframe(zm.Vec),
    rotation: []Keyframe(zm.Quat),
    scale: []Keyframe(zm.Vec),
    weights: WeightTrack,

    pub fn deinit(self: *Track, allocator: Allocator) void {
        switch (self.*) {
            .translation, .scale => |keys| allocator.free(keys),
            .rotation => |keys| allocator.free(keys),
            .weights => |track| {
                allocator.free(track.times);
                allocator.free(track.values);
            },
        }
        self.* = undefined;
    }

    // The times of a track's first and last key, or null when it has none.
    // Both ends come from one call because every caller wants both, and the
    // keyframe arms share one inline body: their payload types differ but the
    // work does not, and the duplication that produces is the compiler's.
    pub const Span = struct { first: f32, last: f32 };

    pub fn span(self: Track) ?Span {
        switch (self) {
            inline .translation, .rotation, .scale => |keys| {
                if (keys.len == 0) return null;
                return .{ .first = keys[0].time, .last = keys[keys.len - 1].time };
            },
            .weights => |track| {
                if (track.times.len == 0) return null;
                const last = track.times.len - 1;
                return .{ .first = track.times[0], .last = track.times[last] };
            },
        }
    }
};

pub const Channel = struct {
    track: Track,
    // Index into the arrays the caller passes to sample. What a slot means
    // belongs to that caller: a joint for a skeleton, a node for rigid
    // animation.
    target_slot: u32,
    interpolation: Interpolation,

    pub fn deinit(self: *Channel, allocator: Allocator) void {
        self.track.deinit(allocator);
    }
};

pub const InitError = Allocator.Error;
pub const SampleError = error{TargetArrayTooSmall};

pub const Animation = struct {
    // Owned, along with every track inside them.
    channels: []Channel,
    name: []const u8,
    // The keyed window. Playback loops over it rather than over [0, duration]:
    // a clip authored with an offset timeline would otherwise hold its first
    // key for the whole offset on every iteration.
    start_time: f32,
    duration: f32,
    // One past the largest slot any channel targets, so a single check in sample
    // covers every indexing below it.
    //
    // Wider than the slot it is derived from: a channel may name the largest
    // u32 there is, and adding one to that in its own width wraps to zero, which
    // in the shipping build would turn the check below into a permission.
    slot_count: u64,

    pub fn init(
        allocator: Allocator,
        channels: []Channel,
        name: []const u8,
    ) InitError!Animation {
        var start_time: f32 = std.math.floatMax(f32);
        var duration: f32 = 0.0;
        var slot_count: u64 = 0;
        for (channels) |*channel| {
            if (channel.track.span()) |keyed| {
                start_time = @min(start_time, keyed.first);
                duration = @max(duration, keyed.last);
            }
            slot_count = @max(slot_count, @as(u64, channel.target_slot) + 1);
        }
        // An empty track must not contribute a zero to the start: it would win
        // the minimum and hide a genuinely offset timeline.
        if (start_time == std.math.floatMax(f32)) start_time = 0.0;

        const owned_name = try allocator.dupe(u8, name);
        return .{
            .channels = channels,
            .name = owned_name,
            .start_time = start_time,
            .duration = duration,
            .slot_count = slot_count,
        };
    }

    pub fn deinit(self: *Animation, allocator: Allocator) void {
        for (self.channels) |*channel| channel.deinit(allocator);
        allocator.free(self.channels);
        allocator.free(self.name);
        self.* = undefined;
    }

    // The length of the looping window. Zero for a clip that holds a single
    // pose, and never a divisor without that test.
    pub fn loopSpan(self: *const Animation) f32 {
        return self.duration - self.start_time;
    }

    // Maps a monotonically increasing elapsed time onto a looping cursor inside
    // the keyed window, and is idempotent for an input already inside one span.
    //
    // Looping is not applied by sampling, because whether time wraps belongs to
    // playback and not to the clip. A looping caller composes the two; one
    // driving an absolute timeline, a cutscene or an editor scrubber, samples
    // directly and gets the clamp at either end that the keys already define.
    //
    // The window starts at the first key rather than at zero. An offset in an
    // exported clip is an artefact of the timeline it was authored on, and
    // looping over [0, duration] would hold the first key for that offset on
    // every iteration. A deliberate delay belongs to whoever advances the clock.
    pub fn cursorAt(self: *const Animation, elapsed: f32) f32 {
        const span = self.loopSpan();
        // A zero span would make the modulus below a division by zero.
        return if (span > 0.0) self.start_time + @mod(elapsed, span) else self.start_time;
    }

    // Samples every transform channel into the caller's arrays at a time on this
    // clip's own timeline. A looping caller passes cursorAt(elapsed).
    //
    // The bounds are checked once here rather than per channel, which is what
    // lets the indexing below stand without one: target slots come from an
    // asset, and in the shipping build an unchecked one writes outside the
    // array.
    pub fn sample(
        self: *const Animation,
        time: f32,
        translations: []zm.Vec,
        rotations: []zm.Quat,
        scales: []zm.Vec,
    ) SampleError!void {
        if (translations.len < self.slot_count or
            rotations.len < self.slot_count or
            scales.len < self.slot_count)
            return error.TargetArrayTooSmall;

        for (self.channels) |*channel| {
            const slot = channel.target_slot;
            switch (channel.track) {
                .translation => |keys| translations[slot] = sampleKeys(.linear, keys, time, channel.interpolation),
                .scale => |keys| scales[slot] = sampleKeys(.linear, keys, time, channel.interpolation),
                .rotation => |keys| rotations[slot] = sampleKeys(.spherical, keys, time, channel.interpolation),
                // Morph weights are per object rather than per slot, so they
                // are sampled by their own call.
                .weights => {},
            }
        }
    }

    // Samples the first weight track into out, on the same timeline as sample.
    // A clip without one leaves every weight at zero, which is the neutral
    // blend.
    pub fn sampleWeights(self: *const Animation, time: f32, out: []f32) void {
        for (self.channels) |*channel| {
            switch (channel.track) {
                .weights => |track| {
                    if (track.width == 0 or track.times.len == 0) break;
                    sampleWeightTrack(track, time, channel.interpolation, out);
                    return;
                },
                else => {},
            }
        }
        @memset(out, 0);
    }
};

// How two keys are blended. It is a parameter rather than a property of the
// values because zmath/src/root.zig declares both Vec and Quat as F32x4, so
// nothing about a keyframe says which of the two it holds. The tag of the track
// it came from does.
const Blending = enum {
    // A straight line, for translations and scales.
    linear,
    // The short arc of the unit sphere, for rotations. A straight line between
    // two quaternions leaves the sphere, which shortens the rotation and varies
    // its angular speed.
    spherical,
};

// The interval whose left key is at or before time and whose right key is after
// it. Only reached with at least two keys and a time strictly inside them, so
// the search always terminates on a real interval.
fn findInterval(keys: []const Keyframe(zm.Vec), time: f32) struct { usize, usize } {
    var left: usize = 0;
    var right: usize = keys.len - 1;
    while (right - left > 1) {
        const middle = left + (right - left) / 2;
        if (keys[middle].time <= time) left = middle else right = middle;
    }
    return .{ left, right };
}

fn sampleKeys(
    comptime blending: Blending,
    keys: []const Keyframe(zm.Vec),
    time: f32,
    interpolation: Interpolation,
) zm.Vec {
    if (keys.len == 0) return switch (blending) {
        .linear => zm.f32x4(0.0, 0.0, 0.0, 0.0),
        .spherical => zm.qidentity(),
    };
    // A clip clamps outside its own keys. Playback loops over the keyed window,
    // so this is reached for a time exactly at either end.
    if (keys.len == 1 or time <= keys[0].time) return keys[0].value;
    if (time >= keys[keys.len - 1].time) return keys[keys.len - 1].value;

    const interval = findInterval(keys, time);
    const left = keys[interval[0]];
    const right = keys[interval[1]];
    if (interpolation == .step) return left.value;

    const fraction = (time - left.time) / (right.time - left.time);
    return switch (blending) {
        .linear => zm.lerp(left.value, right.value, fraction),
        .spherical => zm.slerp(left.value, right.value, fraction),
    };
}

// Weight tracks hold a handful of keys, so this scans rather than searching.
fn sampleWeightTrack(
    track: WeightTrack,
    time: f32,
    interpolation: Interpolation,
    out: []f32,
) void {
    const width = @min(@as(usize, track.width), out.len);
    const count = track.times.len;

    if (count == 1 or time <= track.times[0]) {
        @memcpy(out[0..width], track.values[0..width]);
        return;
    }
    if (time >= track.times[count - 1]) {
        @memcpy(out[0..width], track.values[(count - 1) * track.width ..][0..width]);
        return;
    }

    var index: usize = 0;
    while (index + 1 < count and track.times[index + 1] <= time) : (index += 1) {}
    const span = track.times[index + 1] - track.times[index];
    const fraction: f32 = if (interpolation == .step or span <= 0.0)
        0.0
    else
        (time - track.times[index]) / span;

    const left = index * track.width;
    const right = (index + 1) * track.width;
    for (0..width) |component| {
        const from = track.values[left + component];
        const to = track.values[right + component];
        out[component] = from + (to - from) * fraction;
    }
}
