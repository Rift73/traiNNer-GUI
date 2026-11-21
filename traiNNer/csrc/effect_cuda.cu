// LLVM device-only CUDA C++. The host owns buffers, FFT plans and streams.
#define DEVICE __attribute__((device, always_inline)) inline
#define KERNEL extern "C" __attribute__((global))

namespace {

constexpr int tile_w = 32;
constexpr int tile_h = 8;

struct Point { int x, y, layer; };

DEVICE Point point() {
    int tx, ty, bx, by, bz;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tx));
    asm("mov.u32 %0, %%tid.y;" : "=r"(ty));
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(bx));
    asm("mov.u32 %0, %%ctaid.y;" : "=r"(by));
    asm("mov.u32 %0, %%ctaid.z;" : "=r"(bz));
    return {bx * tile_w + tx, by * tile_h + ty, bz};
}

DEVICE int reflect101(int coordinate, int length) {
    if (static_cast<unsigned>(coordinate) < static_cast<unsigned>(length)) return coordinate;
    if (length == 1) return 0;
    const int period = 2 * (length - 1);
    const int wrapped = coordinate % period;
    const int positive = wrapped < 0 ? wrapped + period : wrapped;
    return positive < length ? positive : period - positive;
}

DEVICE long long rgb_index(int x, int y, int batch, int w, int h, bool packed) {
    const long long area = static_cast<long long>(w) * h;
    const long long pixel = static_cast<long long>(y) * w + x;
    return batch * (3 * area) + (packed ? 3 * pixel : pixel);
}

DEVICE float unit(float x) {
    return __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, x));
}

constexpr float kr = .2627f;
constexpr float kb = .0593f;
constexpr float kg = 1.0f - kr - kb;

DEVICE float luma(const float* rgb, long long stride) {
    return kr * rgb[0] + kg * rgb[stride] + kb * rgb[2 * stride];
}

template<bool ShiftLuma>
DEVICE void shift_yuv(const float* source, float* output, int w, int h, int y_dx, int y_dy,
                      int cb_dx, int cb_dy, int cr_dx, int cr_dy, bool packed, bool preserve_luma) {
    const auto [x, y, batch] = point();
    if (x >= w || y >= h) return;
    const long long stride = packed ? 1 : static_cast<long long>(w) * h;
    const long long index = rgb_index(x, y, batch, w, h, packed);
    const auto* luma_source = source + index;
    if constexpr (ShiftLuma) {
        if (y_dx || y_dy) {
            const auto position = rgb_index(reflect101(x - y_dx, w), reflect101(y - y_dy, h), batch, w, h, packed);
            luma_source = source + position;
        }
    }
    const float yy = luma(luma_source, stride);
    const auto blue = source + rgb_index(reflect101(x - cb_dx, w), reflect101(y - cb_dy, h), batch, w, h, packed);
    const auto red = source + rgb_index(reflect101(x - cr_dx, w), reflect101(y - cr_dy, h), batch, w, h, packed);
    const float db = blue[2 * stride] - luma(blue, stride);
    const float dr = red[0] - luma(red, stride);
    const float dg = -(kr * dr + kb * db) / kg;
    float scale = 1.0f;
    if (preserve_luma) {
        const float high = __builtin_fmaxf(dr, __builtin_fmaxf(dg, db));
        const float low = __builtin_fminf(dr, __builtin_fminf(dg, db));
        if (yy + high > 1.0f) scale = (1.0f - yy) / high;
        if (yy + low < 0.0f) scale = __builtin_fminf(scale, -yy / low);
    }
    output[index] = unit(yy + scale * dr);
    output[index + stride] = unit(yy + scale * dg);
    output[index + 2 * stride] = unit(yy + scale * db);
}

}  // namespace

KERNEL void effect_shift_yuv(const float* source, float* output, int w, int h,
                             int dx, int dy, int packed, int preserve_luma) {
    shift_yuv<false>(source, output, w, h, 0, 0, dx, dy, -dx, -dy, packed, preserve_luma);
}

KERNEL void effect_shift_yuv_planes(const float* source, float* output, int w, int h,
                                    int y_dx, int y_dy, int cb_dx, int cb_dy, int cr_dx, int cr_dy,
                                    int packed, int preserve_luma) {
    shift_yuv<true>(source, output, w, h, y_dx, y_dy, cb_dx, cb_dy, cr_dx, cr_dy, packed, preserve_luma);
}

KERNEL void effect_extend_even(const float* source, float* extended, int w, int h, int packed) {
    const auto [x, y, plane] = point();
    if (x >= 2 * w || y >= 2 * h) return;
    const int src_x = x < w ? x : 2 * w - 1 - x;
    const int src_y = y < h ? y : 2 * h - 1 - y;
    const long long from = rgb_index(src_x, src_y, plane / 3, w, h, packed);
    const long long channel_stride = packed ? 1 : static_cast<long long>(w) * h;
    const long long to = (static_cast<long long>(plane) * (2 * h) + y) * (2 * w) + x;
    extended[to] = source[from + (plane % 3) * channel_stride];
}

struct Complex32 { float real, imag; };

KERNEL void effect_butterworth(Complex32* spectrum, int w, int h,
                               long long sb, long long sc, long long sy, long long sx,
                               float norm_x, float norm_y, int order, float amount, int sharpen) {
    const auto [x, y, plane] = point();
    if (x >= w / 2 + 1 || y >= h) return;
    const int signed_y = y <= h / 2 ? y : h - y;
    const float fx = x * norm_x;
    const float fy = signed_y * norm_y;
    const float radius2 = fx * fx + fy * fy;
    float power = radius2;
    for (int i = 1; i < order; ++i) power *= radius2;
    const float pass = 1.0f / (1.0f + power);
    const float gain = sharpen ? 1.0f + amount * (1.0f - pass) : pass;
    auto& bin = spectrum[(plane / 3) * sb + (plane % 3) * sc + y * sy + x * sx];
    bin.real *= gain;
    bin.imag *= gain;
}

namespace {

DEVICE void finish_fft(const float* source, float* output, int w, int h, bool packed,
                       long long sb, long long sc, long long sy, long long sx, float scale) {
    const auto [x, y, batch] = point();
    if (x >= w || y >= h) return;
    const long long from = batch * sb + y * sy + x * sx;
    const long long to = rgb_index(x, y, batch, w, h, packed);
    const long long stride = packed ? 1 : static_cast<long long>(w) * h;
    output[to] = unit(scale * source[from]);
    output[to + stride] = unit(scale * source[from + sc]);
    output[to + 2 * stride] = unit(scale * source[from + 2 * sc]);
}

}  // namespace

KERNEL void effect_finish_fft(const float* source, float* output, int w, int h, int packed,
                              long long sb, long long sc, long long sy, long long sx) {
    finish_fft(source, output, w, h, packed, sb, sc, sy, sx, 1.0f);
}

KERNEL void effect_finish_cufft(const float* source, float* output, int w, int h, int fft_w, int fft_h) {
    const long long area = static_cast<long long>(fft_w) * fft_h;
    finish_fft(source, output, w, h, true, 3 * area, area, fft_w, 1, 1.0f / static_cast<float>(area));
}
