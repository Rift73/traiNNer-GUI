#include "rgb8_pack.hpp"

#ifndef __clang__
#error LLVM Clang is required.
#endif

// One logical RGB view, possibly cropped, flipped, or transposed. The
// interleaved output mode preserves the existing CHW tensor strides.
extern "C" void pack_rgb8_sample_v1(
    const std::uint8_t* pixels, std::size_t height, std::size_t width,
    std::ptrdiff_t row_stride, std::ptrdiff_t pixel_stride,
    std::ptrdiff_t channel_stride, int planar, const float* normalized,
    float* output
) noexcept {
    if (planar) {
        pack_rgb8_chw<true>(
            pixels, height, width, row_stride, pixel_stride, channel_stride, normalized, output
        );
    } else {
        pack_rgb8_chw<false>(
            pixels, height, width, row_stride, pixel_stride, channel_stride, normalized, output
        );
    }
}
