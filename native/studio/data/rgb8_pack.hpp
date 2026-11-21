#pragma once

#include <cstddef>
#include <cstdint>

template <bool Planar, typename Offset>
inline void pack_rgb8_chw(
    const std::uint8_t* pixels, std::size_t height, std::size_t width,
    Offset row_stride, Offset pixel_stride, Offset channel_stride,
    const float* normalized, float* output
) noexcept {
    const auto channel_step = Planar ? height * width : 1;
    constexpr auto output_step = Planar ? 1 : 3;
    for (std::size_t y = 0; y < height; ++y) {
        const auto* row = pixels + static_cast<Offset>(y) * row_stride;
        auto* red = output + y * width * output_step;
        auto* green = red + channel_step;
        auto* blue = green + channel_step;
        for (std::size_t x = 0; x < width; ++x) {
            const auto* pixel = row + static_cast<Offset>(x) * pixel_stride;
            const auto out = x * output_step;
            red[out] = normalized[pixel[0]];
            green[out] = normalized[pixel[channel_stride]];
            blue[out] = normalized[pixel[2 * channel_stride]];
        }
    }
}
