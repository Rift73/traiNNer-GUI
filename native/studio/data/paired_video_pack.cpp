#include "rgb8_pack.hpp"

struct RgbFrame {
    const std::uint8_t* pixels;
    std::size_t row_stride;
    std::size_t pixel_stride;
};

// The caller supplies normalized float values from the trainer's conversion.
extern "C" void pack_rgb8_tchw(const RgbFrame* frames, std::size_t count,
                               std::size_t height, std::size_t width,
                               const float* normalized, float* output) noexcept {
    const auto plane_size = height * width;
    for (std::size_t frame_index = 0; frame_index < count; ++frame_index) {
        const auto& frame = frames[frame_index];
        pack_rgb8_chw<true>(
            frame.pixels, height, width, frame.row_stride, frame.pixel_stride,
            std::size_t{1}, normalized, output
        );
        output += 3 * plane_size;
    }
}
