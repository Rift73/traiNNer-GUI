#include <cstddef>
#include <cstdint>

// The libvips region covers the original square before flip/rotate.
template <typename T, typename Store>
static int walk_oriented_crop(
    const T* pixels, std::size_t bytes, int size, int bands,
    unsigned flags, Store store) noexcept {
    if (!pixels || size <= 0 || bands <= 0 || flags > 7)
        return 1;
    const auto side = static_cast<std::size_t>(size);
    if (side > bytes / sizeof(T) / side / static_cast<std::size_t>(bands))
        return 1;
    const std::ptrdiff_t row = static_cast<std::ptrdiff_t>(side) * bands;
    const bool flip_x = flags & 1;
    const bool flip_y = flags & 2;
    const bool rotate = flags & 4;
    const auto dx = flip_x ? -bands : bands;
    const auto dy = flip_y ? -row : row;
    const auto across = rotate ? -dy : dx;
    const auto down = rotate ? dx : dy;
    const auto* origin = pixels + (flip_x ? (side - 1) * bands : 0)
                                + ((flip_y != rotate) ? (side - 1) * row : 0);
    for (int y = 0; y < size; ++y) {
        const auto* start = origin + y * down;
        auto offset = static_cast<std::size_t>(y) * side;
        for (int x = 0; x < size; ++x, ++offset) {
            store(start + x * across, offset);
        }
    }
    return 0;
}

extern "C" int pack_oriented_crop(
    const std::uint8_t* pixels, std::size_t bytes, int size, int bands,
    unsigned flags, const float* normalized, float* output) noexcept {
    if (!normalized || !output || (bands != 1 && bands != 3 && bands != 4))
        return 1;
    const auto plane = static_cast<std::size_t>(size) * size;
    return walk_oriented_crop(pixels, bytes, size, bands, flags,
        [=](const std::uint8_t* p, std::size_t offset) {
            output[offset] = normalized[p[0]];
            output[plane + offset] = normalized[p[bands == 1 ? 0 : 1]];
            output[2 * plane + offset] = normalized[p[bands == 1 ? 0 : 2]];
        });
}

// Color projection uses the ordinary loader's coefficients and rounding.
// Write its one float plane directly into the requested batch slice.
extern "C" int pack_oriented_plane_f32(
    const float* pixels, std::size_t bytes, int size, unsigned flags,
    float* output) noexcept {
    if (!output)
        return 1;
    return walk_oriented_crop(pixels, bytes, size, 1, flags,
        [=](const float* p, std::size_t offset) { output[offset] = *p; });
}
