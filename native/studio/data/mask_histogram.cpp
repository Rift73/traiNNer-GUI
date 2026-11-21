// Count one noninterlaced PNG frame without expanding a palette or retaining
// the decoded raster. The caller has already checked the image with Pillow.
#include <png.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <unistd.h>

namespace {
struct Input {
    int fd;
    off_t offset = 0;
};

using Cancelled = int (*)();

void read_png(png_structp png, png_bytep destination, png_size_t length) {
    auto& input = *static_cast<Input*>(png_get_io_ptr(png));
    while (length != 0) {
        const auto count = pread(input.fd, destination, length, input.offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            png_error(png, "Could not read PNG data");
        }
        input.offset += count;
        destination += count;
        length -= static_cast<png_size_t>(count);
    }
}

void fail(png_structp png, png_const_charp) {
    png_longjmp(png, 1);
}

void warn(png_structp, png_const_charp) {}

int decode(
    png_structp png, png_infop info, Input& input, png_bytep row,
    png_uint_32 width, png_uint_32 height, std::uint64_t* counts,
    std::uint16_t* values, Cancelled cancelled
) {
    // libpng can longjmp. Resource ownership stays in the caller.
    if (setjmp(png_jmpbuf(png))) {
        return 0;
    }
    png_set_read_fn(png, &input, read_png);
    png_read_info(png, info);

    const auto color = png_get_color_type(png, info);
    const auto bits = png_get_bit_depth(png, info);
    if (png_get_image_width(png, info) != width ||
        png_get_image_height(png, info) != height ||
        png_get_interlace_type(png, info) != PNG_INTERLACE_NONE ||
        (color != PNG_COLOR_TYPE_PALETTE && color != PNG_COLOR_TYPE_GRAY)) {
        return 0;
    }
    if (bits < 8) {
        if (color == PNG_COLOR_TYPE_GRAY && bits != 1) {
            // Pillow exposes 2- and 4-bit grayscale as L with scaled values.
            png_set_expand_gray_1_2_4_to_8(png);
        } else {
            png_set_packing(png);
        }
    }
    png_read_update_info(png, info);

    const bool wide = bits == 16;
    const auto row_bytes = static_cast<std::size_t>(width) * (wide ? 2 : 1);
    if (png_get_rowbytes(png, info) != row_bytes) {
        return 0;
    }
    std::fill_n(counts, wide ? 65536 : 256, 0);
    int distinct = 0;
    for (png_uint_32 y = 0; y < height; ++y) {
        if (cancelled && y % 128 == 0 && cancelled()) {
            return -1;
        }
        png_read_row(png, row, nullptr);
        if (wide) {
            for (std::size_t x = 0; x < row_bytes; x += 2) {
                const auto value = static_cast<std::uint16_t>((row[x] << 8) | row[x + 1]);
                if (counts[value]++ == 0) {
                    values[distinct++] = value;
                }
            }
        } else {
            for (std::size_t x = 0; x < row_bytes; ++x) {
                const auto value = row[x];
                if (counts[value]++ == 0) {
                    values[distinct++] = value;
                }
            }
        }
    }
    png_read_end(png, nullptr);
    return distinct;
}
}  // namespace

// Return the occupied-bin count, 0 for Pillow fallback, or -1 on cancellation.
// Both output arrays have capacity 65536. pread leaves Pillow's file position alone.
extern "C" int studio_png_histogram_v1(
    int fd, png_uint_32 width, png_uint_32 height, std::uint64_t* counts,
    std::uint16_t* values, Cancelled cancelled
) noexcept {
    auto row = std::unique_ptr<png_byte[]>(new (std::nothrow) png_byte[
        static_cast<std::size_t>(width) * 2
    ]);
    if (!row) {
        return 0;
    }
    auto* png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, fail, warn);
    if (!png) {
        return 0;
    }
    auto* info = png_create_info_struct(png);
    Input input{fd};
    const int result = info
        ? decode(png, info, input, row.get(), width, height, counts, values, cancelled)
        : 0;
    png_destroy_read_struct(&png, &info, nullptr);
    return result;
}
