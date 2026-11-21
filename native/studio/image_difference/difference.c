// Compare raw PNG channel values in lockstep. libpng owns filtering and inflation.
#ifndef __clang__
#error This target requires LLVM Clang.
#endif
#define _POSIX_C_SOURCE 200809L
#include <png.h>

#include <errno.h>
#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    double mae[3], rmse[3];
    uint32_t maximum[3];
    uint64_t changed;
} Summary;

typedef struct {
    uint32_t width, height, channels, depth;
    Summary pair[3];  // left/right, left/reference, right/reference
    uint64_t closer[3];  // left, right, tie
} Result;

typedef struct {
    FILE *stream;
    png_structp png;
    png_infop info;
    png_bytep row;
} Image;

typedef struct {
    jmp_buf stop;
    const char *role;
    char *message;
    size_t capacity;
    Image input[3], output;
} Job;

typedef struct {
    __uint128_t absolute[3], squared[3];
} Sums;

typedef int (*Progress)(uint32_t completed, uint32_t total);
static const char *const names[] = {"left", "right", "reference"};

static _Noreturn void fail(Job *job, const char *message) {
    snprintf(job->message, job->capacity, "%s: %s", job->role, message);
    longjmp(job->stop, 1);
}

static void png_failure(png_structp png, png_const_charp message) {
    fail(png_get_error_ptr(png), message);
}

static FILE *open_stream(Job *job, int fd, const char *mode) {
    int copy = dup(fd);
    if (copy < 0) fail(job, strerror(errno));
    FILE *stream = fdopen(copy, mode);
    if (!stream) {
        int error = errno;
        close(copy);
        fail(job, strerror(error));
    }
    return stream;
}

static void open_input(Job *job, int i, int fd, Result *result) {
    job->role = names[i];
    Image *in = &job->input[i];
    in->stream = open_stream(job, fd, "rb");
    in->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, job, png_failure, png_failure);
    if (!in->png) fail(job, "Could not allocate PNG reader.");
    in->info = png_create_info_struct(in->png);
    if (!in->info) fail(job, "Could not allocate PNG header.");
    png_init_io(in->png, in->stream);
    png_set_crc_action(in->png, PNG_CRC_ERROR_QUIT, PNG_CRC_ERROR_QUIT);
    // Discard optional metadata without expanding profiles or text. Retain
    // the small animation header so an APNG cannot be mistaken for one frame.
    png_set_keep_unknown_chunks(in->png, PNG_HANDLE_CHUNK_NEVER, NULL, -1);
    png_set_keep_unknown_chunks(in->png, PNG_HANDLE_CHUNK_ALWAYS, (png_const_bytep)"acTL", 1);
    png_set_chunk_malloc_max(in->png, 65536);
    png_read_info(in->png, in->info);
    png_unknown_chunkp chunks = NULL;
    int count = png_get_unknown_chunks(in->png, in->info, &chunks);
    for (int k = 0; k < count; ++k)
        if (!memcmp(chunks[k].name, "acTL", 4)) fail(job, "Animated PNG is unsupported.");

    int color = png_get_color_type(in->png, in->info);
    uint32_t depth = png_get_bit_depth(in->png, in->info);
    if ((color != PNG_COLOR_TYPE_GRAY && color != PNG_COLOR_TYPE_RGB) ||
        (depth != 8 && depth != 16) || png_get_valid(in->png, in->info, PNG_INFO_tRNS))
        fail(job, "Expected opaque 8- or 16-bit grayscale or RGB PNG.");
    if (png_get_interlace_type(in->png, in->info) != PNG_INTERLACE_NONE)
        fail(job, "Interlaced PNG is unsupported by the row-streaming diagnostic.");

    uint32_t width = png_get_image_width(in->png, in->info);
    uint32_t height = png_get_image_height(in->png, in->info);
    uint32_t channels = png_get_channels(in->png, in->info);
    if (i == 0) {
        result->width = width;
        result->height = height;
        result->channels = channels;
        result->depth = depth;
    } else {
        if (result->width != width || result->height != height)
            fail(job, "PNG dimensions differ from the left image.");
        if (result->depth != depth || result->channels != channels)
            fail(job, "PNG channel layout or bit depth differs from the left image.");
    }
    in->row = malloc(png_get_rowbytes(in->png, in->info));
    if (!in->row) fail(job, "Could not allocate a PNG scanline.");
}

static void open_output(Job *job, int fd, const Result *result, uint32_t width, uint32_t height) {
    job->role = "absolute-difference.png";
    Image *out = &job->output;
    out->stream = open_stream(job, fd, "wb");
    out->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, job, png_failure, png_failure);
    if (!out->png) fail(job, "Could not allocate PNG writer.");
    out->info = png_create_info_struct(out->png);
    if (!out->info) fail(job, "Could not allocate PNG header.");
    png_init_io(out->png, out->stream);
    png_set_IHDR(out->png, out->info, width, height, result->depth,
                 result->channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_text units = {.compression = PNG_TEXT_COMPRESSION_NONE, .key = "Description",
                      .text = "Per-channel absolute difference in original stored integer sample levels."};
    png_set_text(out->png, out->info, &units, 1);
    png_write_info(out->png, out->info);
}

static uint32_t sample(const png_byte *p, unsigned bytes) {
    return bytes == 1 ? p[0] : ((uint32_t)p[0] << 8) | p[1];
}

static uint32_t accumulate(uint32_t a, uint32_t b, unsigned ch, Sums *sum, Summary *summary) {
    uint32_t d = a >= b ? a - b : b - a;
    sum->absolute[ch] += d;
    sum->squared[ch] += d * d;  // 65535 squared fits in uint32_t.
    if (d > summary->maximum[ch]) summary->maximum[ch] = d;
    return d;
}

static void compare_row(Job *job, const Result *format, Summary pair[3], uint64_t closer[3],
                        Sums sums[3], uint32_t x, uint32_t width, int with_reference) {
    unsigned bytes = format->depth / 8;
    size_t stride = format->channels * bytes;
    size_t begin = (size_t)x * stride, end = begin + (size_t)width * stride;
    for (size_t offset = begin; offset < end; offset += stride) {
        uint32_t different = 0;
        uint64_t left_error = 0, right_error = 0;
        for (unsigned ch = 0; ch < format->channels; ++ch) {
            size_t at = offset + ch * bytes;
            png_bytep out = job->input[0].row + at;
            uint32_t a = sample(out, bytes), b = sample(job->input[1].row + at, bytes);
            uint32_t d = accumulate(a, b, ch, sums, pair);
            different |= d;
            if (with_reference) {
                uint32_t truth = sample(job->input[2].row + at, bytes);
                uint32_t l = accumulate(a, truth, ch, sums + 1, pair + 1);
                uint32_t r = accumulate(b, truth, ch, sums + 2, pair + 2);
                left_error += l * l;
                right_error += r * r;
            }
            // The left sample has now been consumed by every comparison.
            if (bytes == 2) *out++ = (png_byte)(d >> 8);
            *out = (png_byte)d;
        }
        pair[0].changed += different != 0;
        if (with_reference) {
            pair[1].changed += left_error != 0;
            pair[2].changed += right_error != 0;
            ++closer[left_error < right_error ? 0 : right_error < left_error ? 1 : 2];
        }
    }
    job->role = "absolute-difference.png";
    png_write_row(job->output.png, job->input[0].row + begin);
}

static int run(Job *job, const int fds[3], int output_fd, const uint32_t *region,
               Result *result, Progress progress) {
    // Mutable ownership lives in the caller's Job across any libpng longjmp.
    if (setjmp(job->stop)) return -1;
    const int count = fds[2] < 0 ? 2 : 3;
    for (int i = 0; i < count; ++i) open_input(job, i, fds[i], result);
    uint32_t x = region ? region[0] : 0, y = region ? region[1] : 0;
    uint32_t width = region ? region[2] : result->width;
    uint32_t height = region ? region[3] : result->height;
    if (!width || !height || x > result->width || width > result->width - x ||
        y > result->height || height > result->height - y) {
        job->role = "region";
        fail(job, "Rectangle must be nonempty and inside the source grid.");
    }

    open_output(job, output_fd, result, width, height);
    Sums sums[3] = {0};
    for (uint32_t row = 0; row < result->height; ++row) {
        if (progress && row % 16 == 0 && progress(row, result->height)) return 1;
        for (int i = 0; i < count; ++i) {
            job->role = names[i];
            png_read_row(job->input[i].png, job->input[i].row, NULL);
        }
        if (row >= y && row - y < height)
            compare_row(job, result, result->pair, result->closer, sums, x, width, count == 3);
    }
    for (int i = 0; i < count; ++i) {
        job->role = names[i];
        png_read_end(job->input[i].png, NULL);
    }
    job->role = "absolute-difference.png";
    png_write_end(job->output.png, NULL);
    if (fflush(job->output.stream) != 0) fail(job, strerror(errno));
    if (progress && progress(result->height, result->height)) return 1;
    long double pixels = (uint64_t)width * height;
    for (int i = 0; i < (count == 3 ? 3 : 1); ++i) {
        for (unsigned ch = 0; ch < result->channels; ++ch) {
            result->pair[i].mae[ch] = (double)((long double)sums[i].absolute[ch] / pixels);
            result->pair[i].rmse[ch] = (double)sqrtl((long double)sums[i].squared[ch] / pixels);
        }
    }
    return 0;
}

int studio_png_difference_v1(int left, int right, int reference, int output_fd,
                             const uint32_t *region, Result *result, Progress progress,
                             char *error, size_t capacity) {
    Job job = {.message = error, .capacity = capacity};
    memset(result, 0, sizeof(*result));
    const int fds[] = {left, right, reference};
    int status = run(&job, fds, output_fd, region, result, progress);
    for (unsigned i = 0; i < 3; ++i) {
        Image *in = &job.input[i];
        png_destroy_read_struct(&in->png, &in->info, NULL);
        free(in->row);
        if (in->stream) fclose(in->stream);
    }
    png_destroy_write_struct(&job.output.png, &job.output.info);
    if (job.output.stream && fclose(job.output.stream) != 0 && status == 0) {
        snprintf(error, capacity, "absolute-difference.png: %s", strerror(errno));
        status = -1;
    }
    return status;
}
