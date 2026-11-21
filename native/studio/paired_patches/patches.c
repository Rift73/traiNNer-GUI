// Extract a registered LQ/GT grid by copying stored PNG scanline spans.
#ifndef __clang__
#error This target requires LLVM Clang.
#endif
#define _POSIX_C_SOURCE 200809L
#include <png.h>

#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t width, height, channels, depth;
} Format;

typedef struct {
    FILE *file;
    png_structp png;
    png_infop info, end;
    size_t rowbytes;
} Reader;

// name == NULL is a cancellation checkpoint. Otherwise one PNG is closed.
typedef int (*Notify)(int role, uint32_t x, uint32_t y, const char *name);

typedef struct {
    Reader in[2];
    Format *format;
    FILE *file;
    png_structp writer;
    png_infop header;
    png_bytep band;
    int role;
    Notify notify;
    jmp_buf stop;
    char *error;
    size_t capacity;
} Job;

static _Noreturn void fail(Job *j, const char *message) {
    snprintf(j->error, j->capacity, "%s: %s", j->role == 0 ? "LQ" : "GT", message);
    longjmp(j->stop, 1);
}

static void png_failure(png_structp png, png_const_charp message) {
    fail(png_get_error_ptr(png), message);
}

static void event(Job *j, uint32_t x, uint32_t y, const char *name) {
    if (j->notify(j->role, x, y, name)) longjmp(j->stop, 2);
}

static FILE *as_file(Job *j, int owned_fd, const char *mode) {
    if (owned_fd < 0) fail(j, strerror(errno));
    FILE *fp = fdopen(owned_fd, mode);
    if (!fp) {
        const int reason = errno;
        close(owned_fd);
        fail(j, strerror(reason));
    }
    return fp;
}

// Carry encoded color/density metadata without applying a transfer function
// or decompressing a profile. Geometry/animation chunks require other semantics.
static const png_byte kept[] =
    "iCCP\0sRGB\0gAMA\0cHRM\0sBIT\0pHYs\0cICP\0mDCV\0cLLI\0acTL\0eXIf\0sTER\0oFFs";

static void metadata(Job *j, png_infop info, int writing, int after_pixels) {
    Reader *in = j->in + j->role;
    png_unknown_chunkp chunks = NULL;
    int count = png_get_unknown_chunks(in->png, info, &chunks);
    for (int i = 0; i < count; ++i) {
        png_unknown_chunkp c = chunks + i;
        if (!memcmp(c->name, "acTL", 4)) fail(j, "Animated PNG is unsupported.");
        if (!memcmp(c->name, "eXIf", 4) || !memcmp(c->name, "sTER", 4) || !memcmp(c->name, "oFFs", 4))
            fail(j, "PNG with EXIF, stereo layout or image offsets is unsupported.");
        if (after_pixels) fail(j, "Unexpected color or layout metadata after image data.");
        if (writing) png_write_chunk(j->writer, c->name, c->data, c->size);
    }
}

static void open_reader(Job *j, int fd) {
    Reader *in = j->in + j->role;
    Format *f = j->format + j->role;
    in->file = as_file(j, dup(fd), "rb");
    in->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, j, png_failure, png_failure);
    if (!in->png) fail(j, "Could not allocate PNG reader.");
    in->info = png_create_info_struct(in->png);
    in->end = png_create_info_struct(in->png);
    if (!in->info || !in->end) fail(j, "Could not allocate PNG header.");
    png_init_io(in->png, in->file);
    png_set_crc_action(in->png, PNG_CRC_ERROR_QUIT, PNG_CRC_ERROR_QUIT);
    png_set_keep_unknown_chunks(in->png, PNG_HANDLE_CHUNK_NEVER, NULL, -1);
    png_set_keep_unknown_chunks(in->png, PNG_HANDLE_CHUNK_ALWAYS, kept, sizeof(kept) / 5);
    png_read_info(in->png, in->info);
    metadata(j, in->info, 0, 0);
    const int color = png_get_color_type(in->png, in->info);
    f->depth = png_get_bit_depth(in->png, in->info);
    if ((color != PNG_COLOR_TYPE_GRAY && color != PNG_COLOR_TYPE_RGB && color != PNG_COLOR_TYPE_RGBA) ||
        (f->depth != 8 && f->depth != 16) || png_get_valid(in->png, in->info, PNG_INFO_tRNS))
        fail(j, "Expected grayscale, RGB or RGBA PNG with 8 or 16 bits and no color-key transparency.");
    if (png_get_interlace_type(in->png, in->info) != PNG_INTERLACE_NONE)
        fail(j, "Interlaced PNG is unsupported.");
    f->width = png_get_image_width(in->png, in->info);
    f->height = png_get_image_height(in->png, in->info);
    f->channels = png_get_channels(in->png, in->info);
    in->rowbytes = png_get_rowbytes(in->png, in->info);
}

static void write_patch(Job *j, int dirfd, uint32_t x, uint32_t y,
                        uint32_t side, uint32_t factor) {
    Reader *in = j->in + j->role;
    const Format *f = j->format + j->role;
    char name[64];
    snprintf(name, sizeof(name), "y%u-x%u.png", y, x);
    j->file = as_file(j, openat(dirfd, name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600), "wb");
    j->writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, j, png_failure, png_failure);
    if (!j->writer) fail(j, "Could not allocate PNG writer.");
    j->header = png_create_info_struct(j->writer);
    if (!j->header) fail(j, "Could not allocate output header.");
    png_init_io(j->writer, j->file);
    png_set_compression_level(j->writer, 3);
    png_set_IHDR(j->writer, j->header, side, side, f->depth, png_get_color_type(in->png, in->info),
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(j->writer, j->header);
    metadata(j, in->info, 1, 0);
    const size_t left = (size_t)x * factor * f->channels * (f->depth / 8);
    for (uint32_t row = 0; row < side; ++row) {
        if (row && row % 128 == 0) event(j, x, y, NULL);
        size_t slot = (y * factor + row) % side;
        png_write_row(j->writer, j->band + slot * in->rowbytes + left);
    }
    png_write_end(j->writer, NULL);
    png_destroy_write_struct(&j->writer, &j->header);
    int status = fclose(j->file);
    j->file = NULL;
    if (status) fail(j, strerror(errno));
    event(j, x, y, name);
}

static uint32_t next_origin(uint32_t value, uint32_t stride, uint32_t last) {
    return last - value < stride ? last : value + stride;
}

static void extract_role(Job *j, int dirfd, uint32_t crop, uint32_t stride, uint32_t factor) {
    Reader *in = j->in + j->role;
    uint32_t side = crop * factor;
    const size_t limit = (size_t)256 << 20;
    if (in->rowbytes > limit / side) fail(j, "Decoded row band exceeds 256 MiB. Choose a smaller LQ crop.");
    j->band = malloc(in->rowbytes * side);
    if (!j->band) fail(j, "Could not allocate the PNG row band.");

    // Both roles enumerate the LQ grid. The last valid origin is emitted once.
    uint32_t last_x = j->format[0].width - crop, last_y = j->format[0].height - crop;
    uint32_t read_rows = 0;
    for (uint32_t y = 0;; y = next_origin(y, stride, last_y)) {
        while (read_rows < y * factor + side) {
            if (read_rows % 64 == 0) event(j, 0, y, NULL);
            size_t slot = read_rows % side;
            png_read_row(in->png, j->band + slot * in->rowbytes, NULL);
            ++read_rows;
        }
        for (uint32_t x = 0;; x = next_origin(x, stride, last_x)) {
            write_patch(j, dirfd, x, y, side, factor);
            if (x == last_x) break;
        }
        if (y == last_y) break;
    }
    png_read_end(in->png, in->end);
    metadata(j, in->end, 0, 1);
    free(j->band);
    j->band = NULL;
}

static int run(Job *j, const int fds[2], const int dirs[2], uint32_t scale, uint32_t crop, uint32_t stride) {
    if (setjmp(j->stop)) return 1;
    for (int i = 0; i < 2; ++i) {
        j->role = i;
        open_reader(j, fds[i]);
    }
    const Format *lq = j->format, *gt = j->format + 1;
    if (!scale || (uint64_t)lq->width * scale != gt->width || (uint64_t)lq->height * scale != gt->height)
        fail(j, "Expected GT dimensions to equal the LQ dimensions multiplied by scale.");
    if (!crop || crop > lq->width || crop > lq->height || !stride || stride > crop)
        fail(j, "The crop must fit both LQ axes, with 1 <= stride <= crop.");
    for (int i = 0; i < 2; ++i) {
        j->role = i;
        extract_role(j, dirs[i], crop, stride, i == 0 ? 1 : scale);
    }
    return 0;
}

int studio_paired_patches_v1(int lq, int gt, int lq_dir, int gt_dir, uint32_t scale,
                             uint32_t crop, uint32_t stride, Format format[2],
                             Notify notify, char *error, size_t capacity) {
    Job job = {.format = format, .notify = notify, .error = error, .capacity = capacity};
    const int fds[] = {lq, gt}, dirs[] = {lq_dir, gt_dir};
    // Keep all mutable ownership outside the frame that calls setjmp.
    int status = run(&job, fds, dirs, scale, crop, stride);
    png_destroy_write_struct(&job.writer, &job.header);
    if (job.file) fclose(job.file);
    free(job.band);
    for (unsigned i = 0; i < 2; ++i) {
        Reader *in = job.in + i;
        png_destroy_read_struct(&in->png, &in->info, &in->end);
        if (in->file) fclose(in->file);
    }
    return status;
}
