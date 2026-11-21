/* Numeric checkpoint comparison for the three floating-point training formats.
 * Input chunks contain little-endian safetensors payload bytes, with arbitrary
 * alignment. The caller bounds chunks and validates their encoded lengths. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(__clang__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "This target requires LLVM Clang on a little-endian host."
#endif

enum precision { F16, BF16, F32 };

typedef struct {
    uint64_t finite_pairs;
    uint64_t changed;
    uint64_t left_nonfinite;
    uint64_t right_nonfinite;
    double sum_abs;
    double sum_squared;
    double left_squared;
    double right_squared;
    double cross;
    double max_abs;
} measures;

static inline double read_weight(const unsigned char *bytes, enum precision dtype) {
    switch (dtype) {
        case F16: {
            _Float16 x;
            memcpy(&x, bytes, sizeof(x));
            return x;
        }
        case BF16: {
            uint16_t small;
            memcpy(&small, bytes, sizeof(small));
            uint32_t bits = (uint32_t)small << 16;
            float x;
            memcpy(&x, &bits, sizeof(x));
            return x;
        }
        case F32: {
            float x;
            memcpy(&x, bytes, sizeof(x));
            return x;
        }
    }
    return NAN;
}

int measure_values(const unsigned char *left, enum precision left_type,
                   const unsigned char *right, enum precision right_type,
                   size_t count, measures *out) {
    if (!out || !left || !right || (unsigned)left_type > F32 || (unsigned)right_type > F32)
        return -1;
    const size_t left_stride = left_type == F32 ? 4 : 2;
    const size_t right_stride = right_type == F32 ? 4 : 2;
    measures chunk = {0};
    for (size_t i = 0; i < count; ++i) {
        const double a = read_weight(left + i * left_stride, left_type);
        const double b = read_weight(right + i * right_stride, right_type);
        if (!isfinite(a) || !isfinite(b)) {
            chunk.left_nonfinite += !isfinite(a);
            chunk.right_nonfinite += !isfinite(b);
            continue;
        }
        const double delta = b - a;
        const double magnitude = fabs(delta);
        ++chunk.finite_pairs;
        chunk.changed += a != b;
        chunk.sum_abs += magnitude;
        chunk.sum_squared += delta * delta;
        chunk.left_squared += a * a;
        chunk.right_squared += b * b;
        chunk.cross += a * b;
        if (magnitude > chunk.max_abs)
            chunk.max_abs = magnitude;
    }
    *out = chunk;
    return 0;
}

/* A null sum validates a fixed buffer. The first source initializes the sum
   so even a signed zero is treated the same as loading the first tensor. */
int accumulate_values(const unsigned char *input, enum precision dtype,
                      size_t count, double *sum, int first) {
    if (!input || (unsigned)dtype > F32) return -1;
    const size_t stride = dtype == F32 ? 4 : 2;
    for (size_t i = 0; i < count; ++i) {
        const double x = read_weight(input + stride * i, dtype);
        if (!isfinite(x)) return 1;
        if (sum) sum[i] = first ? x : sum[i] + x;
    }
    return 0;
}

int finish_average(unsigned char *output, enum precision dtype, size_t count,
                   const double *sum, unsigned sources) {
    if (!output || !sum || sources < 2 || (unsigned)dtype > F32) return -1;
    const size_t stride = dtype == F32 ? 4 : 2;
    for (size_t i = 0; i < count; ++i) {
        const float x = (float)(sum[i] / (double)sources);
        unsigned char *dst = output + i * stride;
        if (dtype == F32) {
            memcpy(dst, &x, sizeof(x));
        } else if (dtype == F16) {
            const _Float16 half = (_Float16)x;
            memcpy(dst, &half, sizeof(half));
        } else {
            uint32_t bits;
            memcpy(&bits, &x, sizeof(bits));
            /* Round to nearest BF16, ties to even. All inputs are finite. */
            const uint16_t high = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
            memcpy(dst, &high, sizeof(high));
        }
    }
    return 0;
}
