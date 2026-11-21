#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Apply a block of normalized, compact FIR rows to one axis of contiguous FP32
// planes. Python owns the tensors and their lifetimes for this synchronous call.
void neosr_filter_rows_f32(
    const float *restrict src, float *restrict dst, const int64_t *starts,
    const float *weights, size_t outer, size_t input_rows, size_t inner,
    size_t output_rows, size_t row_offset, size_t rows, size_t taps, bool first) {
    for (size_t plane = 0; plane < outer; ++plane) {
        for (size_t row = 0; row < rows; ++row) {
            const float *in = src + (plane * input_rows + (size_t)starts[row]) * inner;
            float *out = dst + (plane * output_rows + row_offset + row) * inner;
            const float *kernel = weights + row * taps;
            if (inner == 1) {
                float value = first ? 0.f : *out;
                for (size_t k = 0; k < taps; ++k)
                    value += in[k] * kernel[k];
                *out = value;
            } else {
                // A vertical tap spans contiguous columns for LLVM's vectorizer.
                if (first)
                    for (size_t i = 0; i < inner; ++i)
                        out[i] = 0.f;
                for (size_t k = 0; k < taps; ++k)
                    for (size_t i = 0; i < inner; ++i)
                        out[i] += in[k * inner + i] * kernel[k];
            }
        }
    }
}
