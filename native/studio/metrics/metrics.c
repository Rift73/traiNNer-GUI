/* Evaluate the trainer's valid 11x11 Gaussian SSIM using eleven rows of
 * horizontal moments. Inputs are strided CPU tensors after its own crop,
 * color conversion and quantization. */
#include <stddef.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

enum { WINDOW = 11, MOMENTS = 5 };

static double sample(const void *pixels, int wide, int64_t index, double scale) {
    return (wide ? ((const double *)pixels)[index] : ((const float *)pixels)[index]) * scale;
}

static void horizontal(const double *source, double *result, size_t width,
                       const double *kernel) {
    for (size_t x = 0; x < width; ++x) result[x] = kernel[0] * source[x];
    for (size_t k = 1; k < WINDOW; ++k) {
        const double weight = kernel[k];
        for (size_t x = 0; x < width; ++x) result[x] += weight * source[x + k];
    }
}

static void vertical(const double *history, double *moments, size_t width,
                     unsigned first, const double *kernel) {
    const size_t length = MOMENTS * width;
    for (unsigned k = 0; k < WINDOW; ++k) {
        const double *row = history + ((first + k) % WINDOW) * length;
        const double weight = kernel[k];
        if (k == 0) {
            for (size_t x = 0; x < length; ++x) moments[x] = weight * row[x];
        } else {
            for (size_t x = 0; x < length; ++x) moments[x] += weight * row[x];
        }
    }
}

static double row_score(const double *m, size_t width) {
    const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
    const double c2 = (0.03 * 255.0) * (0.03 * 255.0);
    double sum = 0.0;
    for (size_t x = 0; x < width; ++x) {
        const double a = m[x], b = m[width + x];
        const double aa = a * a, bb = b * b, ab = a * b;
        const double var_a = m[2 * width + x] - aa;
        const double var_b = m[3 * width + x] - bb;
        const double cov = m[4 * width + x] - ab;
        sum += ((2.0 * ab + c1) / (aa + bb + c1)) *
               ((2.0 * cov + c2) / (var_a + var_b + c2));
    }
    return sum;
}

static int ssim_rows(const void *a, const int64_t *sa, const void *b, const int64_t *sb,
                     int wide, const int64_t *shape, const double *kernel, double *output,
                     double scale) {
    if (!a || !b || !sa || !sb || !kernel || !shape || !output ||
        shape[0] <= 0 || shape[1] <= 0 || shape[2] < WINDOW || shape[3] < WINDOW)
        return 1;
    const size_t width = (size_t)shape[3], result_width = width - WINDOW + 1;
    if (width > SIZE_MAX / sizeof(double) / MOMENTS / (WINDOW + 2)) return 1;
    const size_t count = MOMENTS * (width + (WINDOW + 1) * result_width);
    double *workspace = malloc(count * sizeof(double));
    if (!workspace) return 2;
    double *history = workspace + MOMENTS * width;
    double *moments = history + WINDOW * MOMENTS * result_width;
    for (int64_t n = 0; n < shape[0]; ++n) {
        double total = 0.0;
        for (int64_t c = 0; c < shape[1]; ++c) {
            const int64_t a_base = n * sa[0] + c * sa[1];
            const int64_t b_base = n * sb[0] + c * sb[1];
            for (int64_t y = 0; y < shape[2]; ++y) {
                const int64_t ai = a_base + y * sa[2], bi = b_base + y * sb[2];
                for (size_t x = 0; x < width; ++x) {
                    const double left = sample(a, wide, ai + (int64_t)x * sa[3], scale);
                    const double right = sample(b, wide, bi + (int64_t)x * sb[3], scale);
                    workspace[x] = left;
                    workspace[width + x] = right;
                    workspace[2 * width + x] = left * left;
                    workspace[3 * width + x] = right * right;
                    workspace[4 * width + x] = left * right;
                }
                const unsigned slot = (unsigned)(y % WINDOW);
                double *current = history + slot * MOMENTS * result_width;
                for (unsigned m = 0; m < MOMENTS; ++m) {
                    horizontal(workspace + m * width, current + m * result_width,
                               result_width, kernel);
                }
                if (y >= WINDOW - 1) {
                    vertical(history, moments, result_width, (slot + 1) % WINDOW, kernel);
                    total += row_score(moments, result_width);
                }
            }
        }
        output[n] = total / ((double)shape[1] * (double)(shape[2] - WINDOW + 1) * (double)result_width);
    }
    free(workspace);
    return 0;
}

int trainner_ssim_rows(const void *a, const int64_t *sa, const void *b, const int64_t *sb,
                       int wide, const int64_t *shape, const double *kernel, double *output) {
    return ssim_rows(a, sa, b, sb, wide, shape, kernel, output, 255.0);
}

int trainner_ssim_grid_rows(const void *a, const int64_t *sa, const void *b, const int64_t *sb,
                            int wide, const int64_t *shape, const double *kernel, double *output) {
    return ssim_rows(a, sa, b, sb, wide, shape, kernel, output, 1.0);
}

/* Reduce squared errors directly over cropped NCHW views. Independent lanes
 * keep the additions vectorizable without reassociating NaNs under fast-math. */
int trainner_mse_rows(const void *a, const int64_t *sa, const void *b, const int64_t *sb,
                      int wide, const int64_t *shape, double *out) {
    if (!a || !b || !sa || !sb || !shape || !out) return 1;
    for (int i = 0; i < 4; ++i)
        if (shape[i] < 1 || sa[i] < 0 || sb[i] < 0) return 1;
    for (int64_t n = 0; n < shape[0]; ++n) {
        double total = 0.0;
        for (int64_t c = 0; c < shape[1]; ++c) {
            for (int64_t y = 0; y < shape[2]; ++y) {
                const int64_t ai = n * sa[0] + c * sa[1] + y * sa[2];
                const int64_t bi = n * sb[0] + c * sb[1] + y * sb[2];
                double lanes[8] = {0};
                int64_t x = 0;
                for (; x <= shape[3] - 8; x += 8) {
                    for (int j = 0; j < 8; ++j) {
                        double d = sample(a, wide, ai + (x+j)*sa[3], 1.0)
                                 - sample(b, wide, bi + (x+j)*sb[3], 1.0);
                        lanes[j] += d*d;
                    }
                }
                for (int j = 0; j < 8; ++j) total += lanes[j];
                for (; x < shape[3]; ++x) {
                    double d = sample(a, wide, ai + x*sa[3], 1.0) - sample(b, wide, bi + x*sb[3], 1.0);
                    total += d*d;
                }
            }
        }
        out[n] = total / ((double)shape[1] * (double)shape[2] * (double)shape[3]);
    }
    return 0;
}

/* Match torch.round(clamp(x, 0, 1) * 255) including half-way ties. */
static int rgb_code(float value) {
    if (value <= 0.0f) return 0;
    if (value >= 1.0f) return 255;
    float scaled = value * 255.0f;
    int whole = (int)scaled;
    float fraction = scaled - (float)whole;
    return whole + (fraction > 0.5f || (fraction == 0.5f && (whole & 1)));
}

/* Update one contiguous CHW residual history from strided source tensors.
 * A failed call invalidates the history and aborts the enclosing report. */
int trainner_temporal_change(const float *pred, const int64_t *sp,
                            const float *target, const int64_t *st,
                            const int64_t *shape, int16_t *history,
                            int compare, uint64_t *sum) {
    if (!pred || !target || !sp || !st || !shape || !history || !sum) return 1;
    for (int axis = 0; axis < 3; ++axis)
        if (shape[axis] < 1 || sp[axis] < 0 || st[axis] < 0) return 1;
    uint64_t total = 0;
    size_t dst = 0;
    for (int64_t c = 0; c < shape[0]; ++c) {
        for (int64_t y = 0; y < shape[1]; ++y) {
            const float *p = pred + c * sp[0] + y * sp[1];
            const float *t = target + c * st[0] + y * st[1];
            for (int64_t x = 0; x < shape[2]; ++x, ++dst) {
                float a = p[x * sp[2]], b = t[x * st[2]];
                if (!isfinite(a) || !isfinite(b)) return 2;
                int residual = rgb_code(a) - rgb_code(b);
                if (compare) total += (uint64_t)abs(residual - history[dst]);
                history[dst] = (int16_t)residual;
            }
        }
    }
    *sum = total;
    return 0;
}
