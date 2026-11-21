// Device-only LLVM CUDA C++. Kernels run on the caller's current Torch stream.
// Black-box contract: normalized finite RGB, NCHW, positive even source
// dimensions.
#include "dvd_coefficients.hpp"
#define DEVICE __attribute__((device, always_inline)) inline
#define KERNEL extern "C" __attribute__((global))
#define SHARED __attribute__((shared))
DEVICE int tid_x() {
  int x;
  asm("mov.u32 %0, %%tid.x;" : "=r"(x));
  return x;
}
DEVICE int tid_y() {
  int x;
  asm("mov.u32 %0, %%tid.y;" : "=r"(x));
  return x;
}
DEVICE int block_x() {
  int x;
  asm("mov.u32 %0, %%ctaid.x;" : "=r"(x));
  return x;
}
DEVICE int block_y() {
  int x;
  asm("mov.u32 %0, %%ctaid.y;" : "=r"(x));
  return x;
}
DEVICE int block_z() {
  int x;
  asm("mov.u32 %0, %%ctaid.z;" : "=r"(x));
  return x;
}
DEVICE void barrier() { asm volatile("bar.sync 0;" ::: "memory"); }
DEVICE int mirror_edge(int x, int n) {
  if ((unsigned)x < (unsigned)n)
    return x;
  if (n == 1)
    return 0;
  int m = 2 * (n - 1);
  int v = x % m;
  if (v < 0)
    v += m;
  return v < n ? v : m - v;
}
DEVICE float bounded(float a) { return a < 0 ? 0 : a > 1 ? 1 : a; }
static __attribute__((device, constant))
const float catrom[8] = {-3.f / 256,  -9.f / 256, 29.f / 256, 111.f / 256,
                         111.f / 256, 29.f / 256, -9.f / 256, -3.f / 256};
KERNEL void dvd_horizontal(const float *src, float *dst, int w, int h) {
  int x = block_x() * 16 + tid_x(), y = block_y() * 16 + tid_y(),
      plane = block_z(), ow = w / 2;
  if (x >= ow || y >= h)
    return;
  const float *row = src + ((long long)plane * h + y) * w;
  float v = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    v += catrom[j] * row[mirror_edge(2 * x - 3 + j, w)];
  dst[((long long)plane * h + y) * ow + x] = v;
}
KERNEL void dvd_vertical(const float *src, float *out, int w, int h) {
  int x = block_x() * 16 + tid_x(), y = block_y() * 16 + tid_y(), b = block_z(),
      oh = h / 2;
  if (x >= w || y >= oh)
    return;
  float rgb[3] = {};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int iy = mirror_edge(2 * y - 3 + j, h);
#pragma unroll
    for (int c = 0; c < 3; ++c)
      rgb[c] += catrom[j] * src[(((long long)b * 3 + c) * h + iy) * w + x];
  }
  for (int c = 0; c < 3; ++c)
    rgb[c] = bounded(rgb[c]);
  float l = .299f * rgb[0] + .587f * rgb[1] + .114f * rgb[2];
  long long plane = (long long)oh * w, i = (b * 3) * plane + y * w + x;
  out[i] = l;
  out[i + plane] = rgb[0] - l;
  out[i + 2 * plane] = rgb[2] - l;
}
KERNEL void dvd_aligned_fir(const float *planes, float *result, int w, int h,
                            float strength) {
  SHARED float patch[3][32][32];
  int tx = tid_x(), ty = tid_y(), ox = block_x() * 16, oy = block_y() * 16;
  long long area = (long long)w * h, b = (long long)block_z() * 3 * area;
  for (int py = ty; py < 32; py += 16)
    for (int px = tx; px < 32; px += 16) {
      int x = mirror_edge(ox + px - 8, w), y = mirror_edge(oy + py - 8, h);
      for (int c = 0; c < 3; ++c)
        patch[c][py][px] = planes[b + c * area + y * w + x];
    }
  barrier();
  if (ox + tx >= w || oy + ty >= h)
    return;
  float a = 0, dr = 0, db = 0;
  for (int j = 0; j < 17; ++j) {
#pragma unroll
    for (int i = 0; i < 17; ++i) {
      int k = j * 17 + i;
      a += wtp::dvd_coefficients::y_kernel[k] * patch[0][ty + j][tx + i];
      dr += wtp::dvd_coefficients::c_kernel[k] * patch[1][ty + j][tx + i];
      db += wtp::dvd_coefficients::c_kernel[k] * patch[2][ty + j][tx + i];
    }
  }
  float l = bounded(a), dg = -(.299f * dr + .114f * db) / .587f;
  float hi = __builtin_fmaxf(dr, __builtin_fmaxf(dg, db)),
        lo = __builtin_fminf(dr, __builtin_fminf(dg, db));
  float t = 1;
  if (l + hi > 1)
    t = (1 - l) / hi;
  if (l + lo < 0)
    t = __builtin_fminf(t, -l / lo);
  long long idx = b + (oy + ty) * w + ox + tx;
  float base = patch[0][ty + 8][tx + 8], cr = patch[1][ty + 8][tx + 8],
        cb = patch[2][ty + 8][tx + 8];
  float before[] = {base + cr, base - (.299f * cr + .114f * cb) / .587f,
                    base + cb};
  float d[] = {dr, dg, db};
  for (int c = 0; c < 3; ++c)
    result[idx + c * area] =
        bounded(before[c] + strength * (bounded(l + t * d[c]) - before[c]));
}
