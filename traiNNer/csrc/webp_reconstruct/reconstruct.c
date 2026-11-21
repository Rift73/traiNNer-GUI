/* Lossy WebP method-0 pixels without a bitstream, built against libwebp 1.6.0.

   webp_reconstruct() runs the encoder's segment analysis, macroblock decisions,
   prediction, quantisation and reconstruction, then the decoder's loop filter,
   and writes the YUV420 planes a decoder would produce from the encoded file.
   The entropy coding and decoding are skipped because they do not change the
   pixels. The owning translation unit is included for its static encoder
   constructor and destructor. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/enc/webp_enc.c"

int webp_reconstruct_loop_params(VP8Encoder *enc);

static void copy_rows(uint8_t *dst, int dst_stride, const uint8_t *src,
                      int src_stride, int width, int height) {
    for (int y = 0; y < height; ++y) {
        memcpy(dst + y * dst_stride, src + y * src_stride, (size_t)width);
    }
}

/* The decoder's complex loop filter over whole macroblocks, in raster order.
   Padded macroblocks are filtered too, as the decoder does before cropping.
   Returns 0 for filter headers this function does not reproduce. */
static int filter_picture(const VP8Encoder *enc, uint8_t *buf, int stride) {
    if (enc->filter_hdr.level == 0) return 1;
    if (enc->filter_hdr.simple || enc->filter_hdr.sharpness ||
        enc->filter_hdr.i4x4_lf_delta) {
        return 0;
    }
    VP8DspInit();
    const int size = stride * 16 * enc->mb_h;
    const int uv_stride = stride / 2;
    for (int my = 0; my < enc->mb_h; ++my) {
        for (int mx = 0; mx < enc->mb_w; ++mx) {
            const VP8MBInfo *mb = &enc->mb_info[my * enc->mb_w + mx];
            const int level = enc->segment_hdr.num_segments > 1
                                  ? enc->dqm[mb->segment].fstrength
                                  : enc->filter_hdr.level;
            if (!level) continue;
            const int limit = 3 * level;
            const int hev = (level >= 40) ? 2 : (level >= 15);
            const int inner = !mb->type || !mb->skip;
            uint8_t *y = buf + my * 16 * stride + mx * 16;
            uint8_t *u = buf + size + my * 8 * uv_stride + mx * 8;
            uint8_t *v = u + size / 4;
            if (mx) {
                VP8HFilter16(y, stride, limit + 4, level, hev);
                VP8HFilter8(u, v, uv_stride, limit + 4, level, hev);
            }
            if (inner) {
                VP8HFilter16i(y, stride, limit, level, hev);
                VP8HFilter8i(u, v, uv_stride, limit, level, hev);
            }
            if (my) {
                VP8VFilter16(y, stride, limit + 4, level, hev);
                VP8VFilter8(u, v, uv_stride, limit + 4, level, hev);
            }
            if (inner) {
                VP8VFilter16i(y, stride, limit, level, hev);
                VP8VFilter8i(u, v, uv_stride, limit, level, hev);
            }
        }
    }
    return 1;
}

/* src and out are packed planar YUV420: Y (w*h), then U and V
   (ceil(w/2)*ceil(h/2) each). Returns 1 on success, 0 when the input is out
   of range or the filter header is not supported. */
__attribute__((visibility("default")))
int webp_reconstruct(const uint8_t *src, uint8_t *out, int w, int h, int quality) {
    if (!src || !out || w < 1 || h < 1 || w > 2048 || h > 2048 ||
        quality < 40 || quality > 100) {
        return 0;
    }
    WebPConfig config;
    WebPPicture picture;
    if (!WebPConfigInit(&config) || !WebPPictureInit(&picture)) return 0;
    config.method = 0;
    config.quality = (float)quality;
    const int cw = (w + 1) / 2, ch = (h + 1) / 2;
    picture.width = w;
    picture.height = h;
    picture.y = (uint8_t *)src;
    picture.u = picture.y + w * h;
    picture.v = picture.u + cw * ch;
    picture.y_stride = w;
    picture.uv_stride = cw;
    VP8Encoder *enc = InitVP8Encoder(&config, &picture);
    if (!enc) return 0;
    const int stride = enc->mb_w * 16, size = stride * enc->mb_h * 16;
    uint8_t *surface = WebPSafeMalloc(size + size / 2, 1);
    int ok = surface && VP8EncAnalyze(enc) && webp_reconstruct_loop_params(enc);
    if (ok) {
        VP8EncIterator it;
        VP8IteratorInit(enc, &it);
        VP8InitFilter(&it);
        do {
            VP8ModeScore rd;
            VP8IteratorImport(&it, NULL);
            VP8Decimate(&it, &rd, enc->rd_opt_level);
            uint8_t *yp = surface + it.y * 16 * stride + it.x * 16;
            uint8_t *up = surface + size + it.y * 8 * (stride / 2) + it.x * 8;
            copy_rows(yp, stride, it.yuv_out + Y_OFF_ENC, BPS, 16, 16);
            copy_rows(up, stride / 2, it.yuv_out + U_OFF_ENC, BPS, 8, 8);
            copy_rows(up + size / 4, stride / 2, it.yuv_out + V_OFF_ENC, BPS, 8, 8);
            VP8StoreFilterStats(&it);
            VP8IteratorSaveBoundary(&it);
        } while (VP8IteratorNext(&it));
        VP8AdjustFilterStrength(&it);
        ok = filter_picture(enc, surface, stride);
        if (ok) {
            copy_rows(out, w, surface, stride, w, h);
            copy_rows(out + w * h, cw, surface + size, stride / 2, cw, ch);
            copy_rows(out + w * h + cw * ch, cw, surface + size + size / 4,
                      stride / 2, cw, ch);
        }
    }
    WebPSafeFree(surface);
    return DeleteVP8Encoder(enc) && ok;
}
