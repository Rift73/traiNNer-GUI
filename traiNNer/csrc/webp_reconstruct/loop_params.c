/* The encoder's method-0 loop parameters, reached through the owning libwebp
   1.6.0 translation unit because SetLoopParams is static there. */
#include "src/enc/frame_enc.c"

int webp_reconstruct_loop_params(VP8Encoder *enc) {
    SetLoopParams(enc, enc->config->quality);
    return 1;
}
