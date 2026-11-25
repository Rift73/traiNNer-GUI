# Third-party sources

- `third_party/libjxl`: JPEG XL Project Authors, BSD-3-Clause; see its LICENSE.
  The reference library is compiled unchanged. Mathematical translations in
  `src/butteraugli_ane` follow this pinned implementation and retain this attribution.
- `third_party/libjxl/third_party/highway`: Google Highway, Apache-2.0/BSD-3-Clause
  as described in its LICENSE.
- `third_party/gjxl` at `bb7b714e858ffe7668a84c8588745a99b7015f39`, Yunho Cho, Apache-2.0:
  the original investigation used its scalar and Metal implementations to identify
  stage boundaries and readable versions of JPEG XL's mathematical expressions.
  Any copied/transcribed expressions are identified in their source files.

The public Core ML API is provided by Apple. coremltools is BSD-3-Clause.
ANEForge is a separate MIT-licensed project using private Apple interfaces; that
route is not a supported Apple API. Its source remains in an optional ignored
checkout rather than being copied into this package.

The PyTorch training implementation adapts the tensor mathematics from [butteraugli-ane](https://github.com/yunho-c/butteraugli-ane/tree/f65c5e15e8d3888b11590bc9b8f37dd953fb71ab). Its independent numerical target is libjxl e8ff09762481785938d8e4e01333ed3917571161. This package includes the batched PyTorch port and a CUDA stencil adjoint. The JPEG XL license is included in LICENSE.libjxl.
