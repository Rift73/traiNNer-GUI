# Prebuilt PTX kernels

`effect_cuda.ptx` and `training_cuda.ptx` are built from `effect_cuda.cu` and `training_cuda.cu` (LLVM device-only
CUDA C++, no CUDA headers; `training_cuda.cu` includes `dvd_coefficients.hpp`) and loaded at run time by
`traiNNer/data/ptx_cuda.py`; nothing is compiled when training starts. Rebuild one after editing its source:

```
clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -std=c++17 -O3 -S effect_cuda.cu -o effect_cuda.ptx
clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -std=c++17 -O3 -S training_cuda.cu -o training_cuda.ptx
```

`sm_80` PTX is JIT-compiled by the driver for the GPU in use (Ampere or newer). `traiNNer/data/dvd_response.py` also
reads the DVD coefficients from `training_cuda.cu` and `dvd_coefficients.hpp` for its torch version of the same math.
