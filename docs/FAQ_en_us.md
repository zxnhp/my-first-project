# Frequently Asked Questions

### 1. When using the command `pip install magic-pdf[full]` on newer versions of macOS, the error `zsh: no matches found: magic-pdf[full]` occurs.

On macOS, the default shell has switched from Bash to Z shell, which has special handling logic for certain types of string matching. This can lead to the "no matches found" error. You can try disabling the globbing feature in the command line and then run the installation command again.

```bash
setopt no_nomatch
pip install magic-pdf[full]
```

### 2. Encountering the error `pickle.UnpicklingError: invalid load key, 'v'.` during use

This might be due to an incomplete download of the model file. You can try re-downloading the model file and then try again.
Reference: https://github.com/opendatalab/MinerU/issues/143

### 3. Where should the model files be downloaded and how should the `/models-dir` configuration be set?

The path for the model files is configured in "magic-pdf.json". just like:

```json
{
  "models-dir": "/tmp/models"
}
```

This path is an absolute path, not a relative path. You can obtain the absolute path in the models directory using the "pwd" command.
Reference: https://github.com/opendatalab/MinerU/issues/155#issuecomment-2230216874

### 4. Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2

The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:

```bash
sudo apt-get install libgl1-mesa-glx
```

Reference: https://github.com/opendatalab/MinerU/issues/388

### 5. Encountered error `ModuleNotFoundError: No module named 'fairscale'`

You need to uninstall the module and reinstall it:

```bash
pip uninstall fairscale
pip install fairscale
```

Reference: https://github.com/opendatalab/MinerU/issues/411

### 6. On some newer devices like the H100, the text parsed during OCR using CUDA acceleration is garbled.

The compatibility of cuda11 with new graphics cards is poor, and the CUDA version used by Paddle needs to be upgraded.

```bash
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

Reference: https://github.com/opendatalab/MinerU/issues/558

### 7. On some Linux servers, the program immediately reports an error `Illegal instruction (core dumped)`

This might be because the server's CPU does not support the AVX/AVX2 instruction set, or the CPU itself supports it but has been disabled by the system administrator. You can try contacting the system administrator to remove the restriction or change to a different server.

References: https://github.com/opendatalab/MinerU/issues/591 , https://github.com/opendatalab/MinerU/issues/736


### 8. Error when installing MinerU on CentOS 7 or Ubuntu 18: `ERROR: Failed building wheel for simsimd`

The new version of albumentations (1.4.21) introduces a dependency on simsimd. Since the pre-built package of simsimd for Linux requires a glibc version greater than or equal to 2.28, this causes installation issues on some Linux distributions released before 2019. You can resolve this issue by using the following command:
```
pip install -U magic-pdf[full,old_linux] --extra-index-url https://wheels.myhloli.com
```

Reference: https://github.com/opendatalab/MinerU/issues/1004

### 9. Old Graphics Cards Such as M40 Encounter "RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED"

An error occurs during operation (cuda):
```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasGemmStridedBatchedEx(handle, opa, opb, (int)m, (int)n, (int)k, (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea, b, CUDA_R_16BF, (int)ldb, strideb, (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec, (int)num_batches, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
```
Because BF16 precision is not supported on graphics cards before the Turing architecture and some graphics cards are not recognized by torch, it is necessary to manually disable BF16 precision.
Modify the code in lines 287-290 of the "pdf_parse_union_core_v2.py" file (note that the location may vary in different versions):
```
if torch.cuda.is_bf16_supported():
    supports_bfloat16 = True
else:
    supports_bfloat16 = False
```
Change it to:
```
supports_bfloat16 = False
```
Reference: https://github.com/opendatalab/MinerU/issues/1508