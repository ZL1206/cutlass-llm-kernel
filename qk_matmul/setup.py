import sys
import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
import subprocess
import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import platform
from packaging.version import parse, Version
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

if CUDA_HOME is None:
    raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available in order to build the package.")

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
print(parent_dir)
print(CUDA_HOME)
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

raw_output, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("12.1"):
    raise RuntimeError("CUDA 12.1 or higher is required to build the package.")

NVCC_FLAGS = []
for capability in ["80", "89", "90"]:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]
NVCC_FLAGS += [
               "-O3", 
               "-std=c++17", 
               "--use_fast_math", 
               "--expt-extended-lambda", 
               "--expt-relaxed-constexpr", 
               "-U__CUDA_NO_HALF_OPERATORS__",
               "-U__CUDA_NO_HALF_CONVERSIONS__",
               "-U__CUDA_NO_HALF2_OPERATORS__",
               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
              ]
nvcc_threads = os.getenv("NVCC_THREADS") or "2"
NVCC_FLAGS += ["--threads", nvcc_threads]



class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

setup(
    name="qk_matmul",
    ext_modules=[
        CUDAExtension(
            name="qk_matmul_ops",
            sources=["qk_matmul.cpp", "qk_matmul_kernel.cu"],
            include_dirs=[
                Path(parent_dir) / "3rd" / "cutlass" / "include",
                Path(parent_dir),
            ],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": NVCC_FLAGS},
        ),
    ],
    cmdclass={"build_ext": NinjaBuildExtension}
)

