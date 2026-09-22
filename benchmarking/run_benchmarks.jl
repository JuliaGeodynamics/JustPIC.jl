const BACKEND_FLAGS = filter(startswith("--backend="), ARGS)
const BACKEND = isempty(BACKEND_FLAGS) ? "CPU" : split(only(BACKEND_FLAGS), '='; limit = 2)[2]
BACKEND in ("CPU", "CUDA", "AMDGPU", "Metal") ||
    error("Unknown backend $(repr(BACKEND)); use --backend=CPU|CUDA|AMDGPU|Metal")

using JustPIC
using JustPICBenchmarks
@static if BACKEND == "CUDA"
    using CUDA
elseif BACKEND == "AMDGPU"
    using AMDGPU
elseif BACKEND == "Metal"
    using Metal
end

# Separate top-level statement: the vendor bindings exist only after `using` has run.
backend, device = if BACKEND == "CUDA"
    CUDA.CUDABackend, CUDA.name(CUDA.device())
elseif BACKEND == "AMDGPU"
    AMDGPU.ROCBackend, string(AMDGPU.device())
elseif BACKEND == "Metal"
    Metal.MetalBackend, String(Metal.device().name)
else
    JustPIC.CPU, nothing
end

JustPICBenchmarks.main(filter(!startswith("--backend="), ARGS); backend, backend_name = BACKEND, device)
