using JustPIC, CellArrays, ParallelStencil, Test, LinearAlgebra

if get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "AMDGPU"
    using AMDGPU
    AMDGPU.functional() && set_backend("AMDGPU_Float64_3D")
elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CUDA"
    using CUDA
    CUDA.functional() && set_backend("CUDA_Float64_3D")
elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CPU"
    # run on CPU
    set_backend("Threads_Float64_3D")
end
