module JustPIC

using MuladdMacro
using ParallelStencil
using CUDA
using CellArrays

const PS_PACKAGE = ENV["PS_PACKAGE"]

if !ParallelStencil.is_initialized()
    if PS_PACKAGE === "CUDA" 
        @eval @init_parallel_stencil(CUDA, Float64, 2) 
    elseif PS_PACKAGE === "Threads"
        @eval @init_parallel_stencil(Threads, Float64, 2)
    end
end

include("CellArrays/CellArrays.jl")
export @cell

include("Utils.jl")
export init_cell_arrays, cell_array

# INTERPOLATION RELATED FILES

include("Interpolations/utils.jl")

include("Interpolations/gather.jl")
export particle2grid!

include("Interpolations/scatter.jl")
export grid2particle!

include("Interpolations/kernels.jl")
export lerp, bilinear, trilinear

# PARTICLES RELATED FILES

include("Particles/particles.jl")
export Particles

include("Particles/utils.jl")

include("Particles/advection.jl")
export advection_RK!

include("Particles/injection.jl")
export check_injection, inject_particles!, inject_particles_phase!, clean_particles!

include("Particles/shuffle.jl")
export shuffle_particles!

end # module
