ENV["PS_PACKAGE"] = "Threads"

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using JustPIC
using CellArrays
using ImplicitGlobalGrid
import MPI

# @parallel_indices (i, j) function copy_field!(A::CellArray, B::AbstractArray{T, 2}, ip) where T
#     @cell A[ip, i, j] = B[i, j]
#     return nothing
# end

# @parallel_indices (i, j, k) function copy_field!(A::CellArray, B::AbstractArray{T, 3}, ip) where T
#     @cell A[ip, i, j, k] = B[i, j, k]
#     return nothing
# end

lx=4; nx=n; ny=n;
me, = init_global_grid(nx, ny, 0; init_MPI=false);

x = @fill(me, nx, nx, celldims=(2,)) 
x = @fill(false, nx, nx, celldims=(2,), eltype=Bool) 

# function update_cell_halo!(x)
#     ni = size(x)
#     tmp = @zeros(ni...)

#     for ip in cellaxes(x)
#         tmp .= field(x, ip)
#         update_halo!(tmp)
#         @parallel (@range ni) copy_field!(x, tmp, ip)
#     end

# end

update_cell_halo!(x)

me==0 && display(field(x,1))
# println("\n")


finalize_global_grid()