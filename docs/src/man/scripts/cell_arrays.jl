using CellArrays, ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# initialize a cell array with 10x10 cells and 4 particles per cell
A = @zeros(10, 10, celldims = (4,))

# ```julia-repl
# julia> A[1, 1]
# 4-element StaticArraysCore.SVector{4, Float64} with indices SOneTo(4):
#  0.0
#  0.0
#  0.0
#  0.0
# ```

# We cannot modify the values of the cell directly. For this we need help of the `@cell` macro. For example, if we want to set the value of the second particle of the cell (1, 1) to 1.0 we can do it like this:
@cell A[2, 1, 1] = 1.0

# ```julia-repl
# julia> A[1, 1]
# 4-element StaticArraysCore.SVector{4, Float64} with indices SOneTo(4):
#  0.0
#  1.0
#  0.0
#  0.0
# ```

# Finally, if we want read-only to the i-th particles within the cell (1, 1) we can use the `@cell` macro like this:
@cell A[2, 1, 1]



