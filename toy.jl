# using CairoMakie
using JustPIC
using JustPIC._2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# n  = 101
# xq = LinRange(0, 1, n)
# n  = 21
# yq = zeros(n) 
# x  = [0; rand(n)|> sort; 1]
# y  = rand(n+2)

# function interp1D(xq, x, y)
#     @inbounds for j in eachindex(x)[1:end-1]
#         x0, x1 = x[j], x[j+1]

#         # interpolation
#         if x0 < xq < x1
#             y0, y1 = y[j], y[j+1]
#             return fma(
#                 (xq-x0),
#                 (y1-y0)*inv(x1-x0), 
#                 y0
#             )
#         end
#     end
# end

# function left_cell_right_particle(coords::NTuple{2, T}, I::Integer) where T 
#     x, y = coords

#     x_last = @cell x[cellnum(xcells), I-1]
#     !isnan(x_last) && !isinf(x_last) && return (x_last, @cell y[cellnum(xcells), I-1])

#     for ip in cellaxes(x)[1:end-1]
#         if isnan(@cell x[ip, I-1]) || isinf(@cell x[ip, I-1])
#             return (@cell x[ip, I-1], @cell y[ip, I-1])
#         end
#     end
#     error("No valid particle found. Cell is probably empty")
# end

# @inline right_cell_left_particle(coords::NTuple{2, T}, I::Integer) where T = (@cell(coords[1][1, I+1]), @cell(coords[2][1, I+1]))
# @inline _interp1D(xq, x0, x1, y0, y1) = fma((xq-x0), (y1-y0)*inv(x1-x0), y0)

# function interp1D_extremas(xq, x, y)
#     x_lo, x_hi = x[1], x[end]
#     @inbounds for j in eachindex(x)[1:end-1]
#         x0, x1 = x[j], x[j+1]

#         # interpolation
#         if x0 < xq < x1
#             y0, y1 = y[j], y[j+1]
#             return _interp1D(xq, x0, x1, y0, y1)
#         end

#         # extrapolation
#         if xq < x_lo
#             x0, x1 = x[1], x[2]
#             y0, y1 = y[1], y[2]
#             return _interp1D(xq, x0, x1, y0, y1)
#         end

#         if xq > x_hi
#             x0, x1 = x[end], x[end-1]
#             y0, y1 = y[end], y[end-1]
#             return _interp1D(xq, x0, x1, y0, y1)
#         end
#     end
#     error("xq out of range")
# end

# @code_warntype interp1D_extremas(xq, x, y)

# function interp1D_inner(xq, cell_coords, I::Integer)
#     x, y = cell_coords[1][I], cell_coords[2][I]
#     x_lo, x_hi = x[1], x[end]
#     @inbounds for j in eachindex(x)[1:end-1]
#         x0, x1 = x[j], x[j+1]

#         # interpolation
#         if x0 < xq < x1
#             y0, y1 = y[j], y[j+1]
#             return _interp1D(xq, x0, x1, y0, y1)
#         end

#         # extrapolation
#         if xq < x_lo
#             x0, y0 = left_cell_right_particle(cell_coords, I)
#             x1, y1 = x[1], y[1]
#             return _interp1D(xq, x0, x1, y0, y1)
#         end

#         if xq > x_hi
#             x0, y0 = x[end], y[end]
#             x1, y1 = right_cell_left_particle(cell_coords, I)
#             return _interp1D(xq, x0, x1, y0, y1)
#         end
#     end
#     error("xq out of range")
# end

# # function resample_all_cells(coords, index)

# #     if i == 1 

# #     return nothing
# # end

# x, y = xcells[2], ycells[2]
# xq = rand()
# interp1D_extremas(xq, x, y)
# interp1D_inner(xq, coords, 2)
# @code_warntype interp1D_inner(xq, coords, 2)

# @btime interp1D($(xq[2], x, y)...)
# @btime interp1D_2($(xq[2], SVector{n+2}(x), SVector{n+2}(y))...)

# yq= [interp1D(xq, x, y) for xq in xq[2:end-1]]


# f,ax,=scatter(x, y, markersize = 10, color = :red)
# lines!(ax,x,y); 
# lines!(ax, xq[2:end-1], yq); 
# # scatter!(ax, xq[2:end-1], yq, markersize = 10, color = :black)

# display(f)

# cells = @rand(2, 2, celldims = (5,1), eltype = Float64)
# perms = @zeros(2, 2, celldims = (5,1), eltype = Int64)

# # 1D MarkerChain 
# function sort!(coords)
#     # sort permutations of each cell
#     ni = length(first(coords))
#     @parallel (1:ni) _sort!(coords)
# end

# @parallel_indices (I...) function _sort!(coords::NTuple{N, T}) where {N, T}
    
#     particle_xᵢ = ntuple(Val(N)) do i 
#         coords[i][I...]
#     end

#     permutations = sortperm(first(particle_xᵢ))

#     # if cell is already sorted, do nothing
#     issorted(permutations) && continue

#     # otherwise, sort the cell
#     for ip in eachindex(permutations)
#         permutationᵢ = permutations[ip]
#         @assert permutationᵢ ≤ length(permutations)
#         ntuple(Val(N)) do i 
#             @cell coords[i][ip, I...] = particle_xᵢ[i][permutationᵢ]
#         end
#     end
#     return 
# end

# n = 32
# xcells = @rand((n,), celldims = (5,), eltype = Float64);
# ycells = @rand((n,), celldims = (5,), eltype = Float64);
# coords = (xcells, ycells);
# sort!(coords)


backend = CPUBackend
L  = 1
n  = 32 # num cells
xv = LinRange(0, L, n+1)
dx = L/n
nxcell = 8
min_xcell, max_xcell = 6, 10
initial_elevation = 0.0

chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation, dx)


function resample!(p::MarkerChain)

    # resampling launch kernel
    @parallel_indices (i) function resample!(
        coords, cell_vertices, index, min_xcell, max_xcell, dx_cells
    )
        resample_cell!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells, i)
        return nothing
    end

    (; coords, index, cell_vertices, min_xcell, max_xcell) = p
    nx = length(cell_vertices) - 1
    dx_cells = cell_length(chain) 
    
    # call kernel
    @parallel (1:nx) resample!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells)
    return nothing
end

function resample_cell!(
    coords::NTuple{2, T}, cell_vertices, index, min_xcell, max_xcell, dx_cells, I
) where T

    # cell particles coordinates
    x_cell, y_cell = coords[1][I], coords[2][I]
    px, py = coords[1], coords[2]

    cell_vertex = cell_vertices[I]
    # number of particles in the cell
    np = count(index[I])
    # dx of the new chain
    dx_chain = dx_cells / (np + 1)
    # resample the cell if the number of particles is  
    # less than min_xcell or it is too distorted
    do_resampling = (np < min_xcell) * isdistorded(x_cell, dx_chain)

    if do_resampling
        # lower-left corner of the cell
        x0 = cell_vertex
        # fill index array
        for ip in 1:min_xcell
            # x query point
            @cell px[ip, I] = xq = x0 + dx_chain * ip
            # interpolated y coordinated
            yq = if 1 < I < length(x_cell) 
                # inner cells; this is true (ncells-2) consecutive times
                interp1D_inner(xq, coords, I)
            else 
                # first and last cells
                interp1D_extremas(xq, x_cell, y_cell)
            end
            @cell py[ip, I] = yq
            @cell index[ip, I] = true
        end
        # fill empty memory locations
        for ip in (min_xcell+1):max_xcell
            @cell px[ip, I] = NaN
            @cell py[ip, I] = NaN
            @cell index[ip, I] = false
        end
    end
    return nothing
end


function isdistorded(x_cell, dx_ideal)
    for ip in eachindex(x_cell)[1:end-1]
        # current particle
        current_x = x_cell[ip]
        # if there is no particle in this memory location,
        # we do nothing
        isnan(current_x) && continue
        # next particle
        next_x = x_cell[ip+1]
        # check wether next memory location holds a particle;
        # if thats the case, find the next particle
        if isnan(next_x)
            next_index = findnext(!isnan, x_cell, ip + 1)
            isnothing(next_index) && break
            next_x = x_cell[next_index]
        end
        # check if the distance between particles is greater than 2*dx_ideal
        # if so, return true so that the cell is resampled
        dx = next_x - current_x
        if dx > 2 * dx_ideal
            return true
        end
    end
    return false
end

x_cell = chain.coords[1][1]


@code_warntype resample_cell!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells, 1)

(; coords, index, cell_vertices, min_xcell, max_xcell) = chain
nx = length(cell_vertices) - 1
dx_cells = cell_length(chain) 

coords[1][1] = coords[1][1] + (@SVector rand(10) ).* dx_cells
coords[2][1] = @SVector rand(10)

sort_chain!(chain)

lines(Array(coords[1][1]), Array(coords[2][1]))
lines!(Array(coords[1][1]), Array(coords[2][1]))

