# using CairoMakie
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

n  = 101
xq = LinRange(0, 1, n)
n  = 21
yq = zeros(n) 
x  = [0; rand(n)|> sort; 1]
y  = rand(n+2)

function interp1D(xq, x, y)
    @inbounds for j in eachindex(x)[1:end-1]
        x0, x1 = x[j], x[j+1]

        # interpolation
        if x0 < xq < x1
            y0, y1 = y[j], y[j+1]
            return fma(
                (xq-x0),
                (y1-y0)*inv(x1-x0), 
                y0
            )
        end
    end
end

@inline _interp1D(xq, x0, x1, y0, y1) = fma((xq-x0), (y1-y0)*inv(x1-x0), y0)

function interp1D(xq, x, y)
    x_lo, x_hi = x[1], x[end]
    @inbounds for j in eachindex(x)[1:end-1]
        x0, x1 = x[j], x[j+1]

        # interpolation
        if x0 < xq < x1
            y0, y1 = y[j], y[j+1]
            _interp1D(xq, x0, x1, y0, y1)
        end

        # extrapolation
        if xq < x_lo
            x0, x1 = x[1], x[2]
            y0, y1 = y[1], y[2]
            _interp1D(xq, x0, x1, y0, y1)
        end

        if xq > x_hi
            x0, x1 = x[end], x[end-1]
            y0, y1 = y[end], y[end-1]
            _interp1D(xq, x0, x1, y0, y1)
        end
    end
end



@btime interp1D($(xq[2], x, y)...)
@btime interp1D_2($(xq[2], SVector{n+2}(x), SVector{n+2}(y))...)

yq= [interp1D(xq, x, y) for xq in xq[2:end-1]]


f,ax,=scatter(x, y, markersize = 10, color = :red)
lines!(ax,x,y); 
lines!(ax, xq[2:end-1], yq); 
# scatter!(ax, xq[2:end-1], yq, markersize = 10, color = :black)

display(f)

cells = @rand(2, 2, celldims = (5,1), eltype = Float64)
perms = @zeros(2, 2, celldims = (5,1), eltype = Int64)

# 1D MarkerChain 
function sort!(p::MarkerChain)
    (; coords, permutations) = p
    # sort permutations of each cell
    sortperm!(permutations.data, coords[1].data, dims=2)
    ni = size(permutations)
    @parallel (@idx ni) _sort_cells!(coords, permutations)
end

function sort!(coords)
    # sort permutations of each cell
    ni = size(permutations)
    @parallel (1:ni[1], 1:ni[2]) _sort!(coords)
end

@parallel_indices (I...) function _sort!(coords::NTuple{N, T}) where {N, T}
    
    particle_xᵢ = ntuple(Val(N)) do i 
        coords[i][I...]
    end

    permutations = sortperm(first(particle_xᵢ))
    # if cell is already sorted, do nothing
    issorted(permutations) && continue

    # otherwise, sort the cell
    
    for ip in eachindex(permutations)
        permutationᵢ = permutations[ip]
        @assert permutationᵢ ≤ length(permutations)
        ntuple(Val(N)) do i 
            @cell coords[i][perm, I...] = particle_xᵢ[i][ip]
            # @cell coords[i][perm, I...] 
        end
    end
    return 
end
@parallel (1:n, 1:n) _sort!(cells)
@time sort!(cells)

n = 64
xcells = @rand(n, n, celldims = (5,), eltype = Float64)
ycells = @rand(n, n, celldims = (5,), eltype = Float64)
cells = (xcells, ycells)
permutations = @zeros(n, n, celldims = (5,), eltype = Int64)

ni = size(permutations)
sortperm!(permutations.data, xcells.data; dims=2)
@parallel (1:2, 1:2) _sort_cells!(cells, permutations)

cells[1][1,1]