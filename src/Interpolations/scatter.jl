#  Interpolation from grid corners to particle positions

function _grid2particle(
    p_i::NTuple, xvi::NTuple, di::NTuple, F::AbstractArray, idx
)
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)
    
    return Fp
end


# LAUNCHERS

## CPU 2D

function grid2particle!(Fp::AbstractArray, xvi, F::Array{T,2}, particle_coords) where {T}
    di = grid_size(xvi)
    nx, ny = length.(xvi)
    # max_xcell = cellnum(particle_coords[1])
    Threads.@threads for jnode in 1:(ny - 1)
        for inode in 1:(nx - 1)
            _grid2particle!(
                Fp, particle_coords, xvi, di, F, (inode, jnode)
            )
        end
    end
    return nothing
end

## CPU 3D

function grid2particle!(Fp::AbstractArray, xvi, F::Array{T,3}, particle_coords) where {T}
    # cell dimensions
    di = grid_size(xvi)
    nx, ny, nz = length.(xvi)
    # max_xcell = size(particle_coords[1], 1)
    Threads.@threads for knode in 1:(nz - 1)
        for jnode in 1:(ny - 1), inode in 1:(nx - 1)
            _grid2particle!(
                Fp, particle_coords, xvi, di, F, (inode, jnode, knode)
            )
        end
    end
end

## CUDA 2D

function grid2particle!(
    Fp, xvi, F::CuArray{T,2}, particle_coords
) where {T}
    # cell dimensions
    di = grid_size(xvi)
    # max_xcell = cellnum(particle_coords[1]) 
    nx, ny   = size(particle_coords[1])
    nblocksx = ceil(Int, nx / 32)
    nblocksy = ceil(Int, ny / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _grid2particle!(
            Fp, particle_coords, xvi, di, F
        )
    end
end

function grid2particle!(
    Fp, xvi, F::CuArray{T,3}, particle_coords
) where {T}

    # cell dimensions
    di = grid_size(xvi)
    # max_xcell = cellnum(particle_coords[1]) 
    nx, ny, nz = size(Fp)
    threadsx   = 8
    threadsy   = 8
    threadsz   = 4
    nblocksx   = ceil(Int, nx / threadsx)
    nblocksy   = ceil(Int, ny / threadsy)
    nblocksz   = ceil(Int, nz / threadsz)
    nthreads   = threadsx, threadsy, threadsz
    nblocks    = nblocksx, nblocksy, nblocksz
        
    CUDA.@sync begin
        @cuda threads = nthreads blocks = nblocks _grid2particle!(Fp, particle_coords, xvi, di, F)
    end
    return nothing
end

# CPU DIMENSION AGNOSTIC KERNEL

_grid2particle!(Fp, p::Tuple, xvi::Tuple, di::Tuple, F::Array, idx) = inner_grid2particle!(Fp, p, xvi, di, F, idx)
    
# CUDA DIMENSION AGNOSTIC KERNEL

function _grid2particle!(
    Fp, p, xvi, di::NTuple{N, T}, F::CuDeviceArray,
) where {N, T}

    idx = cuda_indices(Val(N))
    if all(idx .≤ size(Fp))
        inner_grid2particle!(Fp, p, xvi, di, F, idx)
    end

    return nothing
end

# INNER INTERPOLATION KERNEL

@inline function inner_grid2particle!(Fp, p, xvi, di::NTuple{N, T}, F, idx) where {N, T}
    # iterate over all the particles within the cells of index `idx` 
    for ip in cellaxes(Fp)
        # cache particle coordinates 
        pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))

        any(isnan, pᵢ) && continue # skip lines below if there is no particle in this pice of memory

        # F at the cell corners
        Fᵢ = field_corners(F, idx)

        # normalize particle coordinates
        tᵢ = normalize_coordinates(pᵢ, xvi, di, idx)

        # Interpolate field F onto particle
        @cell Fp[ip, idx...] = ndlinear(tᵢ, Fᵢ)
    end
end

@inline function cuda_indices(::Val{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return i, j
end

@inline function cuda_indices(::Val{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    return i, j, k
end