@inline function distance_weight(a::NTuple{N,T}, b::NTuple{N,T}; order=2) where {N,T}
    return inv(distance(a, b)^order)
end

@inline @generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, di::NTuple{N,T}
) where {N,T}
    quote
        val = one(T)
        Base.Cartesian.@nexprs $N i ->
            @inbounds val *= one(T) - abs(a[i] - b[i]) * inv(di[i])
        return val
    end
end

## CPU 2D

@inbounds function _particle2grid!(
    F, Fp, inode, jnode, xi::NTuple{2,T}, p, di
) where {T}
    px, py = p # particle coordinates
    nx, ny = size(F)
    xvertex = xi[1][inode], xi[2][jnode] # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
    for joffset in -1:0
        jvertex = joffset + jnode
        # make sure we stay within the grid
        for ioffset in -1:0
            ivertex = ioffset + inode
            # make sure we stay within the grid
            if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
                # iterate over cell
                for i in cellaxes(px)
                    p_i = @cell(px[i, ivertex, jvertex]), @cell(py[i, ivertex, jvertex])
                    # ignore lines below for unused allocations
                    any(isnan, p_i) && continue
                    ω_i = bilinear_weight(xvertex, p_i, di)
                    ω += ω_i
                    ωxF += ω_i * @cell(Fp[i, ivertex, jvertex])
                end
            end
        end
    end

    return F[inode, jnode] = ωxF / ω
end

function particle2grid!(
    F::Array, Fp::AbstractArray, xi::NTuple{2,T}, particle_coords
) where {T}
    di = grid_size(xi)
    Threads.@threads for jnode in axes(F, 2)
        for inode in axes(F, 1)
            _particle2grid!(F, Fp, inode, jnode, xi, particle_coords, di)
        end
    end
end

## CUDA 2D

function _particle2grid!!(F, Fp, xi, p, di::NTuple{2,T}) where {T}
    inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds if (inode ≤ length(xi[1])) && (jnode ≤ length(xi[2]))
        px, py = p # particle coordinates
        nx, ny = size(F)
        xvertex = (xi[1][inode], xi[2][jnode]) # cell lower-left coordinates
        ω, ωxF = 0.0, 0.0 # init weights

        # iterate over cells around i-th node
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we stay within the grid
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
                    # iterate over cell
                    @inbounds for i in cellaxes(px)
                        p_i = @cell(px[i, ivertex, jvertex]), @cell(py[i, ivertex, jvertex])
                        # ignore lines below for empty memory
                        any(isnan, p_i) && continue
                        ω_i = bilinear_weight(xvertex, p_i, di)
                        ω += ω_i
                        ωxF += ω_i * @cell(Fp[i, ivertex, jvertex])
                    end
                end
            end
        end

        F[inode, jnode] = ωxF / ω
    end
    return nothing
end

function particle2grid!(
    F::CuArray, Fp::AbstractArray, xi::NTuple{2,T}, particle_coords
) where {T}
    di = grid_size(xi)

    nx, ny = size(particle_coords[1])
    nblocksx = ceil(Int, nx / 32)
    nblocksy = ceil(Int, ny / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _particle2grid!!(
            F, Fp, xi, particle_coords, di
        )
    end

    return nothing
end

## CPU 3D

@inbounds function _particle2grid!(
    F, Fp, inode, jnode, knode, xi::NTuple{3,T}, p, di
) where {T}
    px, py, pz = p # particle coordinates
    nx, ny, nz = size(F)
    xvertex = xi[1][inode], xi[2][jnode], xi[3][knode] # cell lower-left coordinates
    ω, ωF = 0.0, 0.0 # init weights

    # iterate over cells around i-th node
    for koffset in -1:0
        kvertex = koffset + knode
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we operate within the grid
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                    @inbounds for ip in cellaxes(px)
                        p_i = (
                            @cell(px[ip, ivertex, jvertex, kvertex]),
                            @cell(py[ip, ivertex, jvertex, kvertex]),
                            @cell(pz[ip, ivertex, jvertex, kvertex]),
                        )
                        any(isnan, p_i) && continue  # ignore lines below for unused allocations
                        ω_i = bilinear_weight(xvertex, p_i, di)
                        ω  += ω_i
                        ωF += ω_i * @cell(Fp[ip, ivertex, jvertex, kvertex])
                    end
                end
            end
        end
    end

    return F[inode, jnode, knode] = ωF * inv(ω)
end

function particle2grid!(
    F::AbstractArray, Fp::AbstractArray, xi::NTuple{3,T}, particle_coords
) where {T}
    di = grid_size(xi)
    Threads.@threads for knode in axes(F, 3)
        for jnode in axes(F, 2), inode in axes(F, 1)
            _particle2grid!(F, Fp, inode, jnode, knode, xi, particle_coords, di)
        end
    end
end

## CUDA 3D

function particle2grid!(
    F::CuArray, Fp::AbstractArray, xi::NTuple{3,T}, particle_coords
) where {T}
    di = grid_size(xi)
    nx, ny, nz = size(Fp)
    threadsx = 8
    threadsy = 8
    threadsz = 4
    nblocksx = ceil(Int, nx / threadsx)
    nblocksy = ceil(Int, ny / threadsy)
    nblocksz = ceil(Int, nz / threadsz)


    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy, threadsz) blocks = (
            nblocksx, nblocksy, nblocksz
        ) _particle2grid!!(F, Fp, xi, particle_coords, di)
    end
    
    return nothing
end

function _particle2grid!!(F, Fp, xi, p, di::NTuple{3,T}) where {T}
    inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    knode = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    lx, ly, lz = length.(xi)

    @inbounds if (inode ≤ lx) && (jnode ≤ ly) && (knode ≤ lz)
        px, py, pz = p # particle coordinates
        nx, ny, nz = size(F)
        xvertex    = (xi[1][inode], xi[2][jnode], xi[3][knode]) # cell lower-left coordinates
        ω, ωF      = 0.0, 0.0 # init weights

        # iterate over cells around i-th node
        for koffset in -1:0
            kvertex = koffset + knode
            for joffset in -1:0
                jvertex = joffset + jnode
                for ioffset in -1:0
                    ivertex = ioffset + inode
                    # make sure we stay within the grid
                    if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                        # iterate over cell
                        @inbounds for i in cellaxes(Fp)
                            p_i = (
                                @cell(px[i, ivertex, jvertex, kvertex]),
                                @cell(py[i, ivertex, jvertex, kvertex]),
                                @cell(pz[i, ivertex, jvertex, kvertex]),
                            )
                            # ignore lines below for unused allocations
                            isnan(p_i[1]) && continue
                            ω_i = bilinear_weight(xvertex, p_i, di)
                            ω += ω_i
                            ωF += ω_i * @cell Fp[i, ivertex, jvertex, kvertex]
                        end
                    end
                end
            end
        end

        F[inode, jnode, knode] = ωF / ω
    end
    return nothing
end
