# Main Runge-Kutta back tracking function for 2D staggered grids

function backtrack_LinP!(
        F,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        grid::NTuple{N, T},
        dt,
    ) where {N, T}

    dxi = compute_dx(first(grid_vi))
    # compute some basic stuff
    ni = size(F)
    ranges = ntuple(Val(N)) do i
        2:ni[i]-1
    end
    # launch parallel backtrack kernel
    @parallel (ranges) backtrack_kernel_LinP!(
        F, method, V, grid_vi, grid, dxi, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function backtrack_kernel_LinP!(
        F::AbstractArray{T, N},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vi,
        grid,
        dxi,
        dt,
    ) where {NF, N, T}

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        # extract particle coordinates from the grid
        @inbounds grid_vi[i][I[i]]
    end
    # advect particle
    local_limits = ntuple(Val(N)) do i
        extrema(grid_vi[i])
    end
    pᵢ_new  = advect_particle(method, pᵢ, V, grid, local_limits, dxi, dt, interp_velocity2particle_LinP, I; backtracking = true)
    F[I...] = _grid2particle(pᵢ_new, grid, dxi, F, I)

    return nothing
end

@parallel_indices (I...) function backtrack_kernel_LinP!(
        F::NTuple{NF, AbstractArray{T, N}},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vi,
        grid,
        dxi,
        dt,
    ) where {NF, N, T}

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        @inline
        # extract particle coordinates from the grid
        @inbounds grid_vi[i][I[i]]
    end
    # advect particle
    local_limits = ntuple(Val(N)) do i
        @inline
        extrema(grid_vi[i])
    end
    # backtrack particle position
    pᵢ_new  = advect_particle(method, pᵢ, V, grid, local_limits, dxi, dt, interp_velocity2particle_LinP, I; backtracking = true)
    ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_new, grid, dxi, F, I)
    end
    
    return nothing
end