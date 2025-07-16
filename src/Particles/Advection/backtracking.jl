# Main Runge-Kutta back tracking function for 2D staggered grids


function backtrack!(
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
    @parallel (ranges) backtrack_kernel!(
        F, method, V, grid_vi, grid, dxi, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function backtrack_kernel!(
        F::AbstractArray,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vi,
        grid,
        dxi,
        dt,
    ) where {N, T}

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        # extract particle coordinates from the grid
        @inbounds grid[i][I[i]]
    end
    # backtrack particle
    pᵢ_new  = advect_particle(method, pᵢ, V, grid_vi, dxi, dt, I; backtracking = true)
    F[I...] = _grid2particle(pᵢ_new, grid, dxi, F, I)
    return nothing
end

@parallel_indices (I...) function backtrack_kernel!(
        F::NTuple{NF, AbstractArray},
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
    # backtrack particle position
    pᵢ_new  = advect_particle(method, pᵢ, V, grid_vi, dxi, dt, I; backtracking = true)
    ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_new, grid, dxi, F, I)
    end
    
    return nothing
end

@inline function interp_velocity2particle(
        particle_coords::NTuple{N, Any},
        grid_vi,
        dxi,
        V::NTuple{N, Any},
        idx::NTuple{N, Any},
    ) where {N}
    return ntuple(Val(N)) do i
        @inline
        interp_velocity2particle(particle_coords, grid_vi[i], dxi, V[i], idx)
    end
end
