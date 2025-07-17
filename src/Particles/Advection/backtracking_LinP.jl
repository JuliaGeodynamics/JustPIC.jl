# Main Runge-Kutta back tracking function for 2D staggered grids

function semilagrangian_advection_LinP!(
        F,
        F0,
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
        F, F0, method, V, grid_vi, grid, dxi, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function backtrack_kernel_LinP!(
        F::AbstractArray,
        F0::AbstractArray,
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
    pᵢ_backtrack  = advect_particle_SML(method, pᵢ, V, grid_vi, dxi, dt, interp_velocity2particle_LinP, I; backtracking = true)
    I_backtrack  = cell_index(pᵢ_backtrack, grid)
    F[I...]      = _grid2particle(pᵢ_backtrack, grid, dxi, F, I_backtrack)

    return nothing
end

@parallel_indices (I...) function backtrack_kernel_LinP!(
        F::NTuple{NF, AbstractArray},
        F0::NTuple{NF, AbstractArray},
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
        @inbounds grid[i][I[i]]
    end
     # backtrack particle position
  pᵢ_backtrack  = advect_particle_SML(method, pᵢ, V, grid_vi, dxi, dt, interp_velocity2particle_LinP, I; backtracking = true)
    I_backtrack  = cell_index(pᵢ_backtrack, grid)
        ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_backtrack, grid, dxi, F0[i], I_backtrack)
    end
    
    return nothing
end

@inline function interp_velocity2particle_LinP(
        particle_coords::NTuple{N}, grid_vi, dxi, V::NTuple{N}, idx::NTuple{N}
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        # @show typeof(particle_coords)
        # @show typeof(grid_vi[i])
        # @show typeof(dxi)
        # @show typeof(V[i])
        # @show typeof(Val(i))
        # @show typeof(idx)
        interp_velocity2particle_LinP(particle_coords, grid_vi[i], dxi, V[i], Val(i), idx)
    end
end