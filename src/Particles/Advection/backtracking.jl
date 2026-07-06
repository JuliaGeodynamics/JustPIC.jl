"""
    semilagrangian_advection!(F, F0, method, V, grid_vi, grid, dt)

Advect a grid field with a semi-Lagrangian backtracking step.

Each destination node in `F` is traced backward through the velocity field `V`,
then sampled from `F0` on the vertex grid `grid`. `grid_vi` contains the
staggered coordinates associated with the velocity components.

# Notes
- `F` is overwritten in place.
- `F0` is the source field from the previous step.
- For tuple-valued fields, each component is backtracked independently.
"""
function semilagrangian_advection!(
        F,
        F0,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        grid::NTuple{N, T},
        dt,
    ) where {N, T}
    dxi_velocity = compute_dx.(grid_vi)
    dxi_vertex = compute_dx(grid)
    # compute some basic stuff
    ni = size(F)
    ranges = ntuple(Val(N)) do i
        2:(ni[i] - 1)
    end
    # launch parallel backtrack kernel
    @parallel (ranges) backtrack_kernel!(
        F, F0, method, V, grid_vi, grid, dxi_velocity, dxi_vertex, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function backtrack_kernel!(
        F::AbstractArray,
        F0::AbstractArray,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vi,
        grid,
        dxi_velocity,
        dxi_vertex,
        dt,
    ) where {N, T}

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        # extract particle coordinates from the grid
        @inbounds grid[i][I[i]]
    end
    # backtrack particle
    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, I; backtracking = true)
    I_backtrack = ntuple(Val(N)) do i
        find_parent_cell_bisection(pᵢ_backtrack[i], grid[i], I[i])
    end
    di_vertex = @dxi(dxi_vertex, I_backtrack...)
    F[I...] = _grid2particle(pᵢ_backtrack, grid, di_vertex, F, I_backtrack)
    return nothing
end

@parallel_indices (I...) function backtrack_kernel!(
        F::NTuple{NF, AbstractArray},
        F0::NTuple{NF, AbstractArray},
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        grid_vi,
        grid,
        dxi_velocity,
        dxi_vertex,
        dt,
    ) where {NF, N, T}

    di_vertex = @dxi(di_vertex, I...)
    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        @inline
        @inbounds grid[i][I[i]]
    end
    # backtrack particle position
    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, I; backtracking = true)
    I_backtrack = ntuple(Val(N)) do i
        find_parent_cell_bisection(pᵢ_backtrack[i], grid[i], I[i])
    end
    ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_backtrack, grid, di_vertex, F0[i], I_backtrack)
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
        interp_velocity2particle(particle_coords, grid_vi[i], dxi[i], V[i], idx)
    end
end
