"""
    semilagrangian_advection!F, F0, integrator, V, grid_vi, grid, dt)

Performs semi-Lagrangian advection by backtracking particle positions in a velocity field.
This function updates the positions and/or properties of particles according to the semi-Lagrangian scheme.

# Arguments

- `F`: The new state of the grid field (e.g., density, temperature).
- `F0`: The current state of the grid field (used for interpolation).
- `integrator`: The numerical integrator to use for advection (e.g., Euler, Rk2, RK4).
- `V`: The velocity field at the particle positions.
- `grid_vi`: The grid cell indices for the velocity field.
- `grid`: The spatial grid information.
- `dt`: The time step for the advection.
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
    dxi_vertex   = compute_dx(grid)
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

    di_vertex   = @dxi(dxi_vertex, I...)
    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        # extract particle coordinates from the grid
        @inbounds grid[i][I[i]]
    end
    # backtrack particle
    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, I; backtracking = true)
    # I_backtrack = cell_index(pᵢ_backtrack, grid)
    I_backtrack = ntuple(Val(N)) do i
        find_parent_cell_bisection(pᵢ_backtrack[i], grid[i]; seed = I[i])
    end
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

    dxi_vertex   = @dxi(di_vertex, I...)
    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        @inline
        @inbounds grid[i][I[i]]
    end
    # backtrack particle position
    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, I; backtracking = true)
    I_backtrack = cell_index(pᵢ_backtrack, grid)
    ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_backtrack, grid, dxi_vertex, F0[i], I_backtrack)
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
