"""
    semilagrangian_advection_MQS!(F, F0, method, V, grid_vi, grid, dt)

Semi-Lagrangian advection variant that evaluates backtracked velocities with the
`MQS` interpolation scheme.

Use this when the advecting velocity should be reconstructed with the `MQS`
scheme instead of plain linear interpolation.
"""
function semilagrangian_advection_MQS!(
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
    ni = size(F isa Tuple ? first(F) : F)
    # launch parallel backtrack kernel
    launch!(
        ka_backend(F isa Tuple ? first(F) : F), backtrack_kernel_MQS!, ni .- 2,
        F, F0, method, V, grid_vi, grid, dxi_velocity, dxi_vertex, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@kernel function backtrack_kernel_MQS!(
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
    I0 = @index(Global, NTuple)
    I = I0 .+ 1

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        # extract particle coordinates from the grid
        @inbounds grid[i][I[i]]
    end

    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, interp_velocity2particle_MQS, I; backtracking = true)
    I_backtrack = ntuple(Val(N)) do i
        find_parent_cell_bisection(pᵢ_backtrack[i], grid[i], I[i])
    end
    di_vertex = @dxi(dxi_vertex, I_backtrack...)
    F[I...] = _grid2particle(pᵢ_backtrack, grid, di_vertex, F0, I_backtrack)
end

@kernel function backtrack_kernel_MQS!(
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
    I0 = @index(Global, NTuple)
    I = I0 .+ 1

    # extract particle coordinates
    pᵢ = ntuple(Val(N)) do i
        @inline
        # extract particle coordinates from the grid
        @inbounds grid[i][I[i]]
    end
    # backtrack particle position
    pᵢ_backtrack = advect_particle_SML(method, pᵢ, V, grid_vi, dxi_velocity, dt, interp_velocity2particle_MQS, I; backtracking = true)
    I_backtrack = ntuple(Val(N)) do i
        find_parent_cell_bisection(pᵢ_backtrack[i], grid[i], I[i])
    end
    di_vertex = @dxi(dxi_vertex, I_backtrack...)
    ntuple(Val(NF)) do i
        @inline
        # interpolate field F onto particle
        F[i][I...] = _grid2particle(pᵢ_backtrack, grid, di_vertex, F0[i], I_backtrack)
    end
end

@inline function interp_velocity2particle_MQS(
        particle_coords::NTuple{N}, grid_vi, dxi, V::NTuple{N}, idx::NTuple{N}
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        interp_velocity2particle_MQS(particle_coords, grid_vi[i], dxi[i], V[i], Val(i), idx)
    end
end
