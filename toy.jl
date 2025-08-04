import JustPIC._2D as JP

function foo(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N}},
        dt,
    ) where {N}
    interpolation_fn = JP.interp_velocity2particle_MQS

    dxi = JP.compute_dx(first(grid_vi))
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = JP.inner_limits(grid_vi)

    # launch parallel advection kernel
    advection_kernel_MQS!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt, interpolation_fn
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS
function advection_kernel_MQS!(
        p,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N},
        index,
        grid,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
    ) where {N, F}

    for j in axes(index, 2), i in axes(index, 1)
        I = i,j
        # iterate over particles in the I-th cell
        for ipart in JP.cellaxes(index)
            # skip if particle does not exist in this memory location
            JP.doskip(index, ipart, I...) && continue
            # extract particle coordinates
            pᵢ = JP.get_particle_coords(p, ipart, I...)
            # # advect particle
            pᵢ_new = advect_particle(
                method, pᵢ, V, grid, local_limits, dxi, dt, interpolation_fn, I
            )
            # update particle coordinates
            # for k in 1:N
            #     @inbounds @index p[k][ipart, I...] = pᵢ_new[k]
            # end
        end
    end
    return nothing
end

@inline function advect_particle(
        method::RungeKutta2,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
        idx::NTuple,
    ) where {N, F}

    # interpolate velocity to current location
    vp0 = interpolation_fn(p0, grid_vi, local_limits, dxi, V, idx)

    # first advection stage x = x + v * dt * α
    p1 = JP.first_stage(method, dt, vp0, p0)

    # interpolate velocity to new location
    vp1 = interpolation_fn(p1, grid_vi, local_limits, dxi, V, idx)

    # final advection step
    p2 = JP.second_stage(method, dt, vp0, vp1, p0)

    return p2
end

@b foo($(particles3, integrator, V, grid_vxi, dt)...)
ProfileCanvas.@profview for i in 1:10000 foo(particles3, integrator, V, grid_vxi, dt) end

@inline function interp_velocity2particle_MQS(
        particle_coords::NTuple{N}, grid_vi, local_limits, dxi, V::NTuple{N}, idx::NTuple{N}
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle_MQS(particle_coords, grid_vi[i], dxi, V[i], Val(i), idx)
        else
            Inf
        end
    end
end

@inline function interp_velocity2particle_MQS(
        p_i::Union{SVector, NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, ::Val{N}, idx
    ) where {N}
    # F and coordinates of the cell corners
    Fi, xci, indices = corner_field_nodes_LinP(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    t = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F from pressure node onto particle
    V = if all(x -> 1 < x[1] < x[2] - 1, zip(indices, size(F)))
        MQS(F, Fi, t, indices..., Val(N))
    else
        lerp(Fi, t)
    end
    return V
end


A = rand(20,20);
arrays = A,A,A;
I = 1,2;
getindex.(arrays, I...)