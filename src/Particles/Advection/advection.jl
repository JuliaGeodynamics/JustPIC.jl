# Main Runge-Kutta advection function for 2D staggered grids
function advection!(
    particles::Particles,
    method::AbstractAdvectionIntegrator,
    V,
    grid_vi::NTuple{N,NTuple{N,T}},
    dt,
) where {N,T}
    dxi = compute_dx(first(grid_vi))
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_kernel!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function advection_kernel!(
    p,
    method::AbstractAdvectionIntegrator,
    V::NTuple{N,T},
    index,
    grid,
    local_limits,
    dxi,
    dt,
) where {N,T}

    # iterate over particles in the I-th cell
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, I...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)
        # # advect particle
        pᵢ_new = advect_particle(method, pᵢ, V, grid, local_limits, dxi, dt, I)
        # update particle coordinates
        for k in 1:N
            @inbounds @cell p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle(
    particle_coords::NTuple{N,Any},
    grid_vi,
    local_limits,
    dxi,
    V::NTuple{N,Any},
    idx::NTuple{N,Any},
) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle(particle_coords, grid_vi[i], dxi, V[i], idx)
        else
            Inf
        end
    end
end

# Interpolate velocity from staggered grid to particle. Innermost kernel
@inline function interp_velocity2particle(
    p_i::Union{SVector,NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, idx
)
    # F and coordinates at/of the cell corners
    Fi, xci = corner_field_nodes(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    return Fp
end

## Other functions

@generated function corner_field_nodes(
    F::AbstractArray{T,N},
    particle,
    xi_vx,
    dxi,
    idx::Union{SVector{N,Integer},NTuple{N,Integer}},
) where {N,T}
    quote
        Base.@_inline_meta
        @inbounds begin
            Base.@nexprs $N i -> begin
                # unpack
                corrected_idx_i = idx[i]
                # compute offsets and corrections
                corrected_idx_i += @inline vertex_offset(
                    xi_vx[i][corrected_idx_i], particle[i], dxi[1]
                )
                cell_i = xi_vx[i][corrected_idx_i]
            end

            indices = Base.@ncall $N tuple corrected_idx
            cells = Base.@ncall $N tuple cell

            # F at the four centers
            Fi = @inbounds extract_field_corners(F, indices...)
        end

        return Fi, cells
    end
end

@inline function vertex_offset(xi, pxi, di)
    dist = normalised_distance(xi, pxi, di)
    return (dist > 2) * 2 + (2 > dist > 1) * 1 + (-1 < dist < 0) * -1 + (dist < -1) * -2
end
