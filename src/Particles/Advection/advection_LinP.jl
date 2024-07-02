# Main Runge-Kutta advection function for 2D staggered grids
"""
    advection!(particles::Particles, method::AbstractAdvectionIntegrator, V, grid_vi::NTuple{N,NTuple{N,T}}, dt)

Advects the particles using the advection scheme defined by `method`.

# Arguments
- `particles`: Particles object to be advected.
- `method`: Time integration method (`Euler` or `RungeKutta2`).
- `V`: Tuple containing `Vx`, `Vy`; and `Vz` in 3D.
- `grid_vi`: Tuple containing the grids corresponding to `Vx`, `Vy`; and `Vz` in 3D.
- `dt`: Time step.
"""
function advection_LinP!(
    particles::Particles,
    method::AbstractAdvectionIntegrator,
    V,
    grid_vi::NTuple{N,NTuple{N,T}},
    dt,
) where {N,T}

    interpolation_fn = interp_velocity2particle_LinP

    dxi = compute_dx(first(grid_vi))
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_kernel_LinP!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt, interpolation_fn
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function advection_kernel_LinP!(
    p,
    method::AbstractAdvectionIntegrator,
    V::NTuple{N},
    index,
    grid,
    local_limits,
    dxi,
    dt,
    interpolation_fn::F,
) where {N,F}

    # iterate over particles in the I-th cell
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, I...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)
        # # advect particle
        pᵢ_new = advect_particle(method, pᵢ, V, grid, local_limits, dxi, dt, interpolation_fn, I)
        # update particle coordinates
        for k in 1:N
            @inbounds @cell p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle_LinP(
    particle_coords::NTuple{N},
    grid_vi,
    local_limits,
    dxi,
    V::NTuple{N},
    idx::NTuple{N},
) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle_LinP(particle_coords, grid_vi[i], dxi, V[i], Val(i), idx)
        else
            Inf
        end
    end
end

@inline function interp_velocity2particle_LinP(
    p_i::Union{SVector,NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, ::Val{N}, idx
) where N
    # F and coordinates at/of the cell corners
    Fi, xci, indices = corner_field_nodes_LinP(F, p_i, xi_vx, dxi, idx)

    # normalize particle coordinates
    tL = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    VL = lerp(Fi, tL)

    # interpolate velocity to pressure nodes
    FP = interpolate_V_to_P(F, xci, p_i, dxi, Val(N), indices...)
    # normalize particle coordinates
    tP = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F from pressure node onto particle
    VP = lerp(FP, tP)
    A = 2/3

    return A * VL + (1 - A) * VP
end

@generated function corner_field_nodes_LinP(
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

            # # F at the four centers
            Fi = @inbounds extract_field_corners(F, indices...)
           
            return Fi, cells, indices
        end
    end
end

function interpolate_V_to_P(F, xi_corner, xi_particle, dxi, ::Val{N}, i, j) where N
    VN = Val(N)
    nx, ny = size(F)
    i += offset_LinP_x(VN, xi_corner, xi_particle, dxi)
    j += offset_LinP_y(VN, xi_corner, xi_particle, dxi)
    
    offsetᵢ, offsetⱼ,  = augment_offset(VN)

    # corners
    # F00 = F[i + offsetᵢ[1][1], j + offsetⱼ[1][1]]
    # F10 = F[i + offsetᵢ[1][2], j + offsetⱼ[1][2]]
    # F20 = F[i + offsetᵢ[1][3], j + offsetⱼ[1][3]]
    # F01 = F[i + offsetᵢ[2][1], j + offsetⱼ[2][1]]
    # F11 = F[i + offsetᵢ[2][2], j + offsetⱼ[2][2]]
    # F21 = F[i + offsetᵢ[2][3], j + offsetⱼ[2][3]]
    # F00_av = (F00 + F10) * 0.5
    # F10_av = (F20 + F10) * 0.5
    # F01_av = (F01 + F11) * 0.5
    # F11_av = (F21 + F11) * 0.5

    # corners condensed way to do it
    vi0 = ntuple(Val(3)) do k
        Base.@_inline_meta
        F[clamp(i + offsetᵢ[1][k], 1, nx), clamp(j + offsetⱼ[1][k], 1, ny)]
    end
    v0i = ntuple(Val(3)) do k
        Base.@_inline_meta
        F[clamp(i + offsetᵢ[2][k], 1, nx), clamp(j + offsetⱼ[2][k], 1, ny)]
    end
    v0 = (vi0[1], vi0[3], v0i[1], v0i[3])
    v1 = (vi0[2], vi0[2], v0i[2], v0i[2])
    av = @. (v0 + v1) * 0.5

    return av
end

function interpolate_V_to_P(F, xi_corner, xi_particle, dxi, ::Val{N}, i, j, k) where N
    nx, ny, nz = size(F)
    i += clamp(offset_LinP_x(VN, xi_corner, xi_particle, dxi), 1, nx)
    j += clamp(offset_LinP_y(VN, xi_corner, xi_particle, dxi), 1, ny)
    k += clamp(offset_LinP_z(VN, xi_corner, xi_particle, dxi), 1, ny)
    

    offsetᵢ, offsetⱼ, offsetₖ = augment_offset(VN)

    #  corners
    F000 = F[i + offsetᵢ[1][1], j + offsetⱼ[1][1], k + offsetₖ[1][1]]
    F100 = F[i + offsetᵢ[1][2], j + offsetⱼ[1][2], k + offsetₖ[1][2]]
    F200 = F[i + offsetᵢ[1][3], j + offsetⱼ[1][3], k + offsetₖ[1][3]]
    F010 = F[i + offsetᵢ[2][1], j + offsetⱼ[2][1], k + offsetₖ[2][1]]
    F110 = F[i + offsetᵢ[2][2], j + offsetⱼ[2][2], k + offsetₖ[2][2]]
    F210 = F[i + offsetᵢ[2][3], j + offsetⱼ[2][3], k + offsetₖ[2][3]]
    F001 = F[i + offsetᵢ[3][1], j + offsetⱼ[1][1], k + offsetₖ[3][1]]
    F101 = F[i + offsetᵢ[3][2], j + offsetⱼ[1][2], k + offsetₖ[3][2]]
    F201 = F[i + offsetᵢ[3][3], j + offsetⱼ[1][3], k + offsetₖ[3][3]]
    F011 = F[i + offsetᵢ[4][1], j + offsetⱼ[2][1], k + offsetₖ[4][1]]
    F111 = F[i + offsetᵢ[4][2], j + offsetⱼ[2][2], k + offsetₖ[4][2]]
    F211 = F[i + offsetᵢ[4][3], j + offsetⱼ[2][3], k + offsetₖ[4][3]]

    F000_av = (F000 + F100) * 0.5
    F100_av = (F200 + F100) * 0.5
    F010_av = (F010 + F110) * 0.5
    F110_av = (F210 + F110) * 0.5
    F001_av = (F001 + F101) * 0.5
    F101_av = (F201 + F101) * 0.5
    F011_av = (F011 + F111) * 0.5
    F111_av = (F211 + F111) * 0.5

    return F000_av, F100_av, F010_av, F110_av, F001_av, F101_av, F011_av, F111_av
end

@inline function offset_LinP(xi_corner, xi_particle, dxi)
    return - 1 *(xi_particle < xi_corner + dxi * 0.5)
end

for (i, fn) in enumerate((:offset_LinP_x, :offset_LinP_y, :offset_LinP_z))
    # Val(i) => ith-direction
    @eval begin
        @inline function ($fn)(::Val{$i}, xi_corner::NTuple{N}, xi_particle::NTuple{N}, dxi::NTuple{N}) where N
            offset_LinP(xi_corner[$i], xi_particle[$i], dxi[$i])
        end
        @inline ($fn)(::Val{I}, xi_corner::NTuple{N}, xi_particle::NTuple{N}, dxi::NTuple{N}) where {N, I} = 0
    end
end

function augment_offset(::Val{1})
    offsetᵢ = (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)
    offsetⱼ = (0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)
    offsetₖ = (0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)
    return offsetᵢ, offsetⱼ, offsetₖ
end

@inline function augment_offset(::Val{2})
    offsetᵢ = (0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)
    offsetⱼ = (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)
    offsetₖ = (0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)

    return offsetᵢ, offsetⱼ, offsetₖ
end

@inline function augment_offset(::Val{3})
    offsetᵢ = (0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)
    offsetⱼ = (0, 0, 0), (0, 0, 0), (1, 1, 1), (1, 1, 1)
    offsetₖ = (0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)

    return offsetᵢ, offsetⱼ, offsetₖ
end