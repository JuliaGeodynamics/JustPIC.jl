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
function advection_MQS!(
    particles::Particles,
    method::AbstractAdvectionIntegrator,
    V,
    grid_vi::NTuple{N, NTuple{N}},
    dt,
) where {N}

    interpolation_fn = interp_velocity2particle_MQS

    dxi = compute_dx(first(grid_vi))
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_kernel_MQS!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt, interpolation_fn
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function advection_kernel_MQS!(
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

@inline function interp_velocity2particle_MQS(
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
            interp_velocity2particle_MQS(particle_coords, grid_vi[i], dxi, V[i], Val(i), idx)
        else
            Inf
        end
    end
end

@inline function interp_velocity2particle_MQS(
    p_i::Union{SVector,NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, ::Val{N}, idx
) where N
    # F and coordinates of the cell corners
    Fi, xci, indices = corner_field_nodes_LinP(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    t = normalize_coordinates(
        flip_particle(p_i, Val(N)),
        flip_particle(xci, Val(N)),
        flip_particle(dxi, Val(N)),
    )

    # interpolate velocity to pressure nodes
    FMQS = expand_F(F, Val(N), indices...)

    # Interpolate field F from pressure node onto particle
    V = MQS(FMQS, t)
    return V
end

flip_particle(x::NTuple{2}, ::Val{1}) = x
flip_particle(x::NTuple{2}, ::Val{2}) = x[2], x[1]
flip_particle(x::NTuple{3}, ::Val{1}) = x
flip_particle(x::NTuple{3}, ::Val{2}) = x[2], x[1], x[3]
flip_particle(x::NTuple{3}, ::Val{3}) = x[3], x[2], x[1]

# Since the cell-center grid is offset by dxᵢ/2 w.r.t the velocity grid,
# we need to correct the index of the lower-left corner coordinate
# with ±1
#  x------□------x------□------x------□------x
#  |      |      |      |      |      |      |
#  |      |      |      |      |      |      |  x: velocity nodes
#  |      |      |      |      |      |      |  □: pressure nodes
#  |      |      |      |      |      |      |
#  |      |      |      |      |      |      |
#  x------□------x------□------x------□------x
#     (x-1/2,y)       (i,j)       (x+1/2,y)

# # 2D corner correction for x-dim
# @inline function correct_xci_to_pressure_point(xci::NTuple{2}, pxi::NTuple{2}, dxi::NTuple{2}, ::Val{1}) 
#     offset = 1 - 2 * (pxi[1] < xci[1] + dxi[1] * 0.5)
#     return xci[1] + offset * dxi[1] * 0.5, xci[2]
# end
# # 2D corner correction for y-dim
# @inline function correct_xci_to_pressure_point(xci::NTuple{2}, pxi::NTuple{2}, dxi::NTuple{2}, ::Val{2}) 
#     offset = 1 - 2 * (pxi[2] < xci[2] + dxi[2] * 0.5)
#     return xci[1], xci[2] + offset * dxi[2] * 0.5
# end
# # 3D corner correction for x-dim
# @inline function correct_xci_to_pressure_point(xci::NTuple{3}, pxi::NTuple{3}, dxi::NTuple{3}, ::Val{1}) 
#     offset = 1 - 2 * (pxi[1] < xci[1] + dxi[1] * 0.5)
#     return xci[1] + offset * dxi[1] * 0.5, xci[2], xci[3]
# end
# # 3D corner correction for y-dim
# @inline function correct_xci_to_pressure_point(xci::NTuple{3}, pxi::NTuple{3}, dxi::NTuple{3}, ::Val{2}) 
#     offset = 1 - 2 * (pxi[2] < xci[2] + dxi[2] * 0.5)
#     return xci[1], xci[2] + offset * dxi[2] * 0.5, xci[3]
# end
# # 3D corner correction for z-dim
# @inline function correct_xci_to_pressure_point(xci::NTuple{3}, pxi::NTuple{3}, dxi::NTuple{3}, ::Val{3}) 
#     offset = 1 - 2 * (pxi[3] < xci[3] + dxi[3] * 0.5)
#     return xci[1], xci[2], xci[3]+ offset * dxi[3] * 0.5
# end

# @generated function corner_field_nodes_LinP(
#     F::AbstractArray{T,N},
#     particle,
#     xi_vx,
#     dxi,
#     idx::Union{SVector{N,Integer},NTuple{N,Integer}},
# ) where {N,T}
#     quote
#         Base.@_inline_meta
#         @inbounds begin
#             Base.@nexprs $N i -> begin
#                 # unpack
#                 corrected_idx_i = idx[i]
#                 # compute offsets and corrections
#                 corrected_idx_i += @inline vertex_offset(
#                     xi_vx[i][corrected_idx_i], particle[i], dxi[i]
#                 )
#                 cell_i = xi_vx[i][corrected_idx_i]
#             end

#             indices = Base.@ncall $N tuple corrected_idx
#             cells = Base.@ncall $N tuple cell

#             # # F at the four centers
#             Fi = @inbounds extract_field_corners(F, indices...)
           
#             return Fi, cells, indices
#         end
#     end
# end

# Interpolates velocity from velocity-grid to pressure nodes
#      P[i,j+1]     P[i+1,j+1]   
# V[i-1,j+1]   V[i,j+1]    V[i+1,j+1]
#  x------□------x------□------x
#  |      |      |      |      |
#  |      |      |      |      |  x: velocity nodes
#  |      |      |   ⊕  |      |  □: pressure nodes
#  |      |      |      |      |  ⊕: particle
#  |      |      |      |      |
#  x------□------x------□------x
# V[i-1,j]     V[i,j]       V[i+1,j]
#      P[i,j]         P[i+1,j]         
function expand_F(F, ::Val{N}, i, j) where N
    # this is the dimension we are dealing with
    # 1 => x
    # 2 => y
    VN = Val(N)
    nx, ny = size(F)
    
    offsetᵢ, offsetⱼ,  = augment_offset(VN)

    # velocity at velocity corners
    F00 = F[clamp(i + offsetᵢ[1][1], 1, nx), clamp(j + offsetⱼ[1][1], 1, ny)]
    F10 = F[clamp(i + offsetᵢ[1][2], 1, nx), clamp(j + offsetⱼ[1][2], 1, ny)]
    F20 = F[clamp(i + offsetᵢ[1][3], 1, nx), clamp(j + offsetⱼ[1][3], 1, ny)]
    F01 = F[clamp(i + offsetᵢ[2][1], 1, nx), clamp(j + offsetⱼ[2][1], 1, ny)]
    F11 = F[clamp(i + offsetᵢ[2][2], 1, nx), clamp(j + offsetⱼ[2][2], 1, ny)]
    F21 = F[clamp(i + offsetᵢ[2][3], 1, nx), clamp(j + offsetⱼ[2][3], 1, ny)]

    # # swap is needed in the y dimension
    # # to keep things consistent
    # # due to the indexing convection
    # @inline swap_F(F, ::Val{1}) = F
    # @inline swap_F(F, ::Val{2}) = F[1], F[3], F[2], F[4] 
    
    # F_av = swap_F((F00_av, F10_av, F01_av, F11_av), VN)
    
    return F00, F10, F20, F01, F11, F21
end

function expand_F(F, ::Val{N}, i, j, k) where N
    # this is the dimension we are dealing with
    # 1 => x
    # 2 => y
    # 3 => z
    VN = Val(N)
    nx, ny, nz = size(F)

    offsetᵢ, offsetⱼ, offsetₖ = augment_offset(VN)

    # velocity at velocity corners
    F000 = F[clamp(i + offsetᵢ[1][1], 1, nx), clamp(j + offsetⱼ[1][1], 1, ny), clamp(k + offsetₖ[1][1], 1, nz)]
    F100 = F[clamp(i + offsetᵢ[1][2], 1, nx), clamp(j + offsetⱼ[1][2], 1, ny), clamp(k + offsetₖ[1][2], 1, nz)]
    F200 = F[clamp(i + offsetᵢ[1][3], 1, nx), clamp(j + offsetⱼ[1][3], 1, ny), clamp(k + offsetₖ[1][3], 1, nz)]
    F010 = F[clamp(i + offsetᵢ[2][1], 1, nx), clamp(j + offsetⱼ[2][1], 1, ny), clamp(k + offsetₖ[2][1], 1, nz)]
    F110 = F[clamp(i + offsetᵢ[2][2], 1, nx), clamp(j + offsetⱼ[2][2], 1, ny), clamp(k + offsetₖ[2][2], 1, nz)]
    F210 = F[clamp(i + offsetᵢ[2][3], 1, nx), clamp(j + offsetⱼ[2][3], 1, ny), clamp(k + offsetₖ[2][3], 1, nz)]
    F001 = F[clamp(i + offsetᵢ[3][1], 1, nx), clamp(j + offsetⱼ[1][1], 1, ny), clamp(k + offsetₖ[3][1], 1, nz)]
    F101 = F[clamp(i + offsetᵢ[3][2], 1, nx), clamp(j + offsetⱼ[1][2], 1, ny), clamp(k + offsetₖ[3][2], 1, nz)]
    F201 = F[clamp(i + offsetᵢ[3][3], 1, nx), clamp(j + offsetⱼ[1][3], 1, ny), clamp(k + offsetₖ[3][3], 1, nz)]
    F011 = F[clamp(i + offsetᵢ[4][1], 1, nx), clamp(j + offsetⱼ[2][1], 1, ny), clamp(k + offsetₖ[4][1], 1, nz)]
    F111 = F[clamp(i + offsetᵢ[4][2], 1, nx), clamp(j + offsetⱼ[2][2], 1, ny), clamp(k + offsetₖ[4][2], 1, nz)]
    F211 = F[clamp(i + offsetᵢ[4][3], 1, nx), clamp(j + offsetⱼ[2][3], 1, ny), clamp(k + offsetₖ[4][3], 1, nz)]

    # swap is needed in the y dimension
    # to keep things consistent
    # due to the indexing convection
    # @inline swap_F(F, ::Val{1}) = F
    # @inline swap_F(F, ::Val{N}) where N = F[1], F[3], F[2], F[4], F[5], F[7], F[6], F[8]
    # F_av = swap_F((F000_av, F100_av, F010_av, F110_av, F001_av, F101_av, F011_av, F111_av), VN)
    
    return F000, F100, F200, F010, F110, F210, F001, F101, F201, F011, F111, F211
end