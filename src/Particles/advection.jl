"""
    advection_RK!(particles, V, grid_vx, grid_vy, dt, α)

Advect `particles` with the velocity field `V::NTuple{dims, AbstractArray{T,dims}`
on the staggered grid given by `grid_vx` and `grid_vy`using a Runge-Kutta2 scheme
with `α` and time step `dt`.

    xᵢ ← xᵢ + h*( (1-1/(2α))*f(t,xᵢ) + f(t, y+α*h*f(t,xᵢ))) / (2α)
        α = 0.5 ==> midpoint
        α = 1   ==> Heun
        α = 2/3 ==> Ralston
"""
# Main Runge-Kutta advection function for 2D staggered grids
function advection_RK!(
    particles::Particles, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    dxi = compute_dx(grid_vx)
    (; coords, index) = particles
    px, = coords

    # compute some basic stuff
    ni = size(px)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) _advection_RK!(coords, V, index, grid_vi, local_limits, dxi, dt, α)
    return nothing
end

function advection_RK!(
    particles::Particles,
    V,
    grid_vx::NTuple{3,T},
    grid_vy::NTuple{3,T},
    grid_vz::NTuple{3,T},
    dt,
    α,
) where {T}
    # unpack
    (; coords, index) = particles
    # compute some basic stuff
    dxi = compute_dx(grid_vx)
    ni = size(index)

    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy, grid_vz
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) _advection_RK!(coords, V, index, grid_vi, local_limits, dxi, dt, α)

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (I...) function _advection_RK!(
    p, V::NTuple{N,T}, index, grid, local_limits, dxi, dt, α
) where {N,T}
    for ipart in cellaxes(index)
        doskip(index, ipart, I...) && continue

        # cache particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)

        p_new = advect_particle_RK(pᵢ, V, grid, local_limits, dxi, dt, I, α)

        ntuple(Val(N)) do i
            @inbounds @cell p[i][ipart, I...] = p_new[i]
        end
    end

    return nothing
end

function advect_particle_RK(
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
    idx::NTuple,
    α,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)
    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]

        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed
        # during shuffling
        v = if check_local_limits(local_lims, p0)
            interp_velocity_grid2particle(p0, grid_vi[i], dxi, V[i], idx)
        else
            zero(T)
        end
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        Base.@_inline_meta
        fma(vp0[i], dt * α, p0[i])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        # if this condition is met, it means that the particle
        # went outside the local rank domain. It will be removed
        # during shuffling
        v = if check_local_limits(local_lims, p1)
            interp_velocity_grid2particle(p1, grid_vi[i], dxi, V[i], idx)
        else
            zero(T)
        end
    end

    # final advection step
    pf = ntuple(ValN) do i
        Base.@_propagate_inbounds_meta
        Base.@_inline_meta
        if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
    end

    return pf
end

# Interpolate velocity from staggered grid to particle
@inline function interp_velocity_grid2particle(
    p_i::Union{SVector,NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, idx
)
    # F and coordinates at/of the cell corners
    Fi, xci = corner_field_nodes(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)
    return Fp
end

# Get field F and nodal indices of the cell corners where the particle is located
# @inline function corner_field_nodes(
#     F::AbstractArray{T,N}, p_i, xi_vx, dxi, idx::Union{SVector{N,Integer},NTuple{N,Integer}}
# ) where {N,T}
#     ValN = Val(N)
#     indices = ntuple(ValN) do n
#         Base.@_inline_meta
#         @inbounds begin
#             # unpack
#             idx_n = idx[n]
#             # compute offsets and corrections
#             offset = @inline vertex_offset(xi_vx[n][idx_n], p_i[n], dxi[n])
#         end
#         # cell indices
#         idx_n += offset
#     end

#     # coordinates of lower-left corner of the cell
#     cells = ntuple(ValN) do n
#         Base.@_inline_meta
#         @inbounds xi_vx[n][indices[n]]
#     end

#     # F at the four centers
#     Fi = @inbounds extract_field_corners(F, indices...)

#     return Fi, cells
# end

@generated function corner_field_nodes(
    F::AbstractArray{T,N}, particle, xi_vx, dxi, idx::Union{SVector{N,Integer},NTuple{N,Integer}}
) where {N,T}
    quote
        Base.@_inline_meta
        @inbounds begin
            Base.@nexprs $N i ->  begin
                # unpack
                corrected_idx_i = idx[i]
                # compute offsets and corrections
                corrected_idx_i += @inline vertex_offset(xi_vx[i][corrected_idx_i], particle[i], dxi[1])
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

@inline normalised_distance(x, p, dx) = (p - x) * inv(dx)

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j)
    i1, j1 = i + 1, j + 1
    b = F[i1, j]
    c = F[i, j1]
    d = F[i1, j1]
    a = F[i, j]
    return a, b, c, d # F[i, j], F[i1, j], F[i, j1], F[i1, j1]
end

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j, k)
    i1, j1, k1 = i + 1, j + 1, k + 1
    F000 = F[i, j, k]
    F100 = F[i1, j, k]
    F010 = F[i, j1, k]
    F110 = F[i1, j1, k]
    F001 = F[i, j, k1]
    F101 = F[i1, j, k1]
    F011 = F[i, j1, k1]
    F111 = F[i1, j1, k1]
    return F000, F100, F001, F101, F010, F110, F011, F111
end

@inline firstlast(x::Array) = first(x), last(x)
@inline firstlast(x) = extrema(x)

@inline function inner_limits(grid::NTuple{N,T}) where {N,T}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        ntuple(j -> firstlast.(grid[i])[j], Val(N))
    end
end

@generated function check_local_limits(
    local_lims::NTuple{N,T1}, p::Union{SVector{N,T2},NTuple{N,T2}}
) where {N,T1,T2}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> !(local_lims[i][1] < p[i] < local_lims[i][2]) && return false
        return true
    end
end
