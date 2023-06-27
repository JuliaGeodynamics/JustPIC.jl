# Runge-KuttaMETHODS 

## 2D SPECIFIC FUNCTIONS 

# Main Runge-Kuttaadvection function for 2D staggered grids
function advection_RK!(
    particles::Particles, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    # unpack 
    (; coords, index, max_xcell) = particles
    px, = coords
    # compute some basic stuff
    dxi = compute_dx(grid_vx)
   
    nx, ny = size(px)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy
    # launch parallel advection kernel
    @parallel (1:max_xcell, 1:nx, 1:ny) _advection_RK!(
        coords, V, index, grid_vi, dxi, dt, α
    )

    return nothing
end

# ParallelStencil fuction Runge-Kutta advection for 2D staggered grids
@parallel_indices (ipart, icell, jcell) function _advection_RK!(
    p,
    V::NTuple{2, T},
    index::AbstractArray,
    grid,
    dxi,
    dt,
    α,
) where T
    
    px, py = p

    if icell ≤ size(px, 1) && jcell ≤ size(px, 2) && @cell index[ipart, icell, jcell]
        pᵢ = (@cell(px[ipart, icell, jcell]), @cell(py[ipart, icell, jcell]))
        if !any(isnan, pᵢ)
            px_new, py_new = advect_particle_RK(
                pᵢ, V, grid, dxi, dt, (icell, jcell), α
            )
            @cell px[ipart, icell, jcell] = px_new
            @cell py[ipart, icell, jcell] = py_new
        end
    end

    return nothing
end

## 3D SPECIFIC FUNCTIONS 

# Main Runge-Kuttaadvection function for 3D staggered grids

function advection_RK!(
    particles::Particles, V, grid_vx::NTuple{3,T}, grid_vy::NTuple{3,T}, grid_vz::NTuple{3,T}, dt, α
) where {T}
    # unpack 
    (; coords, index) = particles
    # compute some basic stuff
    dxi = compute_dx(grid_vx)   
    nx, ny, nz = size(index)

    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy, grid_vz
    # launch parallel advection kernel
    @parallel (1:nx, 1:ny, 1:nz) _advection_RK!(
        coords, V, index, grid_vi, dxi, dt, α
    )

    return nothing
end

# ParallelStencil fuction Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (icell, jcell, kcell) function _advection_RK!(
    p,
    V::NTuple{3, T},
    index,
    grid,
    dxi,
    dt,
    α,
) where T
    px, py, pz = p
    nx, ny, nz = size(index)
    for ipart in cellaxes(px)
        if icell ≤ nx && jcell ≤ ny && kcell ≤ nz && @cell(index[ipart, icell, jcell, kcell])
            pᵢ = (
                @cell(px[ipart, icell, jcell, kcell]),
                @cell(py[ipart, icell, jcell, kcell]),
                @cell(pz[ipart, icell, jcell, kcell]),
            )

            if !any(isnan, pᵢ)
                px_new, py_new, pz_new = advect_particle_RK(
                    pᵢ, V, grid, dxi, dt, (icell, jcell, kcell), α
                )
                @cell px[ipart, icell, jcell, kcell] = px_new
                @cell py[ipart, icell, jcell, kcell] = py_new
                @cell pz[ipart, icell, jcell, kcell] = pz_new
            end
        end
    end

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

"""
    advect_particle_RK(p0, V, grid_vi, dxi, dt, idx, α)

Runge-Kuttaadvection of a single particle with coordinates `p0::NTuple{dims, T}` and velocity `V::::NTuple{dims, AbstractArray{T,dims}` 
on a staggered grid `grid_vi` with cell sizes `dxi` and cell indices `idx` over a time step `dt` with RK2 parameter `α`.

    xᵢ ← xᵢ + h*( (1-1/(2α))*f(t,xᵢ) + f(t, y+α*h*f(t,xᵢ))) / (2α)
        α = 0.5 ==> midpoint
        α = 1   ==> Heun
        α = 2/3 ==> Ralston
"""
function advect_particle_RK(
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    dxi,
    dt,
    idx::NTuple,
    α,
) where {T,N}
    _α   = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        Base.@_inline_meta
        interp_velocity_grid2particle(p0, grid_vi[i], dxi, V[i], idx)
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        Base.@_inline_meta
        muladd(vp0[i], dt * α, p0[i])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        Base.@_inline_meta
        interp_velocity_grid2particle(p1, grid_vi[i], dxi, V[i], idx)
    end

    # final advection step
    pf = ntuple(ValN) do i
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
function interp_velocity_grid2particle(
    p_i::NTuple, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, idx
)
    # F and coordinates at/of the cell corners
    Fi, xci = @inbounds corner_field_nodes(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)
    return Fp
end

# Get field F and nodal indices of the cell corners where the particle is located
Base.@propagate_inbounds function corner_field_nodes(
    F::AbstractArray{T, N}, p_i, xi_vx, dxi, idx::NTuple{N, Integer}
) where {N, T}

    ValN = Val(N)
    indices = ntuple(ValN) do n
        # unpack
        idx_n = idx[n]
        # compute offsets and corrections
        offset = vertex_offset(xi_vx[n][idx_n], p_i[n], dxi[n])
        # cell indices
        idx_n += offset
    end

    # coordinates of lower-left corner of the cell
    cells = ntuple(ValN) do n
        xi_vx[n][indices[n]]
    end
 
    # F at the four centers
    Fi = extract_field_corners(F, indices...)

    return Fi, cells
end

@inline function vertex_offset(xi, pxi, di)
    dist = normalised_distance(xi, pxi, di)
    (     dist >  2) *  2 +
    ( 2 > dist >  1) *  1 +
    (-1 < dist <  0) * -1 +
    (     dist < -1) * -2 
end

@inline normalised_distance(x, p, dx) = (p - x) * inv(dx)

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j)
    i1, j1 = i + 1, j + 1
    return F[i, j], F[i1, j], F[i, j1], F[i1, j1]
end

@inline Base.@propagate_inbounds function extract_field_corners(F, i, j, k)
    i1, j1, k1 = i + 1, j + 1, k + 1
    return (
        F[i , j , k ], # v000
        F[i1, j , k ], # v100
        F[i , j , k1], # v001
        F[i1, j , k1], # v101
        F[i , j1, k ], # v010
        F[i1, j1, k ], # v110
        F[i , j1, k1], # v011
        F[i1, j1, k1], # v111
    )
end