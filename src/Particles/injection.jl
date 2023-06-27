## FUNCTIONS TO CHECK WHETER INJECTION IS NEEDED OR NOT

"""
    check_injection(particles)

Returns `true` if there is it at least one cell where injection of new particle(s) is needed.
"""
function check_injection(particles::Particles{N, A}) where {N, A}
    (; inject, index, min_xcell) = particles
    nxi = size(index)

    @parallel @idx(nxi) check_injection!(inject, index, min_xcell)

    return check_injection(particles.inject)
end

@inline check_injection(inject::AbstractArray) = count(inject) > 0

@parallel_indices (i, j) function check_injection!(inject::AbstractMatrix, index, min_xcell)
    if i ≤ size(index, 1) && j ≤ size(index, 2)
        inject[i, j] = isemptycell(index, min_xcell, i, j)
    end
    return nothing
end

@parallel_indices (i, j, k) function check_injection!(inject, index, min_xcell)
    if i ≤ size(index, 1) && j ≤ size(index, 2) && k ≤ size(index, 3)
        inject[i, j, k] = isemptycell(index, min_xcell, i, j, k)
    end
    return nothing
end

## PARTICLE INJECTION FUNCTIONS

"""
    inject_particles!(particles, args, fields, grid)

Injects new particles in the struct `particles` and interpolates the nodal fields `fields` defined
on the staggered grid `grid` onto the new particle field `args`.
"""
function inject_particles!(particles::Particles, args, fields, grid::NTuple{2,T}) where {T}
    # unpack
    (; inject, coords, index, nxcell) = particles
    # linear to cartesian object
    icell, jcell = size(inject)
    di = compute_dx(grid)

    @parallel (1:icell, 1:jcell) inject_particles!(
        inject, args, fields, coords, index, grid, di, nxcell
    )
end

@parallel_indices (icell, jcell) function inject_particles!(
    inject, args, fields, coords, index, grid, di::NTuple{2,T}, nxcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2))
        _inject_particles!(
            inject, args, fields, coords, index, grid, di, nxcell, (icell, jcell)
        )
    end
    return nothing
end

function inject_particles!(particles::Particles, args, fields, grid::NTuple{3,T}) where {T}
    # unpack
    (; inject, coords, index, nxcell) = particles
    # linear to cartesian object
    icell, jcell, kcell = size(inject)
    di = compute_dx(grid)

    @parallel (1:icell, 1:jcell, 1:kcell) inject_particles!(
        inject, args, fields, coords, index, grid, di, nxcell
    )
end

@parallel_indices (icell, jcell, kcell) function inject_particles!(
    inject, args, fields, coords, index, grid, di::NTuple{3,T}, nxcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2)) && (kcell ≤ size(inject, 3))
        _inject_particles!(
            inject, args, fields, coords, index, grid, di, nxcell, (icell, jcell, kcell)
        )
    end
    return nothing
end

function _inject_particles!(
    inject, args, fields, coords, index, grid, di, nxcell, idx_cell
)
    max_xcell = cellnum(index)

    # closures -----------------------------------
    first_cell_index(i) = (i - 1) * max_xcell + 1
    # --------------------------------------------

    @inbounds if inject[idx_cell...]
        # count current number of particles inside the cell
        particles_num = false
        for i in 1:max_xcell
            particles_num += index[i, idx_cell...]
        end

        # coordinates of the lower-left center
        xvi = corner_coordinate(grid, idx_cell)

        for i in 1:max_xcell
            if index[i, idx_cell...] === false
                particles_num += 1

                # add at cellcenter + small random perturbation
                p_new = new_particle(xvi, di)
                fill_particle!(coords, p_new, i, idx_cell)
                index[i, idx_cell...] = true

                for (arg_i, field_i) in zip(args, fields)
                    local_field = cell_field(field_i, idx_cell...)
                    upper = maximum(local_field)
                    lower = minimum(local_field)
                    tmp   = _grid2particle_xvertex(p_new, grid, di, field_i, idx_cell)
                    tmp < lower && (tmp = lower)
                    tmp > upper && (tmp = upper)
                    arg_i[i, idx_cell...] = tmp
                    # arg_i[i, idx_cell...] = clamp(tmp, extrema(field_i)...)
                end
            end

            particles_num == nxcell && break
        end
    end

    return inject[idx_cell...] = false
end

# Injection of particles when multiple phases are present

"""
    inject_particles_phase!(particles, particles_phases, args, fields, grid)

Injects new particles in the struct `particles` and interpolates the nodal fields `fields` defined
on the staggered grid `grid` onto the new particle field `args`. The phase of the particle is given
by `particles_phases`.
"""
function inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid)
    # unpack
    (; inject, coords, index, min_xcell) = particles
    # linear to cartesian object
    ni = size(inject)
    di = compute_dx(grid)

    @parallel (@idx ni) inject_particles_phase!(
        inject, particles_phases, args, fields, coords, index, grid, di, min_xcell
    )
end

@parallel_indices (i, j) function inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di::NTuple{2,T}, min_xcell
) where {T}
    if (i ≤ size(inject, 1)) && (j ≤ size(inject, 2))
        _inject_particles_phase!(
            inject, particles_phases, args, fields, coords, index, grid, di, min_xcell, (i, j)
        )
    end
    return nothing
end

@parallel_indices (i, j, k) function inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di::NTuple{3,T}, min_xcell
) where {T}
    if (i ≤ size(inject, 1)) && (j ≤ size(inject, 2)) && (k ≤ size(inject, 3))
        _inject_particles_phase!(
            inject, particles_phases, args, fields, coords, index, grid, di, min_xcell, (i, j, k)
        )
    end
    return nothing
end

function _inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di, min_xcell, idx_cell
)

    if inject[idx_cell...]
      
        # count current number of particles inside the cell
        particles_num = false
        for i in cellaxes(index)
            particles_num += @cell index[i, idx_cell...]
        end
      
        # coordinates of the lower-left center
        xvi = corner_coordinate(grid, idx_cell)

        for i in cellaxes(index)
            if !(@cell(index[i, idx_cell...]))
                particles_num += 1

                # add at cellcenter + small random perturbation
                p_new = new_particle(xvi, di)

                # add phase to new particle
                particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
                new_phase = @cell particles_phases[particle_idx, min_idx...]
                @cell particles_phases[i, idx_cell...] = new_phase

                fill_particle!(coords, p_new, i, idx_cell)
                @cell index[i, idx_cell...] = true

                # interpolate fields into newly injected particle
                for (arg_i, field_i) in zip(args, fields)
                    tmp         = _grid2particle_xvertex(p_new, grid, di, field_i, idx_cell)
                    local_field = cell_field(field_i, idx_cell...)
                    upper       = maximum(local_field)
                    lower       = minimum(local_field)
                    tmp < lower && (tmp = lower)
                    tmp > upper && (tmp = upper)
                    @cell arg_i[i, idx_cell...] = tmp
                end
            end

            particles_num ≥ min_xcell && break
        end

        inject[idx_cell...] = false
    end

    return nothing
end

@inline distance2(x, y) = mapreduce(x -> (x[1]-x[2])^2, +, zip(x,y)) |> sqrt

# find index of the closest particle w.r.t the new particle
function index_min_distance(coords, pn, index, current_cell, icell, jcell)
   
    particle_idx_min = i_min = j_min = 0
    dist_min = Inf
    px, py = coords
    nx, ny = size(px)

    for j in jcell-1:jcell+1, i in icell-1:icell+1, ip in cellaxes(index)
        
        # early escape conditions
        ((i < 1) || (j < 1)) && continue # out of the domain
        ((i > nx) || (j > ny)) && continue # out of the domain
        (i == icell) && (j == jcell) && (ip == current_cell) && continue # current injected particle
        !(@cell index[ip, i, j]) && continue

        # distance from new point to the existing particle        
        pxi = @cell(px[ip, i, j]), @cell(py[ip, i, j])
        d = distance(pxi, pn)

        if d < dist_min
            particle_idx_min = ip
            i_min, j_min = i, j
            dist_min = d
        end
    end

    particle_idx_min, (i_min, j_min)
end

function index_min_distance(coords, pn, index, current_cell, icell, jcell, kcell)
   
    particle_idx_min = i_min = j_min = k_min = 0
    dist_min = Inf
    px, py, pz = coords
    nx, ny, nz = size(px)

    for k in kcell-1:kcell+1, j in jcell-1:jcell+1, i in icell-1:icell+1, ip in cellaxes(index)
        
        # early escape conditions
        ((i < 1)  || (j < 1)  || (k < 1))  && continue # out of the domain
        ((i > nx) || (j > ny) || (k > nz)) && continue # out of the domain
        (i == icell) && (j == jcell) && (k == kcell) && (ip == current_cell) && continue # current injected particle
        !(@cell index[ip, i, j, k]) && continue

        # distance from new point to the existing particle        
        pxi = @cell(px[ip, i, j, k]), @cell(py[ip, i, j, k]), @cell(pz[ip, i, j, k])
        d   = distance(pxi, pn)

        if d < dist_min
            particle_idx_min = ip
            i_min, j_min, k_min = i, j, k
            dist_min =  d
        end
    end

    particle_idx_min, (i_min, j_min, k_min)
end

@inline cell_field(field, i, j)    = field[i, j], field[i+1, j], field[i, j+1], field[i+1, j+1]
@inline cell_field(field, i, j, k) = field[i, j, k], field[i+1, j, k], field[i, j+1, k], field[i+1, j+1, k], field[i, j, k+1], field[i+1, j, k+1], field[i, j+1, k+1], field[i+1, j+1, k+1]

function new_particle(xvi::NTuple{N,T}, di::NTuple{N,T}) where {N,T}
    p_new = ntuple(Val(N)) do i
        xvi[i] + di[i] * rand(0.05:1e-5: 0.95)
    end
    return p_new
end