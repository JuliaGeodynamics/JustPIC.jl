## PARTICLE INJECTION FUNCTIONS

"""
    inject_particles!(particles::Particles, args, grid)

Injects particles if the number of particles in a given cell is such that `n < particles.min_xcell`.

# Arguments
- `particles`: The particles object.
- `args`: `CellArrays`s containing particle fields.
- `grid`: The grid cell vertices.
"""
function inject_particles!(particles::Particles, args, grid::NTuple{N}) where {N}
    # function implementation goes here
    # unpack
    (; coords, index, min_xcell) = particles
    ni = size(index)
    di = compute_dx(grid)
    di_quadrant = di ./ 2
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    # We need a color-coded parallel approach for shared memory devices because
    # we are look for the closest particle, which can be in a neighboring cell
    return if N == 2
        for offsetᵢ in 1:2, offsetⱼ in 1:2
            @parallel (@idx n_color) inject_particles!(
                args, coords, index, grid, di, di_quadrant, min_xcell, (offsetᵢ, offsetⱼ)
            )
        end
    elseif N == 3
        for offsetᵢ in 1:2, offsetⱼ in 1:2, offsetₖ in 1:2
            @parallel (@idx n_color) inject_particles!(
                args,
                coords,
                index,
                grid,
                di,
                di_quadrant,
                min_xcell,
                (offsetᵢ, offsetⱼ, offsetₖ),
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end
end

@parallel_indices (I...) function inject_particles!(
    args, coords, index, grid, di, di_quadrant, min_xcell, offsets::NTuple{N}
) where {N}
    indices = ntuple(Val(N)) do i
        2 * (I[i] - 1) + offsets[i]
    end

    if all(indices .≤ size(index))
        _inject_particles!(args, coords, index, grid, di, di_quadrant, min_xcell, indices)
    end
    return nothing
end

function _inject_particles!(
    args::NTuple{N,T}, coords, index, grid, di, di_quadrant, min_xcell, idx_cell
) where {N,T}

    # coordinates of the lower-left corner of the cell
    xvi = corner_coordinate(grid, idx_cell)
    # coordinates of the lower-left corner of the cell quadrants
    xvi_quadrants = quadrant_corners(xvi, di_quadrant)
    # cache coordinates of all particles inside parent cell
    min_xQuadrant = ceil(Int, min_xcell / length(xvi_quadrants))

    for vertex in xvi_quadrants
        pcell = extract_particle_cell_coordinates(coords, idx_cell...)

        # count current number of particles inside the cell
        particles_num = 0
        for i in cellaxes(index)
            (@index index[i, idx_cell...]) || continue
            # check if particle is in local quadrant
            pcoords = extract_particle_coordinates(pcell, i)
            isincell(pcoords, vertex, di_quadrant) || continue
            # if it's inside, accumulate
            particles_num += 1
        end

        # we are fine, do not inject if
        particles_num ≥ min_xQuadrant && break

        for i in cellaxes(index)
            !(@index index[i, idx_cell...]) || continue

            particles_num += 1
            # add at cellcenter + small random perturbation
            p_new = new_particle(vertex, di_quadrant)
            # fill new particles information
            fill_particle!(coords, p_new, i, idx_cell)
            @index index[i, idx_cell...] = true
            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            for j in 1:N
                new_value = @index args[j][particle_idx, min_idx...]
                @index args[j][i, idx_cell...] = new_value
            end
            # @show particles_num
            # we are done with injection if
            particles_num ≥ min_xQuadrant && break
        end
    end

    return nothing
end

# Injection of particles when multiple phases are present

function inject_particles_phase!(
    particles::Particles, particles_phases, args, fields, grid::NTuple{N}
) where {N}
    # unpack
    (; coords, index, min_xcell) = particles
    ni = size(index)
    di = compute_dx(grid)
    di_quadrant = di ./ 2
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    return if N == 2
        for offsetᵢ in 1:2, offsetⱼ in 1:2
            @parallel (@idx n_color) inject_particles_phase!(
                particles_phases,
                args,
                fields,
                coords,
                index,
                grid,
                di,
                di_quadrant,
                min_xcell,
                (offsetᵢ, offsetⱼ),
            )
        end
    elseif N == 3
        for offsetᵢ in 1:2, offsetⱼ in 1:2, offsetₖ in 1:2
            @parallel (@idx n_color) inject_particles_phase!(
                particles_phases,
                args,
                fields,
                coords,
                index,
                grid,
                di,
                di_quadrant,
                min_xcell,
                (offsetᵢ, offsetⱼ, offsetₖ),
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end
end

@parallel_indices (I...) function inject_particles_phase!(
    particles_phases,
    args,
    fields,
    coords,
    index,
    grid,
    di,
    di_quadrant,
    min_xcell,
    offsets::NTuple{N},
) where {N}
    indices = ntuple(Val(N)) do i
        2 * (I[i] - 1) + offsets[i]
    end

    if all(indices .≤ size(index))
        _inject_particles_phase!(
            particles_phases,
            args,
            fields,
            coords,
            index,
            grid,
            di,
            di_quadrant,
            min_xcell,
            indices,
        )
    end

    return nothing
end

function _inject_particles_phase!(
    particles_phases,
    args,
    fields,
    coords,
    index,
    grid,
    di,
    di_quadrant,
    min_xcell,
    idx_cell,
)
    # coordinates of the lower-left corner of the cell
    xvi = corner_coordinate(grid, idx_cell)
    # coordinates of the lower-left corner of the cell quadrants
    xvi_quadrants = quadrant_corners(xvi, di_quadrant)
    # cache coordinates of all particles inside parent cell
    min_xQuadrant = ceil(Int, min_xcell / length(xvi_quadrants))

    for (ic, vertex) in enumerate(xvi_quadrants)
        pcell = extract_particle_cell_coordinates(coords, idx_cell...)

        # count current number of particles inside the cell
        particles_num = 0
        for i in cellaxes(index)
            (@index index[i, idx_cell...]) || continue
            # check if particle is in local quadrant
            pcoords = extract_particle_coordinates(pcell, i)
            isincell(pcoords, vertex, di_quadrant) || continue
            # if it's inside, accumulate
            particles_num += 1
        end

        # we are fine, do not inject if
        particles_num ≥ min_xQuadrant && continue

        for i in cellaxes(index)
            !(@index index[i, idx_cell...]) || continue

            particles_num += 1
            # add at cellcenter + small random perturbation
            p_new = new_particle(vertex, di_quadrant)
            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            new_phase = @index particles_phases[particle_idx, min_idx...]
            @index particles_phases[i, idx_cell...] = new_phase

            # fill new particle information
            fill_particle!(coords, p_new, i, idx_cell)
            @index index[i, idx_cell...] = true
            # interpolate fields into newly injected particle
            for j in eachindex(args)
                tmp = _grid2particle(p_new, grid, di, fields[j], idx_cell)
                local_field = cell_field(fields[j], idx_cell...)
                lower, upper = extrema(local_field)
                @index args[j][i, idx_cell...] = clamp(tmp, lower, upper)
            end
            # @show i, particles_num
            # we are done with injection if
            particles_num ≥ min_xQuadrant && break
        end
    end
    return nothing
end

## UTILS

# find index of the closest particle w.r.t the new particle
function index_min_distance(coords, pn, index, current_cell, icell, jcell)
    particle_idx_min = i_min = j_min = 0
    dist_min = Inf
    px, py = coords
    nx, ny = size(px)

    for j in (jcell-1):(jcell+1), i in (icell-1):(icell+1), ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1)) && continue # out of the domain
        ((i > nx) || (j > ny)) && continue # out of the domain
        (i == icell) && (j == jcell) && (ip == current_cell) && continue # current injected particle
        (@index index[ip, i, j]) || continue

        # distance from new point to the existing particle
        pxi = @index(px[ip, i, j]), @index(py[ip, i, j])

        any(isnan, pxi) && continue

        d = distance(pxi, pn)

        if d < dist_min
            particle_idx_min = ip
            i_min, j_min = i, j
            dist_min = d
        end
    end

    return particle_idx_min, (i_min, j_min)
end

function index_min_distance(coords, pn, index, current_cell, icell, jcell, kcell)
    particle_idx_min = i_min = j_min = k_min = 0
    dist_min = Inf
    px, py, pz = coords
    nx, ny, nz = size(px)

    for k in (kcell-1):(kcell+1),
        j in (jcell-1):(jcell+1),
        i in (icell-1):(icell+1),
        ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1) || (k < 1)) && continue # out of the domain
        ((i > nx) || (j > ny) || (k > nz)) && continue # out of the domain
        (i == icell) && (j == jcell) && (k == kcell) && (ip == current_cell) && continue # current injected particle
        (@index index[ip, i, j, k]) || continue

        # distance from new point to the existing particle
        pxi = @index(px[ip, i, j, k]), @index(py[ip, i, j, k]), @index(pz[ip, i, j, k])
        d = distance(pxi, pn)

        if d < dist_min
            particle_idx_min = ip
            i_min, j_min, k_min = i, j, k
            dist_min = d
        end
    end

    return particle_idx_min, (i_min, j_min, k_min)
end

@inline function cell_field(field, i, j)
    return field[i, j], field[i+1, j], field[i, j+1], field[i+1, j+1]
end

@inline function cell_field(field, i, j, k)
    return field[i, j, k],
    field[i+1, j, k],
    field[i, j+1, k],
    field[i+1, j+1, k],
    field[i, j, k+1],
    field[i+1, j, k+1],
    field[i, j+1, k+1],
    field[i+1, j+1, k+1]
end

@inline function new_particle(xvi::NTuple{N}, di::NTuple{N}) where {N}
    p_new = ntuple(Val(N)) do i
        # xvi[i] + di[i] * (0.95 * rand() + 0.05)
        xvi[i] + di[i] * rand()
    end
    return p_new
end

@inline function new_particle(xvi::NTuple{2}, di::NTuple{2}, ctr, np)
    th = (2 * pi) / np * (ctr - 1)
    r = min(di...) * 0.25
    p_new = (
        muladd(di[1], 0.5, muladd(r, cos(th), xvi[1])),
        muladd(di[2], 0.5, muladd(r, sin(th), xvi[2])),
    )
    return p_new
end

@inline function new_particle(xvi::NTuple{3}, di::NTuple{3}, ctr, np)
    th = (2 * pi) / np * (ctr - 1)
    r = min(di...) * 0.25
    p_new = (
        muladd(di[1], 0.5, muladd(r, cos(th), xvi[1])),
        muladd(di[2], 0.5, xvi[2]),
        muladd(di[3], 0.5, muladd(r, cos(th), xvi[3])),
    )
    return p_new
end

function quadrant_corners(xvi::NTuple{2}, di_quadrant::NTuple{2})
    c11 = xvi
    c12 = @. xvi + di_quadrant * (1, 0)
    c21 = @. xvi + di_quadrant * (0, 1)
    c22 = @. xvi + di_quadrant * (1, 1)
    return c11, c12, c21, c22
end

function quadrant_corners(xvi::NTuple{3}, di_quadrant::NTuple{3})
    c111 = xvi
    c121 = @. xvi + di_quadrant * (1, 0, 0)
    c211 = @. xvi + di_quadrant * (0, 1, 0)
    c221 = @. xvi + di_quadrant * (1, 1, 0)
    c112 = @. xvi + di_quadrant * (0, 0, 1)
    c122 = @. xvi + di_quadrant * (1, 0, 1)
    c212 = @. xvi + di_quadrant * (0, 1, 1)
    c222 = @. xvi + di_quadrant * (1, 1, 1)

    return c111, c121, c211, c221, c112, c122, c212, c222
end

function extract_particle_cell_coordinates(
    coords::NTuple{N}, I::Vararg{Integer,N}
) where {N}
    return ntuple(Val(N)) do i
        @cell coords[i][I...]
    end
end

function extract_particle_coordinates(coords::NTuple{N}, I::Integer) where {N}
    return ntuple(Val(N)) do i
        coords[i][I]
    end
end

@inline distance2(x, y) = √(mapreduce(x -> (x[1] - x[2])^2, +, zip(x, y)))
