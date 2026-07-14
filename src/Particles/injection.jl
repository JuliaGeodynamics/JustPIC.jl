## PARTICLE INJECTION FUNCTIONS

"""
    inject_particles!(particles::Particles, args)

Inject particles into cells whose occupancy falls below `particles.min_xcell`.

# Arguments
- `particles`: The particles object.
- `args`: tuple of particle fields that should be populated for newly injected particles.

# Notes
- New particles are placed quadrant-by-quadrant inside the cell.
- New field values are copied from the nearest existing particle in the same
  neighborhood.
- The public entry point uses the vertex grid and cell spacing stored in
  `particles`.
"""
inject_particles!(particles::Particles, args) = inject_particles!(particles, args, particles.xvi, particles.di.vertex)

function inject_particles!(particles::Particles, args, grid::NTuple{N}, di) where {N}
    # function implementation goes here
    # unpack
    (; coords, index, min_xcell) = particles
    ni = inner_size(index)
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    # We need a color-coded parallel approach for shared memory devices because
    # we are look for the closest particle, which can be in a neighboring cell
    return if N == 2
        for offsetᵢ in 1:2, offsetⱼ in 1:2
            launch!(
                ka_backend(index), inject_particles_kernel!, n_color,
                args, coords, index, grid, di, min_xcell, ni, (offsetᵢ, offsetⱼ)
            )
        end
    elseif N == 3
        for offsetᵢ in 1:2, offsetⱼ in 1:2, offsetₖ in 1:2
            launch!(
                ka_backend(index), inject_particles_kernel!, n_color,
                args,
                coords,
                index,
                grid,
                di,
                min_xcell,
                ni,
                (offsetᵢ, offsetⱼ, offsetₖ),
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end
end

@kernel function inject_particles_kernel!(
        args, coords, index, grid, di, min_xcell, ni, offsets::NTuple{N}
    ) where {N}
    I = @index(Global, NTuple)
    physical_indices = ntuple(Val(N)) do i
        2 * (I[i] - 1) + offsets[i]
    end

    if all(physical_indices .≤ ni)
        indices = physical_indices .+ 1
        _inject_particles!(args, coords, index, grid, @dxi(di, indices...) ./ 2, min_xcell, indices)
    end
end

function _inject_particles!(
        args::NTuple{N, T}, coords, index, grid, di_quadrant, min_xcell, idx_cell
    ) where {N, T}

    # coordinates of the lower-left corner of the cell
    xvi = corner_coordinate(grid, idx_cell)

    # coordinates of the lower-left corner of the cell quadrants
    xvi_quadrants = quadrant_corners(xvi, di_quadrant)

    # integer ceiling division: `a / b` is a Float64 divide, which Metal cannot do
    min_xQuadrant = cld(min_xcell, length(xvi_quadrants))

    for vertex in xvi_quadrants

        # cache coordinates of all particles inside parent cell
        pcell = extract_particle_cell_coordinates(coords, idx_cell...)

        # count current number of particles inside the cell
        particles_num = 0
        for i in cellaxes(index)
            (CAI.@index index[i, idx_cell...]) || continue

            # check if particle is in local quadrant
            pcoords = extract_particle_coordinates(pcell, i)
            isincell(pcoords, vertex, di_quadrant) || continue

            # if it's inside, accumulate
            particles_num += 1
        end

        # we are fine, do not inject if
        particles_num ≥ min_xQuadrant && break

        for i in cellaxes(index)
            !(CAI.@index index[i, idx_cell...]) || continue

            particles_num += 1

            # add at cellcenter + small random perturbation
            p_new = new_particle(vertex, di_quadrant)

            # # uncomment below for debugging
            # new_cell = find_parent_cell_bisection(p_new, grid, idx_cell)
            # @assert idx_cell == new_cell "Particle in parent cell $idx_cell injected in cell $new_cell"

            # fill new particles information
            fill_particle!(coords, p_new, i, idx_cell)
            CAI.@index index[i, idx_cell...] = true

            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            for j in 1:N
                new_value = CAI.@index args[j][particle_idx, min_idx...]
                CAI.@index args[j][i, idx_cell...] = new_value
            end

            # we are done with injection if
            particles_num ≥ min_xQuadrant && break
        end
    end

    return nothing
end

# Injection of particles when multiple phases are present
"""
    inject_particles_phase!(particles, particles_phases, args, fields, grid)

Inject particles into under-populated cells while also copying phase labels and
field values from nearby particles.

This is the phase-aware variant of `inject_particles!`.

`particles_phases` stores a phase id per particle slot, while `args`/`fields`
hold companion particle properties that must be initialized consistently for the
new particles.
"""
inject_particles_phase!(
    particles::Particles, particles_phases, args, fields
) = inject_particles_phase!(
    particles, particles_phases, args, fields, particles.xvi, particles.xci, particles.di.vertex, particles.di.center
)

function inject_particles_phase!(
        particles::Particles, particles_phases, args, fields, grid::NTuple{N}, grid_center, di, di_center
    ) where {N}
    # unpack
    (; coords, index, min_xcell) = particles
    ni = inner_size(index)
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    return if N == 2
        for offsetᵢ in 1:2, offsetⱼ in 1:2
            launch!(
                ka_backend(index), inject_particles_phase_kernel!, n_color,
                particles_phases,
                args,
                fields,
                coords,
                index,
                grid,
                grid_center,
                di,
                di_center,
                min_xcell,
                ni,
                (offsetᵢ, offsetⱼ),
            )
        end
    elseif N == 3
        for offsetᵢ in 1:2, offsetⱼ in 1:2, offsetₖ in 1:2
            launch!(
                ka_backend(index), inject_particles_phase_kernel!, n_color,
                particles_phases,
                args,
                fields,
                coords,
                index,
                grid,
                grid_center,
                di,
                di_center,
                min_xcell,
                ni,
                (offsetᵢ, offsetⱼ, offsetₖ),
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end
end

@kernel function inject_particles_phase_kernel!(
        particles_phases,
        args,
        fields,
        coords,
        index,
        grid,
        grid_center,
        dxi,
        dxi_center,
        min_xcell,
        ni,
        offsets::NTuple{N},
    ) where {N}
    I = @index(Global, NTuple)
    physical_indices = ntuple(Val(N)) do i
        2 * (I[i] - 1) + offsets[i]
    end

    if all(physical_indices .≤ ni)
        indices = physical_indices .+ 1
        di = @dxi(dxi, indices...)
        di_quadrant = di ./ 2
        _inject_particles_phase!(
            particles_phases,
            args,
            fields,
            coords,
            index,
            grid,
            grid_center,
            di,
            di_quadrant,
            dxi_center,
            min_xcell,
            indices,
        )
    end
end

function _inject_particles_phase!(
        particles_phases,
        args,
        fields,
        coords,
        index,
        grid,
        grid_center,
        di,
        di_quadrant,
        dxi_center,
        min_xcell,
        idx_cell,
    )
    # coordinates of the lower-left corner of the cell
    xvi = corner_coordinate(grid, idx_cell)
    ni_cells = size(index)

    # coordinates of the lower-left corner of the cell quadrants
    xvi_quadrants = quadrant_corners(xvi, di_quadrant)
    # integer ceiling division: `a / b` is a Float64 divide, which Metal cannot do
    min_xQuadrant = cld(min_xcell, length(xvi_quadrants))
    xci = xvi_quadrants[1] .+ di_quadrant # center of the cell

    for (ic, vertex) in enumerate(xvi_quadrants)

        # cache coordinates of all particles inside parent cell
        pcell = extract_particle_cell_coordinates(coords, idx_cell...)

        # count current number of particles inside the cell
        particles_num = 0
        for i in cellaxes(index)
            (CAI.@index index[i, idx_cell...]) || continue

            # check if particle is in local quadrant
            pcoords = extract_particle_coordinates(pcell, i)
            isincell(pcoords, vertex, di_quadrant) || continue

            # if it's inside, accumulate
            particles_num += 1
        end

        # we are fine, do not inject if
        particles_num ≥ min_xQuadrant && continue

        for i in cellaxes(index)
            !(CAI.@index index[i, idx_cell...]) || continue

            particles_num += 1
            # add at cellcenter + small random perturbation
            p_new = new_particle(vertex, di_quadrant)

            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            new_phase = CAI.@index particles_phases[particle_idx, min_idx...]
            CAI.@index particles_phases[i, idx_cell...] = new_phase

            # fill new particle information
            fill_particle!(coords, p_new, i, idx_cell)
            CAI.@index index[i, idx_cell...] = true

            # interpolate fields into newly injected particle
            for j in eachindex(args)
                sz = size(fields[j])
                if sz == ni_cells
                    # if field is defined at cell centers, interpolate from cell center to particle
                    idx_center = shifted_index(p_new, xci, idx_cell)
                    idx_center = clamp.(idx_center, 1, sz .- 1)
                    di_center = @dxi(dxi_center, idx_center...)
                    tmp = _grid2particle(p_new, grid_center, di_center, fields[j], idx_center)
                    local_field = cell_field(fields[j], idx_center...)
                    lower, upper = extrema(local_field)
                    CAI.@index args[j][i, idx_cell...] = clamp(tmp, lower, upper)

                else
                    tmp = _grid2particle(p_new, grid, di, fields[j], idx_cell)
                    local_field = cell_field(fields[j], idx_cell...)
                    lower, upper = extrema(local_field)
                    CAI.@index args[j][i, idx_cell...] = clamp(tmp, lower, upper)
                end
            end

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
    # typed sentinel: a bare `Inf` is Float64 and would widen the running minimum,
    # carrying a Float64 into the kernel (fatal on Metal)
    dist_min = convert(eltype(pn), Inf)
    px, py = coords
    nx, ny = size(px)

    for j in (jcell - 1):(jcell + 1), i in (icell - 1):(icell + 1), ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1)) && continue # out of the domain
        ((i > nx) || (j > ny)) && continue # out of the domain
        (i == icell) && (j == jcell) && (ip == current_cell) && continue # current injected particle
        (CAI.@index index[ip, i, j]) || continue

        # distance from new point to the existing particle
        pxi = CAI.@index(px[ip, i, j]), CAI.@index(py[ip, i, j])

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
    # see the 2D method: typed sentinel keeps the running minimum Float32 on Metal
    dist_min = convert(eltype(pn), Inf)
    px, py, pz = coords
    nx, ny, nz = size(px)

    for k in (kcell - 1):(kcell + 1),
            j in (jcell - 1):(jcell + 1),
            i in (icell - 1):(icell + 1),
            ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1) || (k < 1)) && continue # out of the domain
        ((i > nx) || (j > ny) || (k > nz)) && continue # out of the domain
        (i == icell) && (j == jcell) && (k == kcell) && (ip == current_cell) && continue # current injected particle
        (CAI.@index index[ip, i, j, k]) || continue

        # distance from new point to the existing particle
        pxi = CAI.@index(px[ip, i, j, k]), CAI.@index(py[ip, i, j, k]), CAI.@index(pz[ip, i, j, k])
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
    return field[i, j], field[i + 1, j], field[i, j + 1], field[i + 1, j + 1]
end

@inline function cell_field(field, i, j, k)
    return field[i, j, k],
        field[i + 1, j, k],
        field[i, j + 1, k],
        field[i + 1, j + 1, k],
        field[i, j, k + 1],
        field[i + 1, j, k + 1],
        field[i, j + 1, k + 1],
        field[i + 1, j + 1, k + 1]
end

# keep all arithmetic in the grid eltype: Float64 literals/rand() break Metal
@inline function new_particle(xvi::NTuple{N}, di::NTuple{N}) where {N}
    T = typeof(first(di))
    p_new = ntuple(Val(N)) do i
        xvi[i] + di[i] * muladd(convert(T, 0.95), rand(T), convert(T, 0.05))
    end
    return p_new
end

@inline function new_particle(xvi::NTuple{2}, di::NTuple{2}, ctr, np)
    T = typeof(first(di))
    th = 2 * convert(T, pi) * (ctr - 1) / np
    r = min(di...) / 4
    p_new = (
        muladd(di[1], convert(T, 0.5), muladd(r, cos(th), xvi[1])),
        muladd(di[2], convert(T, 0.5), muladd(r, sin(th), xvi[2])),
    )
    return p_new
end

@inline function new_particle(xvi::NTuple{3}, di::NTuple{3}, ctr, np)
    T = typeof(first(di))
    th = 2 * convert(T, pi) * (ctr - 1) / np
    r = min(di...) / 4
    p_new = (
        muladd(di[1], convert(T, 0.5), muladd(r, cos(th), xvi[1])),
        muladd(di[2], convert(T, 0.5), xvi[2]),
        muladd(di[3], convert(T, 0.5), muladd(r, cos(th), xvi[3])),
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
        coords::NTuple{N}, I::Vararg{Integer, N}
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
