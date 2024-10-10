## PARTICLE INJECTION FUNCTIONS

"""
    inject_particles!(particles::Particles, args, grid)

Injects particles if the number of particles in a given cell is such that `n < particles.min_xcell`.

# Arguments
- `particles`: The particles object.
- `args`: `CellArrays`s containing particle fields.
- `grid`: The grid cell vertices.
"""
function inject_particles!(particles::Particles, args, grid::NTuple{N}) where N
    # function implementation goes here
    # unpack
    (; coords, index, min_xcell) = particles
    ni = size(index)
    di = compute_dx(grid)
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    if N == 2
        for offsetᵢ in 1:3, offsetⱼ in 1:3        
            @parallel (@idx n_color) inject_particles!(args, coords, index, grid, di, min_xcell, (offsetᵢ, offsetⱼ))
        end
    elseif N == 3
        for offsetᵢ in 1:3, offsetⱼ in 1:3, offsetₖ in 1:3
            @parallel (@idx n_color) inject_particles!(args, coords, index, grid, di, min_xcell, (offsetᵢ, offsetⱼ, offsetₖ))
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end
end

@parallel_indices (I...) function inject_particles!(
    args, coords, index, grid, di, min_xcell, offsets::NTuple{N}
) where N
    indices = ntuple(Val(N)) do i
        3 * (I[i] - 1) + offsets[i]
    end

    if mapreduce(x -> x[1] ≤ x[2], &, zip(indices, size(index))) &&
        isemptycell(index, min_xcell, indices...)
        _inject_particles!(args, coords, index, grid, di, min_xcell, indices)
    end
    return nothing
end

function _inject_particles!(
    args::NTuple{N,T}, coords, index, grid, di, min_xcell, idx_cell
) where {N,T}
    max_xcell = cellnum(index)

    # count current number of particles inside the cell
    particles_num = false
    for i in 1:max_xcell
        particles_num += @index index[i, idx_cell...]
    end

    # coordinates of the lower-left center
    xvi = corner_coordinate(grid, idx_cell)

    for i in 1:max_xcell
        if @index(index[i, idx_cell...]) === false
            particles_num += 1
            # add at cellcenter + small random perturbation
            p_new = new_particle(xvi, di)
            fill_particle!(coords, p_new, i, idx_cell)
            @index index[i, idx_cell...] = true
            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            for j in 1:N
                new_value = @index args[j][particle_idx, min_idx...]
                @index args[j][i, idx_cell...] = new_value
            end
        end
        particles_num == min_xcell && break
    end

    return nothing
end

# Injection of particles when multiple phases are present

function inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid::NTuple{N}) where N
    # unpack
    (; coords, index, min_xcell) = particles
    # linear to cartesian object
    ni = size(index)
    di = compute_dx(grid)
    n_color = ntuple(i -> ceil(Int, ni[i] * 0.5), Val(N))

    if N == 2
        for offsetᵢ in 1:3, offsetⱼ in 1:3        
            @parallel (@idx ni) inject_particles_phase!(
                particles_phases, args, fields, coords, index, grid, di, min_xcell, (offsetᵢ, offsetⱼ)
            )            
        end
    elseif N == 3
        for offsetᵢ in 1:3, offsetⱼ in 1:3, offsetₖ in 1:3
            @parallel (@idx ni) inject_particles_phase!(
                particles_phases, args, fields, coords, index, grid, di, min_xcell, (offsetᵢ, offsetⱼ, offsetₖ)
            )
        end
    else
        error(ThrowArgument("The dimension of the problem must be either 2 or 3"))
    end

end

@parallel_indices (I...) function inject_particles_phase!(
    particles_phases, args, fields, coords, index, grid, di, min_xcell, offsets::NTuple{N}
) where N

    indices = ntuple(Val(N)) do i
        3 * (I[i] - 1) + offsets[i]
    end

    if mapreduce(x -> x[1] ≤ x[2], &, zip(indices, size(index))) &&
        isemptycell(index, min_xcell, indices...)
        _inject_particles_phase!(
            particles_phases, args, fields, coords, index, grid, di, min_xcell, indices
        )    
    end
   
    return nothing
end

function _inject_particles_phase!(
    particles_phases, args, fields, coords, index, grid, di, min_xcell, idx_cell
)
    np = prod(cellsize(index))

    # count current number of particles inside the cell
    particles_num = false
    for i in cellaxes(index)
        particles_num += @index index[i, idx_cell...]
    end

    # coordinates of the lower-left center
    xvi = corner_coordinate(grid, idx_cell)

    for i in cellaxes(index)
        if !(@index(index[i, idx_cell...]))
            particles_num += 1

            # add at cellcenter + small random perturbation
            p_new = new_particle(xvi, di)
            # p_new = new_particle(xvi, di, particles_num, np)

            # add phase to new particle
            particle_idx, min_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
            new_phase = @index particles_phases[particle_idx, min_idx...]
            @index particles_phases[i, idx_cell...] = new_phase

            fill_particle!(coords, p_new, i, idx_cell)
            @index index[i, idx_cell...] = true

            # interpolate fields into newly injected particle
            for j in eachindex(args)
                tmp = _grid2particle(p_new, grid, di, fields[j], idx_cell)
                local_field = cell_field(fields[j], idx_cell...)
                lower, upper = extrema(local_field)
                tmp < lower && (tmp = lower)
                tmp > upper && (tmp = upper)
                @index args[j][i, idx_cell...] = tmp
            end
        end

        particles_num ≥ min_xcell && break
    end

    return nothing
end

@inline distance2(x, y) = √(mapreduce(x -> (x[1] - x[2])^2, +, zip(x, y)))

# find index of the closest particle w.r.t the new particle
function index_min_distance(coords, pn, index, current_cell, icell, jcell)
    particle_idx_min = i_min = j_min = 0
    dist_min = Inf
    px, py = coords
    nx, ny = size(px)

    for j in (jcell - 1):(jcell + 1), i in (icell - 1):(icell + 1), ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1)) && continue # out of the domain
        ((i > nx) || (j > ny)) && continue # out of the domain
        (i == icell) && (j == jcell) && (ip == current_cell) && continue # current injected particle
        !(@index index[ip, i, j]) && continue

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

    for k in (kcell - 1):(kcell + 1),
        j in (jcell - 1):(jcell + 1),
        i in (icell - 1):(icell + 1),
        ip in cellaxes(index)

        # early escape conditions
        ((i < 1) || (j < 1) || (k < 1)) && continue # out of the domain
        ((i > nx) || (j > ny) || (k > nz)) && continue # out of the domain
        (i == icell) && (j == jcell) && (k == kcell) && (ip == current_cell) && continue # current injected particle
        !(@index index[ip, i, j, k]) && continue

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

@inline function new_particle(xvi::NTuple{N}, di::NTuple{N}) where {N}
    p_new = ntuple(Val(N)) do i
        xvi[i] + di[i] * (0.95 * rand() + 0.05)
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
