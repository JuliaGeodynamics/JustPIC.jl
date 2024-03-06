## FUNCTIONS TO CHECK WHETHER INJECTION IS NEEDED OR NOT

"""
    check_injection(particles)

Returns `true` if there is it at least one cell where injection of new particle(s) is needed.
"""
function check_injection(particles::Particles{B,N,A}) where {B,N,A}
    (; inject, index, min_xcell) = particles
    nxi = size(index)

    @parallel @range(nxi) check_injection!(inject, index, min_xcell)

    return check_injection(particles.inject)
end

@inline check_injection(inject::Array) = count(inject) > 0
@inline check_injection(inject::AbstractArray) = check_injection(Array(inject))

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
function inject_particles!(particles::Particles, args, fields, grid)
    # unpack
    (; inject, coords, index, min_xcell) = particles
    ni = size(inject)
    di = compute_dx(grid)

    @parallel (@idx ni) inject_particles!(
        inject, args, fields, coords, index, grid, di, min_xcell
    )
end

@parallel_indices (icell, jcell) function inject_particles!(
    inject, args, fields, coords, index, grid, di::NTuple{2,T}, min_xcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2))
        _inject_particles!(
            inject, args, fields, coords, index, grid, di, min_xcell, (icell, jcell)
        )
    end
    return nothing
end

@parallel_indices (icell, jcell, kcell) function inject_particles!(
    inject, args, fields, coords, index, grid, di::NTuple{3,T}, min_xcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2)) && (kcell ≤ size(inject, 3))
        _inject_particles!(
            inject, args, fields, coords, index, grid, di, min_xcell, (icell, jcell, kcell)
        )
    end
    return nothing
end

function _inject_particles!(
    inject, args, fields, coords, index, grid, di::NTuple{N,T}, min_xcell, idx_cell
) where {N,T}
    max_xcell = cellnum(index)

    @inbounds if inject[idx_cell...]

        # count current number of particles inside the cell
        particles_num = false
        for i in 1:max_xcell
            particles_num += @cell index[i, idx_cell...]
        end

        # coordinates of the lower-left center
        xvi = corner_coordinate(grid, idx_cell)

        for i in 1:max_xcell
            if @cell(index[i, idx_cell...]) === false
                particles_num += 1

                # add at cellcenter + small random perturbation
                p_new = new_particle(xvi, di)
                # p_new = new_particle(xvi, di, particles_num, max_xcell)

                fill_particle!(coords, p_new, i, idx_cell)
                @cell index[i, idx_cell...] = true

                # add phase to new particle
                # particle_idx, min_idx = index_min_distance(
                #     coords, p_new, index, i, idx_cell...
                # )

                # for j in eachindex(args)
                #     @cell args[j][i, idx_cell...] = @cell args[j][particle_idx, min_idx...]
                # end

                # for j in eachindex(args)
                #     local_field = cell_field(fields[j], idx_cell...)
                #     upper = maximum(local_field)
                #     lower = minimum(local_field)
                #     tmp = _grid2particle(p_new, grid, di, fields[j], idx_cell)
                #     tmp < lower && (tmp = lower)
                #     tmp > upper && (tmp = upper)
                #     @cell args[j][i, idx_cell...] = tmp
                # end

                for j in eachindex(args)
                    ω, ωxF = 0e0, 0e0
                    # iterate over cell
                    for ip in cellaxes(index)
                        # early exit if particle is not in the cell
                        i == ip && continue # this is the index of the new particle
                        doskip(index, ip, idx_cell...) && continue
                        p_i = ntuple(Val(N)) do n
                            @cell(coords[n][ip, idx_cell...])
                        end
                        ω_i = distance_weight(p_i, p_new; order=2)
                        ω += ω_i
                        ωxF = fma(ω_i, @cell(args[j][ip, idx_cell...]), ωxF)
                    end
                    @cell args[j][i, idx_cell...] = ωxF / ω
                end

            end

            particles_num == min_xcell && break
        end
    end

    return inject[idx_cell...] = false
end

# function _inject_particles!(
#     inject, args, fields, coords, index, grid, di::NTuple{N,T}, min_xcell, idx_cell
# ) where {N,T}
#     max_xcell = cellnum(index)
#     # xvertex = ntuple(Val(N)) do i
#     #     grid[i][idx_cell[i]]
#     # end
#     @inbounds if inject[idx_cell...]

#         # count current number of particles inside the cell
#         particles_num = false
#         for i in 1:max_xcell
#             particles_num += @cell index[i, idx_cell...]
#         end

#         # coordinates of the lower-left center
#         xvi = corner_coordinate(grid, idx_cell)

#         for i in 1:max_xcell
#             if @cell(index[i, idx_cell...]) === false
#                 particles_num += 1

#                 # add at cellcenter + small random perturbation
#                 p_new = new_particle(xvi, di)
#                 # p_new = new_particle(xvi, di, particles_num, max_xcell)

#                 fill_particle!(coords, p_new, i, idx_cell)
#                 @cell index[i, idx_cell...] = true
#                 # iterate over cell
#                 for j in eachindex(args)
#                     ω, ωxF = 0e0, 0e0
#                     for ip in cellaxes(index)
#                         # early exit if particle is not in the cell
#                         i == ip && continue # this is the index of the new particle
#                         doskip(index, ip, idx_cell...) && continue
#                         p_i = ntuple(Val(N)) do n
#                             @cell(coords[n][ip, idx_cell...])
#                         end
#                         ω_i = distance_weight(p_i, p_new; order=3)
#                         ω += ω_i
#                         ωxF = fma(ω_i, @cell(args[j][ip, idx_cell...]), ωxF)
#                     end
#                     @cell args[j][i, idx_cell...] = ωxF / ω
#                 end
#             end

#             particles_num == min_xcell && break
#         end
#     end

#     return inject[idx_cell...] = false
# end

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

    @parallel (@range ni) inject_particles_phase!(
        inject, particles_phases, args, fields, coords, index, grid, di, min_xcell
    )
end

@parallel_indices (i, j) function inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di::NTuple{2,T}, min_xcell
) where {T}
    if (i ≤ size(inject, 1)) && (j ≤ size(inject, 2))
        _inject_particles_phase!(
            inject,
            particles_phases,
            args,
            fields,
            coords,
            index,
            grid,
            di,
            min_xcell,
            (i, j),
        )
    end
    return nothing
end

@parallel_indices (i, j, k) function inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di::NTuple{3,T}, min_xcell
) where {T}
    if (i ≤ size(inject, 1)) && (j ≤ size(inject, 2)) && (k ≤ size(inject, 3))
        _inject_particles_phase!(
            inject,
            particles_phases,
            args,
            fields,
            coords,
            index,
            grid,
            di,
            min_xcell,
            (i, j, k),
        )
    end
    return nothing
end

function _inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, di, min_xcell, idx_cell
)
    if inject[idx_cell...]
        np = prod(cellsize(index))

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
                # p_new = new_particle(xvi, di)
                p_new = new_particle(xvi, di, particles_num, np)

                # add phase to new particle
                particle_idx, min_idx = index_min_distance(
                    coords, p_new, index, i, idx_cell...
                )
                new_phase = @cell particles_phases[particle_idx, min_idx...]
                @cell particles_phases[i, idx_cell...] = new_phase

                fill_particle!(coords, p_new, i, idx_cell)
                @cell index[i, idx_cell...] = true

                # interpolate fields into newly injected particle
                for j in eachindex(args)
                    tmp = _grid2particle(p_new, grid, di, fields[j], idx_cell)
                    local_field = cell_field(fields[j], idx_cell...)
                    upper = maximum(local_field)
                    lower = minimum(local_field)
                    tmp < lower && (tmp = lower)
                    tmp > upper && (tmp = upper)
                    @cell args[j][i, idx_cell...] = tmp
                end
            end

            particles_num ≥ min_xcell && break
        end

        inject[idx_cell...] = false
    end

    return nothing
end

@inline distance2(x, y) = sqrt(mapreduce(x -> (x[1] - x[2])^2, +, zip(x, y)))

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
        !(@cell index[ip, i, j, k]) && continue

        # distance from new point to the existing particle
        pxi = @cell(px[ip, i, j, k]), @cell(py[ip, i, j, k]), @cell(pz[ip, i, j, k])
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

@inline function new_particle(xvi::NTuple{N,T}, di::NTuple{N,T}) where {N,T}
    p_new = ntuple(Val(N)) do i
        xvi[i] + di[i] * rand(0.05:1e-5:0.95)
    end
    return p_new
end

@inline function new_particle(xvi::NTuple{2,T}, di::NTuple{2,T}, ctr, np) where {T}
    th = (2 * pi) / np * (ctr - 1)
    r = min(di...) * 0.25
    p_new = (
        muladd(di[1], 0.5, muladd(r, cos(th), xvi[1])),
        muladd(di[2], 0.5, muladd(r, sin(th), xvi[2])),
    )
    return p_new
end

@inline function new_particle(xvi::NTuple{3,T}, di::NTuple{3,T}, ctr, np) where {T}
    th = (2 * pi) / np * (ctr - 1)
    r = min(di...) * 0.25
    p_new = (
        muladd(di[1], 0.5, muladd(r, cos(th), xvi[1])),
        muladd(di[2], 0.5, xvi[2]),
        muladd(di[3], 0.5, muladd(r, cos(th), xvi[3])),
    )
    return p_new
end
