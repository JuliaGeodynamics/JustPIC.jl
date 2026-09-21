# Helpers to stress `move_particles!` with particles that jump several cells at once.
# Included by `test_2D.jl` and `test_3D.jl`, which define `backend` and `FT`.

import KernelAbstractions: @kernel, @index

# deterministic pseudo-random number in [0, 1), so that failures are reproducible
unit_random(tags...) = (hash(tags) >> 11) / 2.0^53

@kernel function load_particles_kernel!(index, coords, args, occupied, positions, values)
    I = @index(Global, NTuple)
    for ip in cellaxes(index)
        CAI.@index index[ip, I...] = occupied[ip, I...]
        for d in eachindex(coords)
            CAI.@index coords[d][ip, I...] = positions[d][ip, I...]
        end
        for f in eachindex(args)
            CAI.@index args[f][ip, I...] = values[f][ip, I...]
        end
    end
end

# Fill `particles` with random particles, each stored in a randomly chosen source cell while its
# coordinates lie in a destination cell up to `max_jump[d]` cells away along direction `d` (a
# single number applies to every direction), wrapping around the seam of the periodic directions.
# The cell occupancy is fragmented. Every particle carries a unique id in `args[1]` and a
# function of it in the other companion fields. Returns the coordinates of every id.
function load_jumping_particles!(particles, args, max_jump, periodicity; occupancy = 0.15, seed = 1)
    (; index, coords) = particles
    N = length(coords)
    max_jump = max_jump isa Integer ? ntuple(_ -> max_jump, N) : max_jump
    ncells = size(index)
    nslots = cellnum(index)
    xvi = Array.(particles.xvi)

    occupied = zeros(Bool, nslots, ncells...)
    positions = ntuple(_ -> fill(FT(NaN), nslots, ncells...), N)
    values = ntuple(_ -> fill(FT(NaN), nslots, ncells...), length(args))
    expected = Dict{FT, NTuple{N, FT}}()

    for cell in CartesianIndices(ntuple(d -> 2:(ncells[d] - 1), Val(N))), slot in 1:nslots
        I = Tuple(cell)
        unit_random(seed, 1, slot, I...) < occupancy || continue
        destination = ntuple(Val(N)) do d
            jump = floor(Int, (2 * max_jump[d] + 1) * unit_random(seed, 2, d, slot, I...)) - max_jump[d]
            periodicity[d] ? 2 + mod(I[d] + jump - 2, ncells[d] - 2) : clamp(I[d] + jump, 2, ncells[d] - 1)
        end
        x = ntuple(Val(N)) do d
            fraction = FT(0.05) + FT(0.9) * FT(unit_random(seed, 3, d, slot, I...))
            xvi[d][destination[d]] + fraction * (xvi[d][destination[d] + 1] - xvi[d][destination[d]])
        end
        id = FT(length(expected) + 1)
        occupied[slot, I...] = true
        for d in 1:N
            positions[d][slot, I...] = x[d]
        end
        for f in eachindex(args)
            values[f][slot, I...] = f == 1 ? id : f * id + 1
        end
        expected[id] = x
    end

    to_device = TA(backend)
    JustPIC.launch!(
        JustPIC.ka_backend(index), load_particles_kernel!, ncells,
        index, coords, args, to_device(occupied), map(to_device, positions), map(to_device, values)
    )
    return expected
end

# Compare the particles stored in `particles` with the `expected` coordinates of every id.
function audit_jumping_particles(particles, args, expected)
    index = to_cpu(particles.index)
    coords = to_cpu.(particles.coords)
    fields = to_cpu.(args)
    xvi = Array.(particles.xvi)
    N = length(coords)

    found = Set{FT}()
    duplicated = misplaced = corrupted = 0
    for cell in CartesianIndices(size(index)), slot in 1:cellnum(index)
        I = Tuple(cell)
        CAI.@index(index[slot, I...]) || continue
        id = CAI.@index(fields[1][slot, I...])
        id in found ? (duplicated += 1) : push!(found, id)
        x = ntuple(d -> CAI.@index(coords[d][slot, I...]), Val(N))
        misplaced += !all(d -> xvi[d][I[d]] ≤ x[d] < xvi[d][I[d] + 1], 1:N)
        corrupted += !haskey(expected, id) || x != expected[id] ||
            any(f -> CAI.@index(fields[f][slot, I...]) != f * id + 1, 2:length(fields))
    end
    return (; found = length(found), duplicated, misplaced, corrupted, lost = length(setdiff(keys(expected), found)))
end
