# Fixtures for the `move_particles!` slot-occupancy regression tests (2D and 3D).
import KernelAbstractions: @kernel, @index

# Writes particle `n` into slot `slots[n]` of cell `(cells[1][n], cells[2][n], ...)`;
# companion field `k` receives `k * ids[n]`.
@kernel function place_particles_kernel!(
        index, coords, fields, cells::NTuple{N}, slots, pos::NTuple{N}, ids
    ) where {N}
    n = @index(Global)
    I = ntuple(d -> cells[d][n], Val(N))
    ip = slots[n]
    CAI.@index index[ip, I...] = true
    for d in 1:N
        CAI.@index coords[d][ip, I...] = pos[d][n]
    end
    for k in eachindex(fields)
        CAI.@index fields[k][ip, I...] = k * ids[n]
    end
end

function clear_particles!(particles, fields)
    fill!(particles.index.data, false)
    foreach((particles.coords..., fields...)) do x
        fill!(x.data, convert(eltype(x.data), NaN))
    end
    return nothing
end

# One source cell holds `leaving[k]` in slot `k`: a particle whose coordinates lie in the
# neighbour `source + leaving[k]` (or in `source` itself for a zero offset). Every
# destination cell listed in `free` is otherwise full, keeping only `free[offset]` empty.
# After `move_particles!` each particle must sit, with its fields and coordinates, in a
# free slot of its own destination cell, and the residents must not have moved.
function check_fragmented_move(
        backend, particles, fields, source::NTuple{N, Int}, leaving, free
    ) where {N}
    T = eltype(eltype(particles.coords[1]))
    max_xcell = particles.max_xcell
    @assert length(leaving) ≤ max_xcell
    xci = Array.(particles.xci)
    centre(cell) = ntuple(d -> xci[d][cell[d]], Val(N))

    cells = NTuple{N, Int}[]
    slots = Int[]
    pos = NTuple{N, T}[]
    expected = Dict{T, Any}()
    function add!(cell, slot, x, cell_after, slots_after)
        push!(cells, cell)
        push!(slots, slot)
        push!(pos, x)
        expected[T(length(cells))] = (; cell = cell_after, slots = slots_after, x)
        return nothing
    end

    for (slot, offset) in enumerate(leaving)
        dest = source .+ offset
        moves = any(!iszero, offset)
        add!(source, slot, centre(dest), moves ? dest : source, moves ? free[offset] : [slot])
    end
    for (offset, free_slots) in free, slot in setdiff(1:max_xcell, free_slots)
        dest = source .+ offset
        add!(dest, slot, centre(dest), dest, [slot])
    end

    total = length(cells)
    clear_particles!(particles, fields)
    dev = TA(backend)
    JustPIC.launch!(
        JustPIC.ka_backend(particles.index), place_particles_kernel!, total,
        particles.index, particles.coords, fields,
        ntuple(d -> dev(getindex.(cells, d)), Val(N)), dev(slots),
        ntuple(d -> dev(getindex.(pos, d)), Val(N)), dev(T.(1:total)),
    )
    move_particles!(particles, fields)

    index_h = to_cpu(particles.index)
    coords_h = to_cpu.(particles.coords)
    fields_h = to_cpu.(fields)
    found = NamedTuple[]
    for I in CartesianIndices(size(index_h)), ip in 1:JustPIC.cellnum(index_h)
        cell = Tuple(I)
        CAI.@index(index_h[ip, cell...]) || continue
        push!(
            found, (;
                id = CAI.@index(fields_h[1][ip, cell...]),
                cell, slot = ip,
                x = map(c -> CAI.@index(c[ip, cell...]), coords_h),
                f = map(f -> CAI.@index(f[ip, cell...]), fields_h),
            )
        )
    end

    @test length(found) == total
    @test sort(map(p -> p.id, found)) == T.(1:total)
    @test all(p -> p.cell == expected[p.id].cell, found)
    @test all(p -> p.slot in expected[p.id].slots, found)
    @test all(p -> p.x == expected[p.id].x, found)
    @test all(p -> p.f == ntuple(k -> k * p.id, length(fields)), found)
    return nothing
end
