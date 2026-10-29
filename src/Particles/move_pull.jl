# Pull scheme for `move_particles!` on non-periodic grids when every particle is at most one
# cell away from its parent cell along each direction.
#
# Every slot is first classified in one launch over all cells. Then rounds of two launches
# follow, each over all cells: a destination cell copies the particles bound for it from its
# neighbors into its own free slots (pull), and every cell removes the particles that were
# copied away or left the domain (clear). A cell writes only its own slots and the codes of
# the particles it copies, which no other cell copies, so both launches are race free without
# coloring. A destination fills only the slots that were free when the round started; slots
# freed by the clear are used in the next round, so a full cell accepts particles once its
# own outgoing particles have left (#378). When a round removes nothing, the waiting particles
# are bound for full cells: first those bound for a cell with no waiting particles of its own
# (it stays full) are dropped and the rounds resume; if that removes nothing either, the
# remaining particles wait on each other in a cycle of full cells and are all dropped.

# Slot codes. A pending move stores the index in `1:3^N` of the offset from the source cell to
# the destination cell (see `offset_code`); the other codes lie outside that range.
const SLOT_STAY = Int8(0)
const SLOT_FAR = Int8(-1)
const SLOT_OUT = Int8(-2)
const SLOT_PULLED = Int8(-3)

@inline offset_code(offset::NTuple{N, Int}) where {N} =
    Int8(1 + sum(ntuple(d -> (offset[d] + 1) * 3^(d - 1), Val(N))))

@inline code_offset(code::Integer, ::Val{N}) where {N} =
    ntuple(d -> (Int(code) - 1) ÷ 3^(d - 1) % 3 - 1, Val(N))

# `outgoing[I]` has bit `k - 1` set if cell `I` holds a waiting particle with code `k`, and
# `OUT_BIT` set if it holds a particle that left the domain; zero means nothing to clear.
@inline code_bit(code::Integer) = one(UInt32) << (Int(code) - 1)
const OUT_BIT = one(UInt32) << 31

# Positions in the flag and counter buffer
const FAR, MOVING, PENDING, PROGRESS, DROPPED = 1, 2, 3, 4, 5

# Move the particles with the pull scheme and return the number dropped because their
# destination stayed full, or `nothing`, without changing anything, if a particle is more
# than one cell away from its parent cell.
function pull_particles!(particles, grid, args, domain_limits)
    (; coords, index) = particles
    backend = ka_backend(index)
    code = KernelAbstractions.zeros(backend, Int8, size(index)..., cellnum(index))
    outgoing = KernelAbstractions.zeros(backend, UInt32, size(index))
    # Int32: Metal has no 64-bit integer atomics (see #362). One buffer, so each round costs
    # one copy to the device and one back.
    counters = KernelAbstractions.zeros(backend, Int32, 5)
    state = zeros(Int32, 5)

    launch!(
        backend, classify_moves_kernel!, size(index),
        code, outgoing, coords, grid, index, domain_limits, counters
    )
    copyto!(state, counters)
    iszero(state[FAR]) || return nothing
    iszero(state[MOVING]) && return 0

    # 0: pull and clear; 1: also drop particles bound for settled cells; 2: drop all waiting
    level = 0
    while true
        state[PENDING] = state[PROGRESS] = 0
        copyto!(counters, state)
        iszero(level) && launch!(backend, pull_particles_kernel!, size(index), code, outgoing, coords, index, args)
        launch!(
            backend, clear_moved_kernel!, size(index),
            code, coords, index, args, outgoing, level, counters
        )
        copyto!(state, counters)
        iszero(state[PENDING]) && break
        level = iszero(state[PROGRESS]) ? level + 1 : 0
    end
    return Int(state[DROPPED])
end

# Cache the parent bounds once per cell. Uniform ranges also cache the neighboring
# bounds: recomputing a LinRange node for every particle is costly in Float64.
@inline move_cell_bounds(grid::AbstractVector, i) = (grid[i], grid[i + 1])
@inline move_cell_bounds(grid::AbstractRange, i) =
    (grid[max(i - 1, 1)], grid[i], grid[i + 1], grid[min(i + 2, length(grid))])

@inline function move_cell_index(x, grid, i, bounds::NTuple{2})
    lower, upper = bounds
    lower <= x < upper && return i
    return half_open_cell(x, grid, i)
end

@inline function move_cell_index(x, grid, i, bounds::NTuple{4})
    left, lower, upper, right = bounds
    lower <= x < upper && return i
    i > 1 && left <= x < lower && return i - 1
    i < length(grid) - 1 && upper <= x < right && return i + 1
    return half_open_cell(x, grid, i)
end

@kernel function classify_moves_kernel!(code, outgoing, coords, grid, index, domain_limits, counters)
    I = @index(Global, NTuple)
    N = length(I)
    bounds = ntuple(d -> move_cell_bounds(grid[d], I[d]), Val(N))
    anyfar = anymoving = false
    mask = zero(UInt32)
    for ip in cellaxes(index)
        c = SLOT_STAY
        if !doskip(index, ip, I...)
            pᵢ = cache_particle(coords, ip, I)
            new_cell = ntuple(d -> move_cell_index(pᵢ[d], grid[d], I[d], bounds[d]), Val(N))
            if new_cell != I
                offset = new_cell .- I
                c = if !indomain(pᵢ, domain_limits)
                    SLOT_OUT
                elseif any(x -> abs(x) > 1, offset)
                    SLOT_FAR
                else
                    offset_code(offset)
                end
                anymoving = true
                anyfar |= c == SLOT_FAR
                mask |= c > 0 ? code_bit(c) : ifelse(c == SLOT_OUT, OUT_BIT, zero(UInt32))
            end
        end
        code[I..., ip] = c
    end
    outgoing[I...] = mask
    raise!(counters, FAR, anyfar)
    raise!(counters, MOVING, anymoving)
end

# `outgoing` may have bits set for particles no longer waiting; it only prunes the scan.
@kernel function pull_particles_kernel!(code, outgoing, coords, index, args)
    D = @index(Global, NTuple)
    N = length(D)
    nslots = lastindex(cellaxes(index))
    free = 0
    for k in 1:(3^N)
        free > nslots && break
        offset = code_offset(k, Val(N))
        all(iszero, offset) && continue
        S = D .- offset
        all(ntuple(d -> 1 ≤ S[d] ≤ size(index, d), Val(N))) || continue
        iszero(outgoing[S...] & code_bit(k)) && continue
        for ip in cellaxes(index)
            code[S..., ip] == k || continue
            free = next_free_slot(index, free, D)
            free > nslots && break
            CAI.@index index[free, D...] = true
            fill_particle!(coords, cache_particle(coords, ip, S), free, D)
            fill_particle!(args, cache_args(args, ip, S), free, D)
            code[S..., ip] = SLOT_PULLED
        end
    end
end

# `outgoing` is rewritten only at level 0 and read for neighbor cells only at level 1, so no
# launch both reads and writes it.
@kernel function clear_moved_kernel!(code, coords, index, args, outgoing, level, counters)
    I = @index(Global, NTuple)
    N = length(I)
    anypending = anyprogress = false
    ndropped = 0
    mask = zero(UInt32)
    idle = iszero(outgoing[I...])
    for ip in cellaxes(index)
        idle && break
        c = code[I..., ip]
        c == SLOT_STAY && continue
        waiting = c > 0
        if waiting
            remove = level == 2 || (level == 1 && iszero(outgoing[(I .+ code_offset(c, Val(N)))...]))
            if !remove
                anypending = true
                mask |= code_bit(c)
                continue
            end
        end
        CAI.@index index[ip, I...] = false
        empty_particle!(coords, ip, I)
        empty_particle!(args, ip, I)
        code[I..., ip] = SLOT_STAY
        anyprogress = true
        ndropped += waiting
    end
    iszero(level) && (outgoing[I...] = mask)
    raise!(counters, PENDING, anypending)
    raise!(counters, PROGRESS, anyprogress)
    iszero(ndropped) || KernelAbstractions.@atomic counters[DROPPED] += convert(eltype(counters), ndropped)
end

# First slot after `last` that is free in cell `I`, or `lastindex(cellaxes(index)) + 1`.
@inline function next_free_slot(index, last, I)
    for i in cellaxes(index)
        i > last && !(CAI.@index(index[i, I...])) && return i
    end
    return lastindex(cellaxes(index)) + 1
end
