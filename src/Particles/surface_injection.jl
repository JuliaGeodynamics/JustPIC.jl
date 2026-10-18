"""
    inject_particles_phase!(particles, phases, args, chain::MarkerChain;
                            air_phase, boundary_initializer = nothing)

Replenish 2D particles on each side of a free surface, including thin rock slivers.
Unlike grid-field injection, this overload copies every field in `args` from the
nearest live donor on the same side of the chain. Existing valid particles retain
their coordinates and history. Pass all particle fields that must be initialized.

Wrong-side particles are first removed. Each nonempty rock/air portion of each
cell quadrant then receives at least one particle, with targets proportional to
its area and `particles.min_xcell`. The chain must use the particle grid's physical
horizontal vertices. Donors are taken from the physical 3×3 cell neighborhood and
are frozen at entry, so newly injected particles cannot become donors in this call.

For new particles in the leftmost/rightmost physical columns,
`boundary_initializer(position, is_air)` may return `(phase, values)`, where `values`
is a tuple paired with `args`. This explicitly supplies the desired incoming
material/history; it is used only for newly injected particles, on both sides
regardless of velocity. It must be callable on the selected backend.

Missing donors, invalid initializer values or insufficient capacity cause an error
after the replenishment pass. Other successful insertions/removals remain applied;
checkpoint or copy the state first if the caller needs rollback. No phase or
history is invented when neither a donor nor an initializer is available.
"""
function inject_particles_phase!(
        particles::Particles{B, 2}, phases, args::Tuple, chain::MarkerChain;
        air_phase, boundary_initializer = nothing,
    ) where {B}
    air_phase > 0 && isfinite(air_phase) && isinteger(air_phase) || throw(ArgumentError("air_phase must be a finite positive integer"))
    ka_backend(particles) == ka_backend(chain) || throw(ArgumentError("particles and chain must use the same backend"))
    for A in (phases, args...)
        size(A) == size(particles.index) && cellaxes(A) == cellaxes(particles.index) ||
            throw(DimensionMismatch("particle fields must match particle storage"))
    end
    Array(chain.cell_vertices) == Array(particles.xvi[1])[2:(end - 1)] ||
        throw(DimensionMismatch("chain vertices must match the physical particle grid"))
    all(isfinite, chain.h_vertices) || throw(ArgumentError("chain heights must be finite"))
    backend = ka_backend(particles)
    ni = inner_size(particles.index)
    launch!(
        backend, remove_wrong_side_particles!, ni, particles.coords, particles.index,
        phases, args, chain.cell_vertices, chain.h_vertices, air_phase,
    )
    donors = _copy(particles.index)
    missing = KernelAbstractions.zeros(backend, Int, ni...)
    launch!(
        backend, inject_surface_particles!, ni, particles.coords, particles.index,
        donors, phases, args, particles.xvi, particles.di.vertex,
        chain.cell_vertices, chain.h_vertices, air_phase, particles.min_xcell,
        boundary_initializer, missing,
    )
    unmet = sum(missing)
    if !iszero(unmet)
        cell = findfirst(!iszero, Array(missing))
        error("Surface injection left $unmet rock/air quadrant targets unmet (first physical cell: $(Tuple(cell))): check same-side donors, boundary_initializer and particle capacity")
    end
    return nothing
end

@inline function injection_surface_height(x, xv, h, i)
    return muladd((x - xv[i]) / (xv[i + 1] - xv[i]), h[i + 1] - h[i], h[i])
end

@kernel function remove_wrong_side_particles!(coords, index, phases, args, xv, h, air_phase)
    I = @index(Global, NTuple)
    J = I .+ 1
    for ip in cellaxes(index)
        (CAI.@index index[ip, J...]) || continue
        x, y = cache_particle(coords, ip, J)
        phase = CAI.@index phases[ip, J...]
        above = y > injection_surface_height(x, xv, h, I[1])
        if !isfinite(phase) || phase ≤ zero(phase) || !isinteger(phase) || !isfinite(x) || !isfinite(y) ||
                !all(A -> isfinite(CAI.@index A[ip, J...]), args) || above != (phase == air_phase)
            CAI.@index index[ip, J...] = false
            for A in (coords..., phases, args...)
                CAI.@index A[ip, J...] = convert(eltype(eltype(A)), NaN)
            end
        end
    end
end

@kernel function inject_surface_particles!(
        coords, index, donors, phases, args, grid, dxi, xv, h, air_phase,
        min_xcell, boundary_initializer, missing,
    )
    I = @index(Global, NTuple)
    J = I .+ 1
    di = @dxi dxi J...
    half = di ./ 2
    corner = corner_coordinate(grid, J)
    nquad = cld(min_xcell, 4)
    unmet = 0
    boundary = I[1] == 1 || I[1] == size(missing, 1)
    for vertex in quadrant_corners(corner, half)
        h0 = injection_surface_height(vertex[1], xv, h, I[1])
        h1 = injection_surface_height(vertex[1] + half[1], xv, h, I[1])
        segment = Segment(GridGeometryUtils.Point(vertex[1], h0), GridGeometryUtils.Point(vertex[1] + half[1], h1))
        fraction = cell_rock_area(segment, rectangle_from_min_corner(vertex, half))
        for is_air in (false, true)
            area_fraction = is_air ? one(fraction) - fraction : fraction
            area_fraction > zero(fraction) || continue
            target = max(1, ceil(Int, nquad * area_fraction))
            count = 0
            for ip in cellaxes(index)
                (CAI.@index index[ip, J...]) || continue
                p = cache_particle(coords, ip, J)
                phase = CAI.@index phases[ip, J...]
                count += isincell(p, vertex, half) && ((phase == air_phase) == is_air)
            end
            count ≥ target && continue
            for ip in cellaxes(index)
                (CAI.@index index[ip, J...]) && continue
                p = surface_region_sample(vertex, half, h0, h1, is_air)
                phase, values, valid = surface_injection_values(
                    boundary_initializer, boundary, p, is_air, donors, coords,
                    phases, args, J, air_phase,
                )
                valid || continue
                ((phase == air_phase) == is_air) || continue
                # Guard slivers too small to represent in the coordinate precision.
                (p[2] > injection_surface_height(p[1], xv, h, I[1])) == is_air || continue
                fill_particle!(coords, p, ip, J)
                CAI.@index phases[ip, J...] = phase
                for k in eachindex(args)
                    CAI.@index args[k][ip, J...] = values[k]
                end
                CAI.@index index[ip, J...] = true
                count += 1
                count ≥ target && break
            end
            unmet += count < target
        end
    end
    missing[I...] = unmet
end

# Restrict x to where the requested region has positive height, then sample y
# inside that region. This reaches arbitrarily thin slivers without rejection.
@inline function surface_region_sample(vertex, di, h0, h1, is_air)
    T = typeof(h0)
    bottom, top = vertex[2], vertex[2] + di[2]
    lo, hi = zero(T), one(T)
    level = is_air ? top : bottom
    if is_air ? h0 ≥ level : h0 ≤ level
        lo = clamp((level - h0) / (h1 - h0), zero(T), one(T))
    end
    if is_air ? h1 ≥ level : h1 ≤ level
        hi = clamp((level - h0) / (h1 - h0), zero(T), one(T))
    end
    t = muladd(clamp(rand(T), eps(T), one(T) - eps(T)), hi - lo, lo)
    x = muladd(t, di[1], vertex[1])
    surface = clamp(muladd(t, h1 - h0, h0), bottom, top)
    ylo, yhi = is_air ? (surface, top) : (bottom, surface)
    y = muladd(clamp(rand(T), eps(T), one(T) - eps(T)), yhi - ylo, ylo)
    return x, y
end

@inline function surface_injection_values(initializer, boundary, p, is_air, donors, coords, phases, args, J, air_phase)
    if boundary
        phase, values = initializer(p, is_air)
        valid = isfinite(phase) && phase > zero(phase) && isinteger(phase) && length(values) == length(args) && all(isfinite, values)
        return phase, values, valid
    end
    return surface_injection_values(nothing, boundary, p, is_air, donors, coords, phases, args, J, air_phase)
end

@inline function surface_injection_values(::Nothing, boundary, p, is_air, donors, coords, phases, args, J, air_phase)
    best = convert(typeof(p[1]), Inf)
    donor, cell = 0, J
    for j in max(2, J[2] - 1):min(size(donors, 2) - 1, J[2] + 1)
        for i in max(2, J[1] - 1):min(size(donors, 1) - 1, J[1] + 1)
            K = (i, j)
            for ip in cellaxes(donors)
                (CAI.@index donors[ip, K...]) || continue
                phase = CAI.@index phases[ip, K...]
                (phase == air_phase) == is_air || continue
                all(A -> isfinite(CAI.@index A[ip, K...]), args) || continue
                d = distance(cache_particle(coords, ip, K), p)
                if d < best
                    best, donor, cell = d, ip, K
                end
            end
        end
    end
    # A typed dummy tuple keeps the no-donor path inferable on devices.
    values = map(A -> iszero(donor) ? zero(eltype(eltype(A))) : (CAI.@index A[donor, cell...]), args)
    phase = iszero(donor) ? zero(eltype(eltype(phases))) : (CAI.@index phases[donor, cell...])
    return phase, values, !iszero(donor)
end
