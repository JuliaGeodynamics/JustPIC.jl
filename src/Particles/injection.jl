## PARTICLE INJECTION FUNCTIONS

struct NearestInjection end
struct GridInjection end
struct LeastSquaresInjection end

function injection_scheme(scheme)
    scheme === :grid && return GridInjection()
    scheme === :least_squares && return LeastSquaresInjection()
    throw(ArgumentError("unknown injection scheme $(repr(scheme)); use `:grid` or `:least_squares`"))
end

"""
    inject_particles!(particles, args)
    inject_particles!(particles, args, fields; scheme = :grid)

Inject particles into cells whose occupancy falls below `particles.min_xcell`.

New particles are placed quadrant by quadrant. Donors are the particles live
when the call starts; particles injected by the same call are never read.

- Without `fields`, each entry of `args` is copied from the nearest donor in
  the 3^N cell neighborhood; a candidate with no donor is skipped.
- With `fields` (one grid field per entry of `args`, laid out as described in
  [`inject_particles_phase!`](@ref)), values come from `scheme`:
  - `:grid`: interpolation of `fields` at the new particle;
  - `:least_squares`: a linear least-squares fit on donors in the new
    particle's cell (or its 3^N neighborhood when the cell has fewer than
    N + 1 usable donors), clamped to the range of donor values in the
    neighborhood; falls back to `:grid` when no fit is possible.
"""
inject_particles!(particles::Particles, args) = inject_particles!(particles, args, particles.xvi, particles.di.vertex)

inject_particles!(particles::Particles, args, grid::NTuple, di) =
    _inject!(NearestInjection(), particles, (), args, (), grid, particles.xci, di, particles.di.center)

inject_particles!(particles::Particles, args, fields; scheme = :grid) = _inject!(
    injection_scheme(scheme), particles, (), args, fields,
    particles.xvi, particles.xci, particles.di.vertex, particles.di.center
)

"""
    inject_particles_phase!(particles, particles_phases, args, fields; scheme = :grid)

Phase-aware variant of [`inject_particles!`](@ref): each new particle also
copies its phase from the nearest donor. `scheme` selects how `args` are
initialized, as in `inject_particles!`.

Each entry of `fields` may independently use cell centers or vertices, with
one ghost sample per side on any axis or with no ghosts. Its size determines
the layout relative to the particle grid. Unghosted center fields use the
nearest valid interpolation stencil at boundaries, with the interpolated
value clamped to the stencil's range.
"""
inject_particles_phase!(particles::Particles, particles_phases, args, fields; scheme = :grid) =
    inject_particles_phase!(
    particles, particles_phases, args, fields,
    particles.xvi, particles.xci, particles.di.vertex, particles.di.center; scheme
)

inject_particles_phase!(
    particles::Particles, particles_phases, args, fields, grid::NTuple, grid_center, di, di_center; scheme = :grid
) = _inject!(injection_scheme(scheme), particles, (particles_phases,), args, fields, grid, grid_center, di, di_center)

check_injection_inputs(::NearestInjection, particles, phases, args, fields, grid) =
    (check_particle_fields(particles, args); ())
check_injection_inputs(scheme, particles, phases, args, fields, grid) =
    check_phase_injection_inputs(particles, phases, args, fields, grid)

function _inject!(scheme, particles, phases, args, fields, grid, grid_center, di, di_center)
    layouts = check_injection_inputs(scheme, particles, phases, args, fields, grid)
    (; coords, index, min_xcell) = particles
    # Neighbor cells are read only through donor slots, and only non-donor slots of the
    # thread's own cell are written, so all cells run concurrently.
    donor = _copy(index)
    launch!(
        ka_backend(index), inject_kernel!, inner_size(index),
        scheme, phases, args, fields, layouts, coords, index, donor, grid, grid_center, di, di_center, min_xcell
    )
    return nothing
end

@kernel function inject_kernel!(
        scheme, phases, args, fields, layouts, coords, index, donor, grid, grid_center, dxi, dxi_center, min_xcell
    )
    I = @index(Global, NTuple)
    idx_cell = I .+ 1
    di = @dxi(dxi, idx_cell...)
    xci = corner_coordinate(grid_center, idx_cell)
    ctx = (; args, fields, layouts, coords, donor, grid, grid_center, di, dxi_center, xci)
    _inject_cell!(scheme, ctx, phases, index, di ./ 2, min_xcell, idx_cell)
end

@inline function _inject_cell!(scheme, ctx, phases, index, di_quadrant, min_xcell, idx_cell)
    (; args, coords, donor, grid) = ctx
    xvi_quadrants = quadrant_corners(corner_coordinate(grid, idx_cell), di_quadrant)
    # integer ceiling division: `a / b` is a Float64 divide, which Metal cannot do
    min_xQuadrant = cld(min_xcell, length(xvi_quadrants))
    pcell = extract_particle_cell_coordinates(coords, idx_cell...)
    counts = map(xvi_quadrants) do vertex
        n = 0
        for i in cellaxes(index)
            (CAI.@index index[i, idx_cell...]) || continue
            n += isincell(extract_particle_coordinates(pcell, i), vertex, di_quadrant)
        end
        n
    end
    all(≥(min_xQuadrant), counts) && return nothing
    fit = cell_fit(scheme, ctx, idx_cell)

    for (vertex, particles_num) in zip(xvi_quadrants, counts)
        particles_num ≥ min_xQuadrant && continue
        for i in cellaxes(index)
            !(CAI.@index index[i, idx_cell...]) || continue
            p_new = new_particle(vertex, di_quadrant)
            ip_donor, I_donor = nearest_donor(scheme, args, phases, coords, p_new, donor, idx_cell)
            iszero(ip_donor) && continue
            particles_num += 1

            fill_particle!(coords, p_new, i, idx_cell)
            CAI.@index index[i, idx_cell...] = true
            copy_from_donor!(phases, ip_donor, I_donor, i, idx_cell)
            inject_values!(scheme, ctx, fit, p_new, ip_donor, I_donor, i, idx_cell)

            particles_num ≥ min_xQuadrant && break
        end
    end
    return nothing
end

# A donor is needed only to copy a phase, or `args` under `NearestInjection`.
@inline needs_donor(::NearestInjection, args, phases) = !isempty(args)
@inline needs_donor(scheme, args, phases) = !isempty(phases)

# Without a needed donor every candidate is accepted; the returned slot is never read.
@inline nearest_donor(scheme, args, phases, coords, p, donor, idx_cell) =
    needs_donor(scheme, args, phases) ? index_min_distance(coords, p, donor, idx_cell) : (1, idx_cell)

@inline copy_from_donor!(arrays, ip_donor, I_donor, ip, idx_cell) =
    foreach(A -> (CAI.@index A[ip, idx_cell...] = CAI.@index A[ip_donor, I_donor...]), arrays)

@inline inject_values!(::NearestInjection, ctx, fit, p_new, ip_donor, I_donor, ip, idx_cell) =
    copy_from_donor!(ctx.args, ip_donor, I_donor, ip, idx_cell)

@inline inject_values!(::GridInjection, ctx, fit, p_new, ip_donor, I_donor, ip, idx_cell) = inject_phase_fields!(
    ctx.args, ctx.fields, ctx.layouts, p_new, ctx.grid, ctx.grid_center, ctx.di, ctx.dxi_center, ctx.xci, ip, idx_cell
)

@inline function inject_values!(::LeastSquaresInjection, ctx, fit, p_new, ip_donor, I_donor, ip, idx_cell)
    ok, coeffs, limits, xc = fit
    ok || return inject_values!(GridInjection(), ctx, fit, p_new, ip_donor, I_donor, ip, idx_cell)
    φ = SVector(one(eltype(p_new)), ((p_new .- xc) ./ ctx.di)...)
    foreach(ctx.args, coeffs, limits) do A, c, (lo, hi)
        CAI.@index A[ip, idx_cell...] = clamp(sum(c .* φ), lo, hi)
    end
    return nothing
end

# Per-cell state shared by every particle injected into the cell.
@inline cell_fit(scheme, ctx, idx_cell) = nothing

@inline accumulate_fit((M, b, n), φ, F) = M + φ * φ', map((bj, Fj) -> bj + Fj * φ, b, F), n + 1

# Hadamard ratio |det M| / ∏ Mᵢᵢ ∈ [0, 1]; the fitted value's relative error is about
# 5 eps(T) / ratio, so requiring ratio > ∛eps(T) bounds it near eps(T)^(2/3).
@inline function solve_fit((M, b, n))
    T = eltype(M)
    ok = n ≥ size(M, 1) && abs(det(M)) > cbrt(eps(T)) * prod(i -> M[i, i], 1:size(M, 1))
    Minv = ok ? inv(M) : zero(M)
    return ok, map(bj -> Minv * bj, b)
end

# Linear fit F ≈ c₁ + c₂..ₙ₊₁⋅(x - xc)/di on donors, about the cell center xc: cell-local
# when possible, else over the 3^N neighborhood. Values are later clamped to the
# neighborhood's donor range.
@inline function cell_fit(::LeastSquaresInjection, ctx, idx_cell)
    (; args, coords, donor, di) = ctx
    N, NA = length(idx_cell), length(args)
    T = eltype(di)
    xc = corner_coordinate(ctx.grid, idx_cell) .+ di ./ 2
    own = stencil = (zero(SMatrix{N + 1, N + 1, T}), ntuple(_ -> zero(SVector{N + 1, T}), Val(NA)), 0)
    limits = ntuple(_ -> (typemax(T), typemin(T)), Val(NA))
    for J in CartesianIndices(neighborhood(donor, idx_cell)), k in cellaxes(donor)
        I = Tuple(J)
        (CAI.@index donor[k, I...]) || continue
        φ = SVector(one(T), ntuple(d -> (CAI.@index(coords[d][k, I...]) - xc[d]) / di[d], Val(N))...)
        F = map(A -> CAI.@index(A[k, I...]), args)
        stencil = accumulate_fit(stencil, φ, F)
        I == idx_cell && (own = accumulate_fit(own, φ, F))
        limits = map((l, f) -> (min(l[1], f), max(l[2], f)), limits, F)
    end
    fit = solve_fit(own)
    ok, coeffs = first(fit) ? fit : solve_fit(stencil)
    return ok, coeffs, limits, xc
end

@generated function inject_phase_fields!(
        args::NTuple{NF, Any}, fields::NTuple{NF, Any}, layouts, p_new, grid, grid_center,
        di, dxi_center, xci, ip, idx_cell
    ) where {NF}
    return quote
        Base.@_inline_meta
        Base.@nexprs $NF j -> begin
            F = fields[j]
            layout = layouts[j]
            idx = layout.iscenter ? shifted_index(p_new, xci, idx_cell) : idx_cell
            field_idx = clamp.(idx .- layout.offset, 1, size(F) .- 1)
            grid_idx = field_idx .+ layout.offset
            xi = layout.iscenter ? grid_center : grid
            spacing = layout.iscenter ? (@dxi(dxi_center, grid_idx...)) : di
            corners = field_corners(F, field_idx)
            value = _grid2particle(p_new, xi, spacing, corners, grid_idx)
            lower, upper = extrema(corners)
            CAI.@index args[j][ip, idx_cell...] = clamp(value, lower, upper)
        end
        nothing
    end
end

## UTILS

@inline neighborhood(A, idx_cell::NTuple{N}) where {N} =
    ntuple(d -> max(idx_cell[d] - 1, 1):min(idx_cell[d] + 1, size(A, d)), Val(N))

# Nearest donor to `pn` in the 3^N neighborhood of `idx_cell`; slot 0 when there is none.
@inline function index_min_distance(coords::NTuple{N}, pn, donor, idx_cell) where {N}
    # typed sentinel: a bare `Inf` is Float64 and would carry a Float64 into the kernel (fatal on Metal)
    best = (convert(eltype(pn), Inf), 0, idx_cell)
    for J in CartesianIndices(neighborhood(donor, idx_cell)), ip in cellaxes(donor)
        I = Tuple(J)
        (CAI.@index donor[ip, I...]) || continue
        d = distance(ntuple(d -> CAI.@index(coords[d][ip, I...]), Val(N)), pn)
        d < best[1] && (best = (d, ip, I))
    end
    return best[2], best[3]
end

# keep all arithmetic in the grid eltype: Float64 literals/rand() break Metal
@inline function new_particle(xvi::NTuple{N}, di::NTuple{N}) where {N}
    T = typeof(first(di))
    p_new = ntuple(Val(N)) do i
        xvi[i] + di[i] * muladd(convert(T, 0.95), rand(T), convert(T, 0.05))
    end
    return p_new
end

# Lower corners of the 2^N quadrants, first axis fastest.
@inline quadrant_corners(xvi::NTuple{N}, di_quadrant) where {N} =
    ntuple(q -> xvi .+ di_quadrant .* Tuple(CartesianIndices(ntuple(_ -> 0:1, Val(N)))[q]), Val(2^N))

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
