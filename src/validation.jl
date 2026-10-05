# Argument validation for the public entry points.
#
# Every check below runs on the host before the entry point launches a kernel that writes
# user data, and throws an `ArgumentError` or `DimensionMismatch` that names the offending
# argument. The policies they enforce are documented in `docs/src/validation.md`.

# Scalar element type of a field: the entry type of a `CellArray`'s cells, or the eltype of
# a plain array.
@inline scalar_eltype(A::CellArray) = eltype(eltype(A))
@inline scalar_eltype(A::AbstractArray) = eltype(A)

@inline as_tuple(x::Tuple) = x
@inline as_tuple(x) = (x,)

"""
    supports_float64(backend)

Whether the KernelAbstractions `backend` (an instance or a type) can hold `Float64`
data. `true` except on Metal, whose extension returns `false`.
"""
supports_float64(::Any) = true

function check_backend_precision(backend, ::Type{T}) where {T}
    T === Float64 && !supports_float64(backend) && throw(
        ArgumentError(
            "the $(backend_name(backend)) backend does not support Float64; build the grids in Float32"
        )
    )
    return nothing
end

backend_name(backend::Type) = string(nameof(backend))
backend_name(backend) = backend_name(typeof(backend))

# Every array in `x` (a field or a tuple of fields) must live on `backend`. Ranges carry no
# device storage and are accepted on every backend.
function check_backend(name, x, backend)
    for a in as_tuple(x)
        a isa AbstractArray && !(a isa AbstractRange) || continue
        b = ka_backend(a)
        typeof(b) === typeof(backend) || throw(
            ArgumentError(
                "`$name` is stored on the $(backend_name(b)) backend, but the particles are on $(backend_name(backend))"
            )
        )
    end
    return nothing
end

# Every floating-point array in `x` must have scalar element type `T`. Mixed precision is
# rejected rather than converted, because a narrowing conversion inside a kernel is silent.
function check_precision(name, x, ::Type{T}) where {T}
    for a in as_tuple(x)
        a isa AbstractArray || continue
        S = scalar_eltype(a)
        S <: AbstractFloat && S !== T && throw(
            ArgumentError(
                "`$name` has element type $S, but the particle coordinates are $T; convert it to $T (mixed precision is not supported)"
            )
        )
    end
    return nothing
end

# Fields that hold a `NaN` sentinel in empty particle slots must be floating point.
function check_float_field(name, x)
    for a in as_tuple(x)
        scalar_eltype(a) <: AbstractFloat || throw(
            ArgumentError(
                "`$name` has element type $(scalar_eltype(a)); particle fields must be floating point so that empty slots can hold NaN"
            )
        )
    end
    return nothing
end

function check_size(name, x, dims::Tuple)
    for a in as_tuple(x)
        size(a) == dims || throw(
            DimensionMismatch("size(`$name`) = $(size(a)), expected $dims")
        )
    end
    return nothing
end

# Every entry of `x` must be a `CellArray` with the cell layout of `index`.
function check_cell_layout(name, x, index)
    for a in as_tuple(x)
        a isa CellArray || throw(
            ArgumentError("`$name` must be a CellArray with the cell layout of `particles.index`, got $(typeof(a))")
        )
        size(a) == size(index) && cellnum(a) == cellnum(index) || throw(
            DimensionMismatch(
                "`$name` has $(cellnum(a)) slots per cell on a $(size(a)) grid, but the particles have $(cellnum(index)) slots per cell on a $(size(index)) grid"
            )
        )
    end
    return nothing
end

# A destination and its source must both be single fields or tuples of the same length.
function check_field_pairing(name_a, a, name_b, b)
    (a isa Tuple) == (b isa Tuple) && length(as_tuple(a)) == length(as_tuple(b)) || throw(
        ArgumentError(
            "`$name_a` and `$name_b` must both be single fields or tuples of the same length"
        )
    )
    return nothing
end

@inline dataids(A::CellArray) = Base.dataids(A.data)
@inline dataids(A) = Base.dataids(A)
@inline mightalias(a, b) = !Base._isdisjoint(dataids(a), dataids(b))

# No two arrays among the named arguments may share memory. Each argument is a `name => x`
# pair, where `x` is a field or a tuple of fields.
function check_distinct(args::Pair...)
    arrays = Tuple{String, Any}[]
    for (name, x) in args, a in as_tuple(x)
        a isa AbstractArray && !(a isa AbstractRange) && push!(arrays, (string(name), a))
    end
    for i in eachindex(arrays), j in (i + 1):lastindex(arrays)
        (name_i, a), (name_j, b) = arrays[i], arrays[j]
        mightalias(a, b) || continue
        what = name_i == name_j ? "entries of `$name_i`" : "`$name_i` and `$name_j`"
        throw(ArgumentError("$what share memory; pass separate arrays"))
    end
    return nothing
end

check_finite_scalar(name, x) =
    isfinite(x) || throw(ArgumentError("`$name` must be finite, got $x"))

function check_integrator(method)
    method isa Euler || method isa RungeKutta2 || method isa RungeKutta4 || throw(
        ArgumentError("unsupported advection integrator: $(typeof(method)); use Euler, RungeKutta2, or RungeKutta4")
    )
    return nothing
end

# Staggered velocity arrays `V` must match the coordinate grids `grid_vi` they are sampled on.
function check_velocity(V, grid_vi::NTuple{N}, backend, ::Type{T}) where {N, T}
    V isa Tuple && length(V) == N || throw(
        ArgumentError("`V` must be a tuple of $N velocity components")
    )
    for d in 1:N
        name = "V[$d]"
        check_size(name, V[d], map(length, grid_vi[d]))
        check_backend(name, V[d], backend)
        check_precision(name, V[d], T)
    end
    return nothing
end

## Device-side counts over the active slots of a particle container

struct NoNaN end
@inline (::NoNaN)(v) = !any(isnan, v)

# A phase label is valid when it names one of the phases `1:n`.
struct PhaseLabelInRange
    n::Int
end
@inline function (f::PhaseLabelInRange)(v)
    ph = first(v)
    return isinteger(ph) & (one(ph) ≤ ph) & (ph ≤ f.n)
end

# Number of active slots of `index` whose values in `fields` fail the predicate `valid`.
function count_invalid_active(valid, index, fields::Tuple)
    backend = ka_backend(index)
    # Int32 counter: Metal has no 64-bit integer atomics
    counter = KernelAbstractions.zeros(backend, Int32, 1)
    launch!(backend, count_invalid_active_kernel!, size(index), counter, valid, index, fields)
    return Int(maximum(counter))
end

@kernel function count_invalid_active_kernel!(counter, valid, index, fields::NTuple{NF, Any}) where {NF}
    I = @index(Global, NTuple)
    n = Int32(0)
    for ip in cellaxes(index)
        doskip(index, ip, I...) && continue
        v = ntuple(k -> CAI.@index(fields[k][ip, I...]), Val(NF))
        valid(v) || (n += Int32(1))
    end
    if n > 0
        KernelAbstractions.@atomic counter[1] += n
    end
end

# Infinite coordinates are valid input: they mark particles that left the velocity grid
# during advection, which `move_particles!` removes like any other out-of-domain particle.
function check_active_coordinates(particles)
    n = count_invalid_active(NoNaN(), particles.index, particles.coords)
    iszero(n) || throw(
        ArgumentError(
            "$n active particle slots have NaN coordinates; check the velocity field and the time step"
        )
    )
    return nothing
end

function check_phase_labels(phases, index, nphases::Integer)
    n = count_invalid_active(PhaseLabelInRange(nphases), index, (phases,))
    iszero(n) || throw(
        ArgumentError(
            "$n active particles have a phase label that is not an integer in 1:$nphases"
        )
    )
    return nothing
end

# Passive markers carry no occupancy mask, so every marker must lie inside the grid.
function check_markers_inside(coords::Tuple, grid::Tuple)
    for d in eachindex(coords, grid)
        lo, hi = extrema(grid[d])
        n = count(x -> !(lo ≤ x ≤ hi), coords[d])
        iszero(n) || throw(
            ArgumentError(
                "$n passive markers have a coordinate $d outside the grid [$lo, $hi] or not finite"
            )
        )
    end
    return nothing
end

function check_no_mpi(what)
    ImplicitGlobalGrid.grid_is_initialized() && ImplicitGlobalGrid.global_grid().nprocs > 1 &&
        throw(ArgumentError("$what does not support MPI runs with more than one rank"))
    return nothing
end

## Grids

# A coordinate vector must hold at least two finite, strictly increasing entries.
function check_grid_vector(name, x::AbstractVector)
    length(x) ≥ 2 || throw(ArgumentError("`$name` must contain at least two coordinates"))
    all(isfinite, x) || throw(ArgumentError("`$name` must contain only finite coordinates"))
    all(>(zero(eltype(x))), diff(x)) || throw(ArgumentError("`$name` must be strictly increasing"))
    return nothing
end

# Staggered velocity grids: `xi_vel[d][d]` is the vertex grid of direction `d`, and every
# off-diagonal `xi_vel[d][i]` is the cell-center grid of direction `i` extended by one ghost
# center on each side.
function check_staggered_grids(backend, xi_vel::NTuple{N, NTuple{N, AbstractVector}}) where {N}
    N in (2, 3) || throw(ArgumentError("particles must be 2D or 3D, got $N velocity grids"))
    T = eltype(xi_vel[1][1])
    T <: AbstractFloat || throw(ArgumentError("grid coordinates must be floating point, got $T"))
    for d in 1:N, i in 1:N
        name = "grid_v$("xyz"[d])[$i]"
        x = xi_vel[d][i]
        check_grid_vector(name, x)
        eltype(x) === T || throw(
            ArgumentError("`$name` has element type $(eltype(x)), expected $T; all grid coordinates must share one precision")
        )
        i == d && continue
        nv = length(xi_vel[i][i])
        length(x) == nv + 1 || throw(
            DimensionMismatch(
                "`$name` has $(length(x)) entries; a cell-center grid with one ghost center per side needs $(nv + 1) (the vertex grid `grid_v$("xyz"[i])[$i]` has $nv)"
            )
        )
    end
    check_backend_precision(backend, T)
    return nothing
end

function check_particle_capacity(nxcell, max_xcell, min_xcell)
    nxcell isa Number && nxcell < 0 && throw(ArgumentError("`nxcell` must be non-negative, got $nxcell"))
    max_xcell > 0 || throw(ArgumentError("`max_xcell` must be positive, got $max_xcell"))
    min_xcell ≥ 0 || throw(ArgumentError("`min_xcell` must be non-negative, got $min_xcell"))
    return nothing
end

# Per-particle companion fields moved, injected, or cleaned together with the coordinates.
function check_particle_fields(particles, args; name = "args")
    args isa Tuple || throw(ArgumentError("`$name` must be a tuple of particle fields, got $(typeof(args))"))
    check_cell_layout(name, args, particles.index)
    check_backend(name, args, ka_backend(particles))
    check_float_field(name, args)
    check_distinct("particles.coords" => particles.coords, name => args)
    return nothing
end

function check_advection_inputs(particles, method, V, grid_vi, dt)
    check_integrator(method)
    check_finite_scalar("dt", dt)
    check_velocity(V, grid_vi, ka_backend(particles), scalar_eltype(particles.coords[1]))
    return nothing
end

## Interpolation fields

# Size of a field sampled on the particle grid `xi`, whose coordinate vectors include one
# ghost node on each side: the full length along directions flagged in `ghosts`, the
# physical length along the others.
@inline ghosted_size(xi::NTuple{N}, ghosts) where {N} =
    ntuple(d -> length(xi[d]) - 2 * !ghosts[d], Val(N))

function check_grid_field(name, F, dims, particles)
    check_size(name, F, dims)
    check_backend(name, F, ka_backend(particles))
    check_precision(name, F, scalar_eltype(particles.coords[1]))
    return nothing
end

function check_particle_field(name, Fp, particles)
    check_cell_layout(name, Fp, particles.index)
    check_backend(name, Fp, ka_backend(particles))
    check_precision(name, Fp, scalar_eltype(particles.coords[1]))
    return nothing
end

# A particle ↔ grid transfer between the particle field `Fp` and the grid field `F`.
function check_transfer(Fp, F, dims, particles)
    check_field_pairing("Fp", Fp, "F", F)
    check_particle_field("Fp", Fp, particles)
    check_grid_field("F", F, dims, particles)
    check_distinct("Fp" => Fp, "F" => F)
    return nothing
end

## Phase ratios

function check_phase_ratio_field(name, ratios, dims, particles)
    check_size(name, ratios, dims)
    check_backend(name, ratios, ka_backend(particles))
    check_precision(name, ratios, scalar_eltype(particles.coords[1]))
    return nothing
end

function check_phase_inputs(particles, phases, nphases)
    check_cell_layout("phases", phases, particles.index)
    check_backend("phases", phases, ka_backend(particles))
    check_phase_labels(phases, particles.index, nphases)
    return nothing
end

function check_phase_ratio_inputs(phase_ratios, particles::Particles{B, N}, phases) where {B, N}
    ndims(phase_ratios.center) == N || throw(
        ArgumentError("the phase ratios are $(ndims(phase_ratios.center))D but the particles are $(N)D")
    )
    ni = size(particles.index) .- 2
    check_phase_ratio_field("phase_ratios.center", phase_ratios.center, ni, particles)
    check_phase_ratio_field("phase_ratios.vertex", phase_ratios.vertex, ni .+ 1, particles)
    for (d, name) in enumerate((:Vx, :Vy, :Vz)[1:N])
        offsets = ntuple(i -> Int(i == d), Val(N))
        check_phase_ratio_field("phase_ratios.$name", getfield(phase_ratios, name), ni .+ offsets, particles)
    end
    if N == 3
        for name in (:yz, :xz, :xy)
            offsets = midpoint_offset(Val(3), name)
            check_phase_ratio_field("phase_ratios.$name", getfield(phase_ratios, name), ni .+ offsets, particles)
        end
    end
    check_phase_inputs(particles, phases, numphases(phase_ratios))
    return nothing
end

function check_phase_ratio_allocation(::Type{T}, backend, nphases, ni) where {T}
    T <: AbstractFloat || throw(ArgumentError("phase ratios must be floating point, got $T"))
    nphases ≥ 1 || throw(ArgumentError("`nphases` must be at least 1, got $nphases"))
    all(>(0), ni) || throw(ArgumentError("the grid size must be positive, got $ni"))
    check_backend_precision(backend, T)
    return nothing
end

## Passive markers

function check_marker_coordinates(backend, coords::NTuple{N, AbstractArray}) where {N}
    N in (2, 3) || throw(ArgumentError("passive markers must be 2D or 3D, got $N coordinate arrays"))
    T = eltype(coords[1])
    T <: AbstractFloat || throw(ArgumentError("marker coordinates must be floating point, got $T"))
    np = length(coords[1])
    for d in 1:N
        name = "coords[$d]"
        length(coords[d]) == np || throw(
            DimensionMismatch("`$name` has $(length(coords[d])) entries, but `coords[1]` has $np")
        )
        eltype(coords[d]) === T || throw(
            ArgumentError("`$name` has element type $(eltype(coords[d])), expected $T")
        )
        ka_backend(coords[d]) isa backend || throw(
            ArgumentError("`$name` is stored on the $(backend_name(ka_backend(coords[d]))) backend, not on $(backend_name(backend))")
        )
        all(isfinite, coords[d]) || throw(ArgumentError("`$name` must contain only finite coordinates"))
    end
    check_backend_precision(backend, T)
    return nothing
end

function check_marker_field(name, Fp, markers)
    for a in as_tuple(Fp)
        a isa AbstractVector && length(a) == markers.np || throw(
            DimensionMismatch("`$name` must be a vector with one entry per marker ($(markers.np))")
        )
    end
    check_backend(name, Fp, ka_backend(markers))
    check_precision(name, Fp, eltype(markers.coords[1]))
    return nothing
end

function check_marker_grid(xvi::Tuple, markers)
    length(xvi) == length(markers.coords) || throw(
        ArgumentError("the grid has $(length(xvi)) coordinate vectors but the markers are $(length(markers.coords))D")
    )
    check_markers_inside(markers.coords, xvi)
    return nothing
end

function check_marker_transfer(Fp, F, xvi, markers)
    check_no_mpi("PassiveMarkers")
    check_field_pairing("Fp", Fp, "F", F)
    check_marker_grid(xvi, markers)
    check_marker_field("Fp", Fp, markers)
    T = eltype(markers.coords[1])
    check_size("F", F, map(length, xvi))
    check_backend("F", F, ka_backend(markers))
    check_precision("F", F, T)
    check_distinct("Fp" => Fp, "F" => F)
    return nothing
end

# Infer one location per field, allowing a ghost layer independently on each axis.
function phase_injection_layout(F, centers::NTuple{N}, name) where {N}
    sz = size(F)
    ndims(F) == N || throw(DimensionMismatch("`$name` must be $(N)D"))
    for iscenter in (true, false)
        physical = centers .+ !iscenter
        if all(d -> sz[d] in (physical[d], physical[d] + 2), 1:N)
            all(>=(2), sz) || throw(DimensionMismatch("`$name` needs at least two samples per axis"))
            offset = ntuple(d -> Int(sz[d] == physical[d]), Val(N))
            return (; iscenter, offset)
        end
    end
    throw(DimensionMismatch("size(`$name`) = $sz; expected centers $centers or vertices $(centers .+ 1), optionally with two ghost samples per axis"))
end

check_phases(particles, ::Nothing) = nothing
function check_phases(particles, phases)
    check_cell_layout("particles_phases", phases, particles.index)
    check_backend("particles_phases", phases, ka_backend(particles))
    return nothing
end

function check_phase_injection_inputs(particles, phases, args, fields, grid)
    check_particle_fields(particles, args)
    check_phases(particles, phases)
    check_precision("args", args, scalar_eltype(particles.coords[1]))
    check_distinct("particles.coords" => particles.coords, "args" => args, "particles_phases" => phases)
    fields isa Tuple && length(fields) == length(args) || throw(
        ArgumentError("`fields` must be a tuple with one grid field per entry of `args`")
    )
    centers = inner_size(particles.index)
    return map(fields, ntuple(identity, length(fields))) do F, j
        layout = phase_injection_layout(F, centers, "fields[$j]")
        check_backend("fields[$j]", F, ka_backend(particles))
        check_precision("fields[$j]", F, scalar_eltype(particles.coords[1]))
        layout
    end
end
