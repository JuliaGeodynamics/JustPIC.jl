using JLD2

# Version of the checkpoint payload layout. Increment it whenever a stored key
# changes meaning, and teach `load_checkpoint` to read every older version.
checkpoint_schema_version() = 1

checkpoint_name(dst) = joinpath(dst, "particles_checkpoint.jld2")
function checkpoint_name(dst, me)
    return joinpath(dst, "particles_checkpoint" * lpad("$(me)", 4, "0") * ".jld2")
end

"""
    checkpointing_particles(dst, particles; phases=nothing, phase_ratios=nothing, chain=nothing, t=nothing, dt=nothing, particle_args=nothing, kwargs...)
    checkpointing_particles(dst, particles, me::Integer; kwargs...)
    checkpointing_particles(dst, particles, fname::AbstractString; kwargs...)

Write particle state and optional companion data to a JLD2 checkpoint.

By default the file is saved as `particles_checkpoint.jld2` in `dst`. Passing
`me` writes rank-local files named after the zero-based MPI rank:
`particles_checkpoint0000.jld2`, `particles_checkpoint0001.jld2`, and so on.
Passing `fname` writes to that path instead.

# Common keywords
- `phases`: per-particle phase labels.
- `phase_ratios`: `PhaseRatios` container to checkpoint.
- `chain`: marker-chain state.
- `t`: simulation time, stored as `"time"`.
- `dt`: timestep size, stored as `"timestep"`.
- `particle_args`: tuple of extra particle-carried fields.

Additional keyword arguments are stored under their own names.

# Payload
Every value is converted to host memory before writing: arrays and
`CellArray`s become CPU arrays, JustPIC containers are converted with `Array`,
and `Tuple`s and `NamedTuple`s are converted element by element. Numbers,
strings, symbols and `nothing` are stored as they are. Any other value throws
an `ArgumentError` before the file is touched. The keys `checkpoint_schema`,
`justpic_version` and `partition` hold metadata that [`load_checkpoint`](@ref)
checks, and cannot be used as keyword names; neither can `time` and `timestep`.

# Replacement
The checkpoint is written to a temporary file in the destination directory and
then renamed over `fname`. On filesystems where a rename within one directory
is atomic (POSIX filesystems), a failed or interrupted write leaves the previous
checkpoint intact.
"""
function checkpointing_particles(dst, particles; kwargs...)
    return _write_checkpoint(dst, checkpoint_name(dst), nothing, particles; kwargs...)
end

function checkpointing_particles(dst, particles, me::Integer; kwargs...)
    _write_checkpoint(dst, checkpoint_name(dst, me), me, particles; kwargs...)
    return nothing
end

function checkpointing_particles(dst, particles, fname::AbstractString; kwargs...)
    return _write_checkpoint(dst, fname, nothing, particles; kwargs...)
end

function _write_checkpoint(
        dst,
        fname,
        me,
        particles;
        phases = nothing,
        phase_ratios = nothing,
        chain = nothing,
        t = nothing,
        dt = nothing,
        particle_args = nothing,
        kwargs...,
    )
    for key in keys(kwargs)
        key in (:checkpoint_schema, :justpic_version, :partition, :time, :timestep) &&
            throw(ArgumentError("`$key` is a reserved checkpoint key and cannot be passed as a keyword"))
    end

    args = Dict{Symbol, Any}(
        :checkpoint_schema => checkpoint_schema_version(),
        :justpic_version => string(pkgversion(@__MODULE__)),
        :partition => _checkpoint_partition(me),
        :particles => _checkpoint_value(particles),
        :phases => _checkpoint_value(phases),
        :phase_ratios => _checkpoint_value(phase_ratios),
        :chain => _checkpoint_value(chain),
        :time => _checkpoint_value(t),
        :timestep => _checkpoint_value(dt),
        :particle_args => _checkpoint_value(particle_args),
    )
    for (key, value) in pairs(kwargs)
        args[key] = _checkpoint_value(value)
    end

    !isdir(dst) && mkpath(dst) # create folder in case it does not exist
    return _replace_atomically(fname) do tmpfname
        try
            jldsave(tmpfname; args...)
        catch
            jldsave(tmpfname, IOStream; args...)
        end
    end
end

# Write through `write(tmpfname)` into a temporary file next to `fname`, then
# rename it over `fname`. The temporary file must live in the destination
# directory: a rename is only atomic within one filesystem.
function _replace_atomically(write, fname)
    tmpfname = tempname(dirname(abspath(fname)); cleanup = false)
    try
        write(tmpfname)
        Base.Filesystem.rename(tmpfname, fname)
    catch
        rm(tmpfname; force = true)
        rethrow()
    end
    return fname
end

_checkpoint_value(x::Union{Nothing, Number, AbstractString, Symbol}) = x
_checkpoint_value(x::AbstractArray) = to_cpu(x)
_checkpoint_value(x::AbstractParticles) = Array(x)
_checkpoint_value(x::Union{Tuple, NamedTuple}) = map(_checkpoint_value, x)
function _checkpoint_value(x)
    throw(
        ArgumentError(
            "cannot checkpoint a value of type $(typeof(x)); supported values are " *
                "arrays, JustPIC containers, numbers, strings, symbols, `nothing`, " *
                "and tuples or named tuples of these"
        )
    )
end

function _checkpoint_partition(me)
    isnothing(me) && return nothing
    ImplicitGlobalGrid.grid_is_initialized() ||
        return (; rank = me, nprocs = nothing, dims = nothing, coords = nothing, nxyz_g = nothing)
    gg = ImplicitGlobalGrid.get_global_grid()
    return (;
        rank = me,
        nprocs = gg.nprocs,
        dims = Tuple(gg.dims),
        coords = Tuple(gg.coords),
        nxyz_g = Tuple(gg.nxyz_g),
    )
end

"""
    load_checkpoint(fname::AbstractString)
    load_checkpoint(dst, me::Integer)

Read a checkpoint written by [`checkpointing_particles`](@ref) and return a
`Dict{String, Any}` with the same keys that `JLD2.load` returns. Stored arrays
and containers are on the CPU; convert them to the active backend with
`TA(backend)`.

The second form reads the rank-local file of rank `me` in `dst`.

# Compatibility
- Checkpoints whose schema version is newer than this JustPIC supports are
  rejected with an `ArgumentError`.
- Checkpoints without schema metadata (written before schema versioning) load
  with a warning, without compatibility checks.
- The rank-local form checks that the file was written by rank `me`. When
  `ImplicitGlobalGrid` is initialized and the file records a topology, the
  number of ranks, the process grid, the rank's Cartesian coordinates and the
  global grid size must match the current ones.
"""
function load_checkpoint(fname::AbstractString)
    isfile(fname) || throw(ArgumentError("checkpoint file `$fname` does not exist"))
    data = JLD2.load(fname)
    schema = get(data, "checkpoint_schema", nothing)
    if isnothing(schema)
        @warn "Checkpoint `$fname` has no schema version; it predates versioned checkpoints and is loaded without compatibility checks"
    elseif schema > checkpoint_schema_version()
        throw(
            ArgumentError(
                "checkpoint `$fname` uses schema version $schema (written by JustPIC " *
                    "$(get(data, "justpic_version", "unknown"))), but JustPIC " *
                    "$(pkgversion(@__MODULE__)) reads schema versions up to " *
                    "$(checkpoint_schema_version())"
            )
        )
    end
    return data
end

function load_checkpoint(dst, me::Integer)
    fname = checkpoint_name(dst, me)
    data = load_checkpoint(fname)
    haskey(data, "checkpoint_schema") || return data

    stored = data["partition"]
    isnothing(stored) &&
        throw(ArgumentError("checkpoint `$fname` was not written as a rank-local checkpoint"))
    stored.rank == me ||
        throw(ArgumentError("checkpoint `$fname` was written by rank $(stored.rank), not rank $me"))
    if ImplicitGlobalGrid.grid_is_initialized() && !isnothing(stored.nprocs)
        current = _checkpoint_partition(me)
        topology(p) = (; p.nprocs, p.dims, p.coords, p.nxyz_g)
        topology(stored) == topology(current) || throw(
            ArgumentError(
                "checkpoint `$fname` was written with MPI topology $(topology(stored)), " *
                    "but the current topology is $(topology(current))"
            )
        )
    end
    return data
end
