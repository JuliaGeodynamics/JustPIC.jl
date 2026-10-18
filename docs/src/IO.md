# Checkpointing

## Writing checkpoint files

Long-running simulations commonly write checkpoint files so that runs can be
restarted from a recent saved state. In JustPIC, checkpoints are written in
[JLD2 format](https://github.com/JuliaIO/JLD2.jl).

At the lowest level, you can serialize arrays manually:

```julia
jldsave(
    "my_file.jld2";
    particles     = Array(particles),
    phases        = to_cpu(phases),
    phase_ratios  = Array(phase_ratios),
    particle_args = to_cpu.(particle_args),
)
```
This saves particle information to `my_file.jld2`, ready to be reloaded later.

If file size matters more than exact restart reproducibility, you can downcast to
`Float32` before writing:

```julia
jldsave(
    "my_file.jld2";
    particles     = Array(Float32, particles),
    phases        = to_cpu(Float32, phases),
    phase_ratios  = Array(Float32, phase_ratios),
    particle_args = to_cpu.(Float32, particle_args),
)
```

For routine use, prefer the built-in helper:

```julia
checkpointing_particles(
    dst,
    particles;
    phases = nothing,
    phase_ratios = nothing,
    chain = nothing,
    t = nothing,
    dt = nothing,
    particle_args = nothing,
    kwargs...,
)
```
On multiple MPI ranks, pass the rank id to get rank-local filenames:
```julia
checkpointing_particles(
    dst,
    particles,
    me;
    phases = nothing,
    phase_ratios = nothing,
    chain = nothing,
    t = nothing,
    dt = nothing,
    particle_args = nothing,
    kwargs...,
)
```
Any additional keyword arguments are written into the checkpoint as extra
fields. Arrays, `CellArray`s and JustPIC containers are moved to host memory,
tuples and named tuples are converted element by element, and numbers, strings,
symbols and `nothing` are stored as they are. Any other value is rejected with
an `ArgumentError` before the checkpoint file is touched.

The checkpoint is first written to a temporary file in the destination
directory and then renamed over the previous checkpoint. On POSIX filesystems
this replacement is atomic: a write that fails or is interrupted leaves the
previous checkpoint intact.

## Loading a checkpoint file

To restart a simulation, load the file with `load_checkpoint` and cast the
stored arrays back to the active backend:

```julia
data          = load_checkpoint("my_file.jld2")
particles     = TA(backend)(Float64, data["particles"])
phases        = TA(backend)(Float64, data["phases"])
phase_ratios  = TA(backend)(Float64, data["phase_ratios"])
particle_args = TA(backend).(Float64, data["particle_args"])
```
`TA(backend)` selects the backend-appropriate array type, so the same checkpoint
can be restored onto CPU or accelerator arrays. On multiple MPI ranks, each rank
reads its own file with `load_checkpoint(dst, me)`.

## Compatibility

Every checkpoint records a schema version (`"checkpoint_schema"`), the JustPIC
version that wrote it (`"justpic_version"`) and, for rank-local files, the MPI
partition (`"partition"`: rank, number of ranks, process grid, Cartesian
coordinates and global grid size). `load_checkpoint` enforces the following:

- A checkpoint with a newer schema version than the running JustPIC supports is
  rejected with an error.
- A checkpoint without a schema version, written by JustPIC 0.7 or earlier,
  loads with a warning and without further checks.
- A rank-local checkpoint must have been written by the same rank. When
  ImplicitGlobalGrid is initialized, the number of ranks, process grid,
  Cartesian coordinates and global grid size must match the ones recorded in
  the file: restarting on a different decomposition is not supported.

Checkpoints remain plain JLD2 files, so `JLD2.load` still reads them without
these checks.

## API

```@docs
checkpointing_particles
load_checkpoint
```
