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
    phases        = Array(phases),
    phase_ratios  = Array(phase_ratios),
    particle_args = Array.(particle_args),
)
```
This saves particle information to `my_file.jld2`, ready to be reloaded later.

If file size matters more than exact restart reproducibility, you can downcast to
`Float32` before writing:

```julia
jldsave(
    "my_file.jld2";
    particles     = Array(Float32, particles),
    phases        = Array(Float32, phases),
    phase_ratios  = Array(Float32, phase_ratios),
    particle_args = Array.(Float32, particle_args),
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
Any additional keyword arguments are written into the checkpoint as extra fields.
## Loading a checkpoint file

To restart a simulation, load the file and cast the stored arrays back to the
active backend:

```julia
data          = load("my_file.jld2")
particles     = TA(backend)(Float64, data["particles"])
phases        = TA(backend)(Float64, data["phases"])
phase_ratios  = TA(backend)(Float64, data["phase_ratios"])
particle_args = TA(backend).(Float64, data["particle_args"])
```
`TA(backend)` selects the backend-appropriate array type, so the same checkpoint
can be restored onto CPU or accelerator arrays.
