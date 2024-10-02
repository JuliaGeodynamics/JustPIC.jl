# Checkpointing

## Writing checkpoint files

It is customary to employ [checkpointing](https://en.wikipedia.org/wiki/Application_checkpointing) during simulation that involve many time steps. A checkpoint file then needs to be written to disk. Such file allows for restarting a simulation from last checkpoint file written to disk.
Moreover, checkpoint files may occupy a lost of disk space. Here is how to write essential particle information in a checkpoint file in [jld2 format](https://github.com/JuliaIO/JLD2.jl):

```julia
jldsave(
    "my_file.jld2"; 
    particles     = Array(particles), 
    phases        = Array(phases), 
    phase_ratios  = Array(phase_ratios), 
    particle_args = Array.(particle_args),
)
```
This will save particle information to the file `my_file.jld2`, which can be reused in order to restart a simulation.

If file size are huge, on may cast all the fields from particle structures into `Float32`. While this will spare disk space, it may hinder the reproducibility at restart. 

```julia
jldsave(
    "my_file.jld2"; 
    particles     = Array(Float32, particles), 
    phases        = Array(Float32, phases), 
    phase_ratios  = Array(Float32, phase_ratios), 
    particle_args = Array.(Float32, particle_args),
)
```

## Loading a checkpoint file

In order to restart a simulation, one needs to load the checkpoint file of interest. This is how to read the particle information from the checkpoint file `my_file.jld2`:

```julia
    data          = load("my_file.jld2")
    particles     = TA(backend)(Float64, data["particles"])
    phases        = TA(backend)(Float64, data["phases"])
    phase_ratios  = TA(backend)(Float64, data["phase_ratios"])
    particle_args = TA(backend).(Float64, data["particle_args"])
```
The function `TA(backend)` will automatically cast the data to the appropriate type, depending on the requested backend.  
