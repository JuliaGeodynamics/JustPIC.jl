using JLD2

checkpoint_name(dst) = joinpath(dst, "particles_checkpoint.jld2")
function checkpoint_name(dst, me)
    return joinpath(dst, "particles_checkpoint" * lpad("$(me)", 4, "0") * ".jld2")
end

function checkpointing_particles(
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
    fname = checkpoint_name(dst)
    return checkpointing_particles(
        dst,
        particles,
        fname;
        phases = phases,
        phase_ratios = phase_ratios,
        chain = chain,
        t = t,
        dt = dt,
        particle_args = particle_args,
        kwargs...,
    )

end

function checkpointing_particles(
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
    fname = checkpoint_name(dst, me)
    checkpointing_particles(
        dst,
        particles,
        fname;
        phases = phases,
        phase_ratios = phase_ratios,
        chain = chain,
        t = t,
        dt = dt,
        particle_args = particle_args,
        kwargs...,
    )
    return nothing
end

"""
    checkpointing_particles(dst, particles; phases=nothing, phase_ratios=nothing, chain=nothing, t=nothing, dt=nothing, particle_args=nothing)
    checkpointing_particles(dst, particles, me; phases=nothing, phase_ratios=nothing, chain=nothing, t=nothing, dt=nothing, particle_args=nothing)

Write particle state and optional companion data to a JLD2 checkpoint.

By default the file is saved as `particles_checkpoint.jld2` in `dst`. Additional
keyword arguments are serialized into the checkpoint after being converted to
plain Julia arrays where needed.

# Common keywords
- `phases`: per-particle phase labels.
- `phase_ratios`: `PhaseRatios` container to checkpoint.
- `chain`: marker-chain state.
- `t`: simulation time.
- `dt`: timestep size.
- `particle_args`: tuple of extra particle-carried fields.

# Notes
- Arrays are converted to plain Julia arrays before serialization so the
  checkpoint can be reloaded independently of the active backend.
- Passing `me` writes rank-local files named
  `particles_checkpoint0001.jld2`, `particles_checkpoint0002.jld2`, and so on.
"""
function checkpointing_particles(
        dst,
        particles,
        fname::String;
        phases = phases,
        phase_ratios = phase_ratios,
        chain = chain,
        t = t,
        dt = dt,
        particle_args = particle_args,
        kwargs...,
    )
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist

    return mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))

        # Build args dict dynamically
        args = Dict(
            :particles => Array(particles),
            :phases => isnothing(phases) ? nothing : Array(phases),
            :phase_ratios => isnothing(phase_ratios) ? nothing : Array(phase_ratios),
            :chain => isnothing(chain) ? nothing : Array(chain),
            :time => t,
            :timestep => dt,
            :particle_args => isnothing(particle_args) ? nothing : Array.(particle_args),
        )

        # Add any additional kwargs dynamically using their names as keys
        for (key, value) in pairs(kwargs)
            args[key] = isnothing(value) ? nothing :
                isa(value, AbstractArray) ? Array(value) :
                isa(value, Tuple) ? Array.(value) : value
        end

        try
            jldsave(tmpfname; args...)
        catch
            jldsave(tmpfname, IOStream; args...)
        end

        # Move the checkpoint file from the temporary directory to the destination directory
        return mv(tmpfname, fname; force = true)
    end
end
