using JLD2

checkpoint_name(dst) = joinpath(dst, "particles_checkpoint.jld2")
function checkpoint_name(dst, me)
    return joinpath(dst, "particles_checkpoint" * lpad("$(me)", 4, "0") * ".jld2")
end

function checkpointing_particles(
    dst,
    particles;
    phases=nothing,
    phase_ratios=nothing,
    chain=nothing,
    t=nothing,
    dt=nothing,
    particle_args=nothing,
)
    fname = checkpoint_name(dst)
    return checkpointing_particles(
        dst,
        particles,
        fname;
        phases=phases,
        phase_ratios=phase_ratios,
        chain=chain,
        t=t,
        dt=dt,
        particle_args=particle_args,
    )
end

function checkpointing_particles(
    dst,
    particles,
    me;
    phases=nothing,
    phase_ratios=nothing,
    chain=nothing,
    t=nothing,
    dt=nothing,
    particle_args=nothing,
)
    fname = checkpoint_name(dst, me)
    checkpointing_particles(
        dst,
        particles,
        fname;
        phases=phases,
        phase_ratios=phase_ratios,
        chain=chain,
        t=t,
        dt=dt,
        particle_args=particle_args,
    )
    return nothing
end

"""
    checkpointing_particles(dst, particles;phases=nothing, phase_ratios=nothing, chain=nothing, t=nothing, dt=nothing, particle_args=nothing)

Save the state of particles and related data to a checkpoint file in a jld2 format. The name of the checkpoint file is `particles_checkpoint.jld2`.


# Arguments
- `dst`: The destination directory where the checkpoint file will be saved.
- `particles`: The array of particles to be saved.

## Keyword Arguments
- `phases`: The array of phases associated with the particles. If nothing is stated, the default is `nothing`.
- `phase_ratios`: The array of phase ratios. If nothing is stated, the default is `nothing`.
- `chain`: The chain data to be saved. If nothing is stated, the default is `nothing`.
- `t`: The current time to be saved. If nothing is stated, the default is `nothing`.
- `dt`: The timestep to be saved. If nothing is stated, the default is `nothing`.
- `particle_args`: Additional particle arguments to be saved. If nothing is stated, the default is `nothing`.
"""
function checkpointing_particles(
    dst,
    particles,
    fname::String;
    phases=phases,
    phase_ratios=phase_ratios,
    chain=chain,
    t=t,
    dt=dt,
    particle_args=particle_args,
)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist

    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))

        # Prepare the arguments for jldsave
        args = Dict(
            :particles => Array(particles),
            :phases => isnothing(phases) ? nothing : Array(phases),
            :phase_ratios => isnothing(phase_ratios) ? nothing : Array(phase_ratios),
            :chain => isnothing(chain) ? nothing : Array(chain),
            :time => t,
            :timestep => dt,
            :particle_args => isnothing(particle_args) ? nothing : Array.(particle_args),
        )
        jldsave(tmpfname; args...)

        # Move the checkpoint file from the temporary directory to the destination directory
        return mv(tmpfname, fname; force=true)
    end
end
