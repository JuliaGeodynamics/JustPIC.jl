using JLD2
# using ImplicitGlobalGrid
using MPI: MPI

checkpoint_name(dst) = "$dst/particles_checkpoint.jld2"
checkpoint_name(dst, me) = "$dst/particles_checkpoint" * lpad("$(me)", 4, "0") * ".jld2"

function checkpointing_particles(
    dst,
    particles,
    phases,
    phase_ratios;
    chain=nothing,
    t=nothing,
    dt=nothing,
    particle_args=nothing,
)
    fname = checkpoint_name(dst)
    return checkpointing_particles(
        dst,
        particles,
        phases,
        phase_ratios,
        fname;
        chain=chain,
        t=t,
        dt=dt,
        particle_args=particle_args,
    )
end

function checkpointing_particles(
    dst,
    particles,
    phases,
    phase_ratios,
    me;
    chain=nothing,
    t=nothing,
    dt=nothing,
    particle_args=nothing,
)
    fname = checkpoint_name(dst, me)
    checkpointing_particles(
        dst,
        particles,
        phases,
        phase_ratios,
        fname;
        chain=chain,
        t=t,
        dt=dt,
        particle_args=particle_args,
    )
    return nothing
end

"""
    checkpointing_particles(dst, particles, phases, phase_ratios; chain=nothing, t=nothing, dt=nothing, particle_args=nothing)

Save the state of particles and related data to a checkpoint file in a jld2 format. The name of the checkpoint file is `particles_checkpoint.jld2`.


# Arguments
- `dst`: The destination directory where the checkpoint file will be saved.
- `particles`: The array of particles to be saved.
- `phases`: The array of phases associated with the particles.
- `phase_ratios`: The array of phase ratios.

## Keyword Arguments
- `chain`: The chain data to be saved. If nothing is stated, the default is `nothing`.
- `t`: The current time to be saved. If nothing is stated, the default is `nothing`.
- `dt`: The timestep to be saved. If nothing is stated, the default is `nothing`.
- `particle_args`: Additional particle arguments to be saved. If nothing is stated, the default is `nothing`.
"""
function checkpointing_particles(
    dst,
    particles,
    phases,
    phase_ratios,
    fname::String;
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
            :phases => Array(phases),
            :phase_ratios => Array(phase_ratios),
        )
        if !isnothing(chain)
            args[:chain] = Array(chain)
        end
        if !isnothing(t)
            args[:time] = t
        end
        if !isnothing(dt)
            args[:timestep] = dt
        end
        if !isnothing(particle_args)
            args[:particle_args] = Array.(particle_args)
        end

        jldsave(tmpfname; args...)

        # Move the checkpoint file from the temporary directory to the destination directory
        return mv(tmpfname, fname; force=true)
    end
end
