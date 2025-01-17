using JLD2
# using ImplicitGlobalGrid
using MPI: MPI


checkpoint_name(dst) = "$dst/particles_checkpoint.jld2"
checkpoint_name(dst, me) = "$dst/partiles_checkpoint" * lpad("$(me)", 4, "0") * ".jld2"

function checkpointing_particles(dst, particles, phases, phase_ratios; chain=nothing, t=nothing, dt=nothing)
    fname = checkpoint_name(dst)
    checkpointing_particles(dst, particles, phases, phase_ratios, fname; chain=chain, t=t, dt=dt)
end

function checkpointing_particles(dst, particles, phases, phase_ratios, me; chain=nothing, t=nothing, dt=nothing)
    fname = checkpoint_name(dst, me)
    checkpointing_particles(dst, particles, phases, phase_ratios, fname; chain=chain, t=t, dt=dt)
    return nothing
end


function checkpointing_particles(dst, particles, phases, phase_ratios, fname::String; chain=chain, t=t, dt=dt)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist

    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))

        # Prepare the arguments for jldsave
        args = Dict(
            :particles => JustPIC._2D.Array(particles),
            :phases => JustPIC._2D.Array(phases),
            :phase_ratios => JustPIC._2D.Array(phase_ratios)
        )
        if !isnothing(chain)
            args[:chain] = JustPIC._2D.Array(chain)
        end
        if !isnothing(t)
            args[:time] = t
        end
        if !isnothing(dt)
            args[:timestep] = dt
        end

        jldsave(tmpfname; args...)

        # Move the checkpoint file from the temporary directory to the destination directory
        return mv(tmpfname, fname; force=true)
    end
end
