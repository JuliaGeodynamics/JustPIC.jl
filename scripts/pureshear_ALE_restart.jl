using Printf, Base.Threads, JLD2

using JustPIC
import KernelAbstractions: @kernel, @index
import CellArraysIndexing as CAI
const backend = JustPIC.CPU

@kernel function InitialFieldsParticles!(phases, px, py, index)
    I = @index(Global, NTuple)
    for ip in cellaxes(phases)
        # quick escape
        CAI.@index(index[ip, I...]) == 0 && continue
        x = CAI.@index px[ip, I...]
        y = CAI.@index py[ip, I...]
        CAI.@index phases[ip, I...] = x < y ? 1.0 : 2.0
    end
end

@kernel function SetVelocity(V, verts, ε̇bg)
    I = @index(Global, NTuple)
    if I[1] <= size(V.x, 1) &&  I[2] <= size(V.x, 2)
        V.x[I...] = verts.x[I[1]] * ε̇bg
    end
    if I[1] <= size(V.y, 1) &&  I[2] <= size(V.y, 2)
        V.y[I...] = -verts.y[I[2]] * ε̇bg
    end
end

velocity_ndrange(V) = map(max, size(V.x), size(V.y))

function update_particle_grid(particles, grid_vx, grid_vy)
    grid_vi = grid_vx, grid_vy
    xci = add_periodic_ghost_nodes.((grid_vy[1], grid_vx[2]))
    xvi = add_periodic_ghost_nodes.((grid_vx[1], grid_vy[2]))
    di = (
        center = map(x -> x[2] - x[1], xci),
        vertex = map(x -> x[2] - x[1], xvi),
        velocity = map(grids -> map(x -> x[2] - x[1], grids), grid_vi),
    )
    _di_values = map(values(di)) do spacings
        map(x -> x isa Tuple ? inv.(x) : inv(x), spacings)
    end
    _di = (; center = _di_values[1], vertex = _di_values[2], velocity = _di_values[3])

    return Particles(
        backend, particles.coords, particles.index, particles.nxcell,
        particles.max_xcell, particles.min_xcell, particles.np,
        di, _di, xci, xvi, grid_vi,
    )
end

function main(ALE, restart, last_step; do_plot = isinteractive())

    @printf("Running on %d thread(s)\n", nthreads())

    # Load checkpoint data
    if restart
        file = @sprintf("./Checkpoint%05d.jld2", last_step)
        data = load(file)
        particles = data["particles"]
        phases = data["phases"]
        xlims = data["xlims"]
        ylims = data["ylims"]
        t = haskey(data, "time") ? data["time"] : data["t"]
        Nt = last_step + 100
        it0 = last_step + 1
        ε̇bg = -1.0
    else
        xlims = [-0.5, 0.5]
        ylims = [-0.5, 0.5]
        t = 0.0
        Nt = 100
        it0 = 1
        ε̇bg = 1.0
    end

    # Parameters
    L = (x = (xlims[2] - xlims[1]), y = (ylims[2] - ylims[1]))
    Nc = (x = 41, y = 41)
    Nv = (x = Nc.x + 1, y = Nc.y + 1)
    Δ = (x = L.x / Nc.x, y = L.y / Nc.y)
    Nout = 10
    C = 0.25

    # Model extent
    cents_ext = (
        x = LinRange(xlims[1] - Δ.x / 2, xlims[2] + Δ.x / 2, Nc.x + 2),
        y = LinRange(ylims[1] - Δ.y / 2, ylims[2] + Δ.y / 2, Nc.y + 2),
    )
    verts = (
        x = LinRange(xlims[1], xlims[2], Nc.x + 1),
        y = LinRange(ylims[1], ylims[2], Nc.y + 1),
    )
    xlims = [verts.x[1], verts.x[end]]
    ylims = [verts.y[1], verts.y[end]]

    # Arrays
    size_x = (Nc.x + 1, Nc.y + 2)
    size_y = (Nc.x + 2, Nc.y + 1)
    V = (
        x = TA(backend)(zeros(size_x)),
        y = TA(backend)(zeros(size_y)),
    )

    # Set velocity field
    launch!(ka_backend(V.x), SetVelocity, velocity_ndrange(V), V, verts, ε̇bg)

    if !restart
        # Initialize particles -------------------------------
        nxcell, max_xcell, min_xcell = 12, 24, 5
        particles = init_particles(
            backend,
            nxcell,
            max_xcell,
            min_xcell,
            (verts.x, cents_ext.y),
            (cents_ext.x, verts.y),
        ) # random position by default

        # Initialise phase field
        particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

        launch!(
            ka_backend(particles), InitialFieldsParticles!, size(phases),
            phases, particles.coords..., particles.index
        )
    end

    particle_args = (phases,)
    grid_vx = verts.x, cents_ext.y
    grid_vy = cents_ext.x, verts.y
    particles = update_particle_grid(particles, grid_vx, grid_vy)

    phase_ratios = JustPIC.PhaseRatios(backend, 2, values(Nc))
    update_phase_ratios!(phase_ratios, particles, phases)
    Npart = sum(particles.index.data)


    # Time step
    Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
    @show Δt

    for it in it0:Nt

        @printf("Step %05d, #particles = %06d\n", it, Npart)

        t += Δt

        # advection!(particles, RungeKutta2(), values(V), Δt)
        # advection_LinP!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        advection_MQS!(particles, RungeKutta2(), values(V), Δt)

        if ALE
            xlims[1] += xlims[1] * ε̇bg * Δt
            xlims[2] += xlims[2] * ε̇bg * Δt
            ylims[1] -= ylims[1] * ε̇bg * Δt
            ylims[2] -= ylims[2] * ε̇bg * Δt
            L = (x = (xlims[2] - xlims[1]), y = (ylims[2] - ylims[1]))
            Δ = (x = L.x / Nc.x, y = L.y / Nc.y)
            cents_ext = (
                x = LinRange(xlims[1] - Δ.x / 2, xlims[2] + Δ.x / 2, Nc.x + 2),
                y = LinRange(ylims[1] - Δ.y / 2, ylims[2] + Δ.y / 2, Nc.y + 2),
            )
            verts = (
                x = LinRange(xlims[1], xlims[2], Nc.x + 1),
                y = LinRange(ylims[1], ylims[2], Nc.y + 1),
            )
            grid_vx = (verts.x, cents_ext.y)
            grid_vy = (cents_ext.x, verts.y)
            particles = update_particle_grid(particles, grid_vx, grid_vy)
            launch!(ka_backend(V.x), SetVelocity, velocity_ndrange(V), V, verts, ε̇bg)
            Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
        end
        move_particles!(particles, particle_args)
        inject_particles_phase!(particles, phases, (), ())
        update_phase_ratios!(phase_ratios, particles, phases)
        Npart = sum(particles.index.data)

        if mod(it, Nout) == 0 || it == 1
            @show
            particle_density = [sum(p) for p in particles.index]
            # visualisation
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = phases.data[:]
            idxv = particles.index.data[:]
            if do_plot
                @eval import CairoMakie
                f = CairoMakie.Figure()
                ax = CairoMakie.Axis(f[1, 1], title = "Particles", aspect = L.x / L.y, xlabel = "x", ylabel = "y")
                CairoMakie.scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]); color = Array(clr[idxv]), colormap = :roma, markersize = 2)
                CairoMakie.xlims!(ax, verts.x[1], verts.x[end])
                CairoMakie.ylims!(ax, verts.y[1], verts.y[end])
                display(f)
            end
        end

        # Save checkpoint
        if it == Nt
            @show file = @sprintf("./Checkpoint%05d.jld2", Nt)
            checkpointing_particles(pwd(), particles, file; phases, phase_ratios, t, dt = Δt, xlims, ylims)
        end
    end

    return nothing
end

###################################

ALE = get(ENV, "JUSTPIC_ALE", "true") == "true"
last_step = parse(Int, get(ENV, "JUSTPIC_RESTART_STEP", "100"))
restart = get(ENV, "JUSTPIC_RESTART", "false") == "true"

main(ALE, restart, last_step)
