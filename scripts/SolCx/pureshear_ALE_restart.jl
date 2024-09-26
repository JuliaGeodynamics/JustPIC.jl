using Statistics, LinearAlgebra, Printf, Base.Threads, CairoMakie, JLD2
const year     = 365*3600*24
const USE_GPU  = false

using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend 

const ALE    = true
last_step    = 100
restart      = false

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel_indices (I...) function InitialFieldsParticles!(phases, px, py, index)
    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue
        x = @index px[ip, I...]
        y = @index py[ip, I...]
        if x<y
            @index phases[ip, I...] = 1.0
        else
            @index phases[ip, I...] = 2.0
        end
    end
    return nothing
end

@parallel_indices (I...) function SetVelocity(V, verts, ε̇bg)
    if I[1]<=size(V.x,1) &&  I[2]<=size(V.x,2)
        V.x[I...] =  verts.x[I[1]]*ε̇bg
    end
    if I[1]<=size(V.y,1) &&  I[2]<=size(V.y,2)
        V.y[I...] = -verts.y[I[2]]*ε̇bg
    end
return nothing
end


function main()

    @printf("Running on %d thread(s)\n", nthreads())

    # Load Breakpoint data
    if restart
        file          = @sprintf("./Breakpoint%05d.jld2", last_step)
        data          = load(file)
        particles     = data["particles"]
        phases        = data["phases"]   
        phase_ratios  = data["phase_ratios"]
        particle_args = data["particle_args"]
        xlims         = data["xlims"]
        ylims         = data["ylims"]
        t             = data["t"]
        Nt            = last_step + 100
        it0           = last_step + 1
        ε̇bg           = -1.
    else
        xlims = [-0.5 , 0.5]
        ylims = [-0.5 , 0.5]
        t     = 0.
        Nt    = 100
        it0   = 1
        ε̇bg   = 1.
    end

    # Parameters
    L    = (x =(xlims[2]-xlims[1]), y =(ylims[2]-ylims[1]) )
    Nc   = (x=41, y=41 )
    Nv   = (x=Nc.x+1,   y=Nc.y+1   )
    Δ    = (x=L.x/Nc.x, y=L.y/Nc.y )
    Nout = 10
    C    = 0.25

    # Model extent
    cents_ext = (
        x      = LinRange(xlims[1]-Δ.x/2, xlims[2]+Δ.x/2, Nc.x+2),
        y      = LinRange(ylims[1]-Δ.y/2, ylims[2]+Δ.y/2, Nc.y+2),
    )
    cents     = (
        x      = LinRange(xlims[1]+Δ.x/2, xlims[2]-Δ.x/2, Nc.x+2),
        y      = LinRange(ylims[1]+Δ.y/2, ylims[2]-Δ.y/2, Nc.y+2),
    )
    verts = (
        x      = LinRange(xlims[1], xlims[2], Nc.x+1),
        y      = LinRange(ylims[1], ylims[2], Nc.y+1),
    )
    xlims     = [verts.x[1] , verts.x[end]]
    ylims     = [verts.y[1] , verts.y[end]]

    # Arrays
    size_x    = (Nc.x+1, Nc.y+2)
    size_y    = (Nc.x+2, Nc.y+1)
    V = (
        x = @zeros(size_x),
        y = @zeros(size_y),
    )

    # Set velocity field
    @parallel SetVelocity(V, verts, ε̇bg)
 
    if !restart 
        # Initialize particles -------------------------------
        nxcell, max_xcell, min_xcell = 12, 24, 5
        particles = init_particles(
            backend, 
            nxcell, 
            max_xcell,
            min_xcell, 
            values(verts),
            values(Δ),
            values(Nc)
        ) # random position by default

        # Initialise phase field
        particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

        @parallel InitialFieldsParticles!(phases, particles.coords..., particles.index)
    end

    phase_ratios = JustPIC._2D.PhaseRatios(backend, 2, values(Nc));
    phase_ratios_vertex!(phase_ratios, particles, values(verts), phases) 
    phase_ratios_center!(phase_ratios, particles, values(cents), phases) 
    Npart = sum(particles.index.data)  


    # Time step
    Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
    @show Δt

    # Create necessary tuples
    grid_vx = (verts.x, cents_ext.y)
    grid_vy = (cents_ext.x, verts.y)
    Vxc     = 0.5*(V.x[1:end-1,2:end-1] .+ V.x[2:end-0,2:end-1])
    Vyc     = 0.5*(V.y[2:end-1,1:end-1] .+ V.y[2:end-1,2:end-0])
    Vmag    = sqrt.(Vxc.^2 .+ Vyc.^2)

    for it=it0:Nt

        @printf("Step %05d, #particles = %06d\n", it, Npart)

        t += Δt

        # advection!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        # advection_LinP!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        advection_MQS!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt);
        move_particles!(particles, values(verts), particle_args);
        inject_particles_phase!(particles, phases, (), (), values(verts))
        phase_ratios_vertex!(phase_ratios, particles, values(verts), phases);
        phase_ratios_center!(phase_ratios, particles, values(cents), phases);  
        Npart = sum(particles.index.data)  
  
        if ALE
            xlims[1] += xlims[1]*ε̇bg*Δt 
            xlims[2] += xlims[2]*ε̇bg*Δt
            ylims[1] -= ylims[1]*ε̇bg*Δt 
            ylims[2] -= ylims[2]*ε̇bg*Δt
            L  = ( x =(xlims[2]-xlims[1]), y =(ylims[2]-ylims[1]) )  
            Δ  = (x=L.x/Nc.x, y=L.y/Nc.y )
            cents_ext = (
                x      = LinRange(xlims[1]-Δ.x/2, xlims[2]+Δ.x/2, Nc.x+2),
                y      = LinRange(ylims[1]-Δ.y/2, ylims[2]+Δ.y/2, Nc.y+2),
            )
            verts = (
                x      = LinRange(xlims[1], xlims[2], Nc.x+1),
                y      = LinRange(ylims[1], ylims[2], Nc.y+1),
            )
            grid_vx = (verts.x, cents_ext.y)
            grid_vy = (cents_ext.x, verts.y)
            Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
            @parallel SetVelocity(V, verts, ε̇bg)
            move_particles!(particles, values(verts), (phases,)) 
            phase_ratios_vertex!(phase_ratios, particles, values(verts), phases)
        end
        
        if mod(it, Nout) == 0 || it==1
            @show
            particle_density = [sum(p) for p in particles.index]
            # visualisation
            p    = particles.coords
            ppx, ppy = p
            pxv  = ppx.data[:]
            pyv  = ppy.data[:]
            clr  = phases.data[:]
            idxv = particles.index.data[:]
            f = Figure()
            ax = Axis(f[1, 1], title="Particles", aspect=L.x/L.y)
            scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=2)
            xlims!(ax, verts.x[1], verts.x[end])
            ylims!(ax, verts.y[1], verts.y[end])
            display(f)
        end       

        # Save breakpoint
        if it==100
            @show file = @sprintf("./Breakpoint%05d.jld2", Nt)
            jldsave(file; particles, phases, phase_ratios, particle_args, xlims, ylims, t)
        end
    end

    return nothing
end 

main()