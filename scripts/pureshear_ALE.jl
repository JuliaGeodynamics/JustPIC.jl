using Statistics, LinearAlgebra, Printf, Base.Threads, CairoMakie
const year     = 365*3600*24
const USE_GPU  = false
const GPU_ID   = 0
const USE_MPI  = false

using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend 

Visu   = true
ThreeD = false
Save   = false
ALE    = false

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 3)
end

@parallel_indices (I...) function InitialFieldsParticles!( phases, px, py, index)
    @inbounds for ip in cellaxes(phases)
        # quick escape
        @cell(index[ip, I...]) == 0 && continue
        x = @cell px[ip, I...]
        y = @cell py[ip, I...]
        if x<y
            @cell phases[ip, I...] = 1.0
        else
            @cell phases[ip, I...] = 2.0
        end
    end
    return nothing
end

function main()

    @printf("Running on %d thread(s)\n", nthreads())
    L  = (x=1., y=1.)
    Nc = (x=40, y=40 )
    Nv = (x=Nc.x+1,   y=Nc.y+1   )
    Δ  = (x=L.x/Nc.x, y=L.y/Nc.y )
    Nt   = 100
    Nout = 10
    C    = 0.25

    verts     = (x=LinRange(-L.x/2, L.x/2, Nv.x), y=LinRange(-L.y/2, L.y/2, Nv.y))
    cents_ext = (x=LinRange(-Δ.x/2-L.x/2, L.x/2+Δ.x/2, Nc.x+2), y=LinRange(-Δ.y/2-L.y/2, L.y+Δ.y/2+L.y/2, Nc.y+2))
    cents     = (x=LinRange(+Δ.x/2-L.x/2, L.x/2-Δ.x/2, Nc.x  ), y=LinRange(+Δ.y/2-L.y/2, L.y-Δ.y/2+L.y/2, Nc.y  ))

    size_x = (Nc.x+1, Nc.y+2)
    size_y = (Nc.x+2, Nc.y+1)

    V = (
        x      = @zeros(size_x),
        y      = @zeros(size_y),
    )

    # Set velocity field
    ε̇bg = 1.0
    for i=1:size(V.x,1),  j=1:size(V.x,2)
        V.x[i,j] = -verts.x[i]*ε̇bg
    end

    for i=1:size(V.y,1),  j=1:size(V.y,2)
        V.y[i,j] =  verts.y[j]*ε̇bg
    end

 
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 50, 5
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

    # Time step
    Δt = C * min(Δ...) / max(maximum(abs.(V.x)), maximum(abs.(V.y)))
    @show Δt

    # Create necessary tuples
    grid_vx = (verts.x, cents_ext.y)
    grid_vy = (cents_ext.x, verts.y)
    grid_vz = (cents_ext.x, cents_ext.y)
    Vxc     = 0.5*(V.x[1:end-1,2:end-1] .+ V.x[2:end-0,2:end-1])
    Vyc     = 0.5*(V.y[2:end-1,1:end-1] .+ V.y[2:end-1,2:end-0])
    Vmag    = sqrt.(Vxc.^2 .+ Vyc.^2)

    for it=1:Nt

        # advection!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        # advection_LinP!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        advection_MQS!(particles, RungeKutta2(), values(V), (grid_vx, grid_vy), Δt)
        move_particles!(particles, values(verts), particle_args)        
        # inject_particles!(particles, particle_args, values(verts)) 

        if mod(it,Nout) == 0 || it==1

            @show Npart = sum(particles.index.data)
            particle_density = [sum(p) for p in particles.index]
            @show size(particle_density)

            # Plots
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = phases.data[:]
            idxv = particles.index.data[:]
            f = Figure()
            f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=2)
            xlims!(ax, verts.x[1], verts.x[end])
            ylims!(ax, verts.y[1], verts.y[end])
            display(f)
        end
    end
    return nothing
end 

main()
