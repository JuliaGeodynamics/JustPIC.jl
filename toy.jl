using JustPIC, ParallelStencil, StaticArrays, LinearAlgebra, ParallelStencil, TimerOutputs
using GLMakie

function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @rand(ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index, x, y, dx, dy, nxcell, max_xcell)
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:max_xcell
            if l <= nxcell
                @cell px[l, i, j] = x0 + dx * (@cell(px[l, i, j]) * 0.9 + 0.05)
                @cell py[l, i, j] = y0 + dy * (@cell(py[l, i, j]) * 0.9 + 0.05)
                @cell index[l, i, j] = true
            
            else
                @cell px[l, i, j] = NaN
                @cell py[l, i, j] = NaN
                
            end
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index, x, y, dx, dy, nxcell, max_xcell) 

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

function init_particles_classic(nxcell, x, y, dx, dy, nx, ny)
    ncells = nx * ny
    np = nxcell * ncells
    pcoords = TA([SA[(rand()*0.9 + 0.05)*dx, (rand()*0.9 + 0.05)*dy] for _ in 1:np])
    parent_cell = TA([(0, 0) for _ in 1:np])
    
    @parallel_indices (i, j) function fill_coords_index(pcoords, parent_cell, x, y, nxcell)
        nx, ny = length(x)-1, length(y)-1
        k = LinearIndices((1:nx, 1:ny))[i, j]
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in (1 + nxcell * (k-1)):(nxcell * k)
            pcoords[l] = @SVector [
                x0 + pcoords[l][1],
                y0 + pcoords[l][2]
            ]
            parent_cell[l] = (i, j)
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(pcoords, parent_cell, x, y, nxcell)

    return ClassicParticles(
        pcoords, parent_cell, np, (dx, dy)
    )
end

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function advection_test_2D()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 48, 1
    n = 256
    nx = ny = n-1
    ni = nx , ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles0 = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )
    particles = init_particles_classic(nxcell, xvi..., dx, dy, nx, ny)

    # Cell fields -------------------------------
    Vx = TA([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA([y for x in xv, y in yv]);
    T0 = deepcopy(T)
    buffer = similar(T)
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2;
    particle_args = pT0, = init_cell_arrays(particles0, Val(1));
    pT = zeros(length(particles.coords))
    # x_copy = copy(particles.coords)

    niter = 50
    # time_opt = zeros(niter)
    # time_classic = zeros(niter)
    # time_adv_opt = zeros(niter)
    # time_shuffle = zeros(niter)
    # time_adv_classic = zeros(niter)
    # time_g2p_opt = zeros(niter)
    # time_p2g_opt = zeros(niter)
    # time_g2p_classic = zeros(niter)
    # time_p2g_classic = zeros(niter)

    # for it in 1:niter
    #     time_opt[it] = @elapsed begin
    #         time_adv_opt[it] = @elapsed advection_RK!(particles0, V, grid_vx, grid_vy, dt, 2 / 3)
    #         time_shuffle[it] = @elapsed shuffle_particles!(particles0, xvi, particle_args)
    #         time_g2p_opt[it] = @elapsed grid2particle!(pT0, xvi, T, particles0.coords, particles0.index)
    #         time_p2g_opt[it] = @elapsed particle2grid!(T, pT0, xvi, particles0.coords, particles0.index)
    #     end
        
    #     time_classic[it] = @elapsed begin
    #         time_adv_classic[it] = @elapsed advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
    #         time_g2p_classic[it] = @elapsed grid2particle_naive!(pT, xvi, T, particles) 
    #         time_p2g_classic[it] = @elapsed particle2grid_naive!(T, pT, buffer, xvi, particles)
    #     end
    # end

    # timings = (
    #     opt = time_opt,
    #     classic = time_classic,
    #     adv_opt = time_adv_opt,
    #     shuffle = time_shuffle,
    #     adv_classic = time_adv_classic,
    #     g2p_opt = time_g2p_opt,
    #     p2g_opt = time_p2g_opt,
    #     g2p_classic = time_g2p_classic,
    #     p2g_classic = time_p2g_classic,
    # )
    # return timings

    np_tot = sum(sum(p) for p in particles0.index)
    println("Number of particles: ", np_tot)
    niter = 2000

    to = TimerOutput()
    t = 0.0
    # for it in 1:niter
    while t < 0.15
        @timeit to "opt" begin
            @timeit to "adv_opt"     advection_RK!(particles0, V, grid_vx, grid_vy, dt, 2 / 3)
            @timeit to "shuffle"     shuffle_particles!(particles0, xvi, particle_args)
            @timeit to "p2g_opt"     particle2grid!(T, pT0, xvi, particles0.coords, particles0.index)
            @timeit to "g2p_opt"     grid2particle!(pT0, xvi, T, particles0.coords, particles0.index)
        end
        @timeit to "classic" begin
            @timeit to "adv_classic" advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
            @timeit to "p2g_classic" particle2grid_naive!(T, pT, buffer, xvi, particles)
            @timeit to "g2p_classic" grid2particle_naive!(pT, xvi, T, particles)
        end

        # np = [sum(p) for p in particles0.index]
        # maximum(np) == max_xcell && @show it
        # np_tot = sum(np)
        # println("Number of particles at iteration $it: ", np_tot)
        t += dt
    end
    @show t
    to 
end

timings = advection_test_2D()

# scatter(timings.opt[2:end], color=:black)
# scatter!(timings.classic[2:end], color=:red)

# to = advection_test_2D()

# scatter(time_adv_opt.+time_shuffle, color=:black)
# scatter!(time_adv_classic, color=:red)

# scatter(time_g2p_opt, color=:black)
# scatter!(time_g2p_classic, color=:red)
# scatter(time_g2p_opt./time_g2p_classic)

# initial_indices = [ LinearIndices((1:nx, 1:ny))[get_cell(x, dxi)...] for x in x_copy] 
# final_indices = [ LinearIndices((1:nx, 1:ny))[get_cell(x, dxi)...] for x in particles.coords]

# distance = abs.(final_indices .- initial_indices)

# f,ax,h = scatter(x_copy, color=distance)
# Colorbar(f[1,2], h)

# @edit grid2particle!(pT0, xvi, T, particles0.coords, particles0.index)

@btime grid2particle!($(pT0, xvi, T, particles0.coords, particles0.index)...)
@btime grid2particle_naive!($(pT, xvi, T, particles)...)

ProfileCanvas.@profview for i in 1:10
    shuffle_particles!(particles0, xvi, particle_args)
end

ProfileCanvas.@profview shuffle_particles!(particles0, xvi, particle_args)

ProfileCanvas.@profview for i in 1:50
    grid2particle_naive!(pT, xvi, T, particles)
end

@btime @cell $pT[1, 1, 1] = $1.0;

@edit @cell pT[1, 1, 1] = 1.0;

JustPIC.setelement!(pT, 10.0, 1, 1, 1)

# function get_particle_coords(p::NTuple{N,T}, ip, idx::Vararg{Int64, N}) where {N,T} 
#     ntuple(Val(N)) do i 
#         Base.@_inline_meta
#         @inbounds @cell p[i][ip, idx...]
#     end
# end

# @btime get_particle_coords($(particles0.coords, 1, 1, 1)...)


domain_limits = extrema.(xvi)
parent_cell = 5, 5
corner_xi = xvi[1][parent_cell[1]], xvi[2][parent_cell[2]]
idx_loop = 1, 1

function foo!(
    particle_coords,
    domain_limits,
    corner_xi,
    dxi,
    nxi,
    index,
    parent_cell::NTuple{N1,Int64},
    args::NTuple{N2,T},
    idx_loop::NTuple{N1,Int64},
) where {N1,N2,T}
    idx_child = JustPIC.child_index(parent_cell, idx_loop)

    if JustPIC.indomain(idx_child, nxi)

        # iterate over particles in child cell 
        for ip in JustPIC.cellaxes(index)
            @inbounds !@cell(index[ip, idx_child...]) && continue

            p_child = JustPIC.cache_particle(particle_coords, ip, idx_child)

            # particle went of of the domain, get rid of it
            # if !(JustPIC.indomain(p_child, domain_limits))
            #     @cell index[ip, idx_child...] = false
            #     JustPIC.empty_particle!(particle_coords, ip, idx_child)
            #     JustPIC.empty_particle!(args, ip, idx_child)
            # end

            if @inbounds @cell index[ip, idx_child...] # true if memory allocation is filled with a particle
                # check whether the incoming particle is inside the cell and move it
                if JustPIC.isincell(p_child, corner_xi, dxi) #&& !isparticleempty(p_child)
                    # hold particle variables
                    current_p = p_child
                    current_args = @inbounds JustPIC.cache_args(args, ip, idx_child)

                    # remove particle from child cell
                    @inbounds @cell index[ip, idx_child...] = false
                    JustPIC.empty_particle!(particle_coords, ip, idx_child)
                    JustPIC.empty_particle!(args, ip, idx_child)

                    # check whether there's empty space in parent cell
                    free_idx = JustPIC.find_free_memory(index, parent_cell...)
                    free_idx == 0 && continue

                    # move particle and its fields to the first free memory location
                    @inbounds @cell index[free_idx, parent_cell...] = true

                    JustPIC.fill_particle!(particle_coords, current_p, free_idx, parent_cell)
                    JustPIC.fill_particle!(args, current_args, free_idx, parent_cell)
                end
            end
        end
    end
end

@btime foo!(
    $(
    particles0.coords,
    domain_limits,
    corner_xi,
    dxi,
    ni,
    particles0.index,
    parent_cell,
    particle_args,
    idx_loop,
    )...
)

ProfileCanvas.@profview for i in 1:5000000
    foo!(
        particles0.coords,
        domain_limits,
        corner_xi,
        dxi,
        ni,
        particles0.index,
        parent_cell,
        particle_args,
        idx_loop,
    )
end
inds = -1,25


@btime JustPIC.indomain($((0.1, 0.1), domain_limits)...)