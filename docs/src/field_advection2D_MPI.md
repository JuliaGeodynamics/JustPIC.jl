# Field advection in 2D using MPI

This example uses periodic boundaries in both coordinate directions. The local
particle grids therefore include periodic ghost nodes, and MPI halo exchanges
must happen at the right points of the advection loop so that particles and
their fields remain consistent across rank boundaries.

As usual, we start loading JustPIC.jl modules and specifying the backend

```julia
using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend
```

and we define the usual analytical flow solution

```julia
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
```
This time, we also need to load MPI.jl and ImplicitGlobalGrid.jl to handle the MPI communication between the different CPU's

```julia
using ImplicitGlobalGrid
using MPI: MPI
```

Then we define the model domain

```julia
n  = 256        # number of nodes
nx = ny = n-1   # number of cells in x and y
Lx = Ly = 1.0   # domain size
```

, initialize the global grid
```julia
me, dims, = init_global_grid(n-1, n-1, 1; init_MPI=MPI.Initialized() ? false : true)
dxi = dx, dy = Lx /(nx_g()-1), Ly / (ny_g()-1)
```
and the arrays local to each MPI rank

```julia
# nodal vertices
xvi = xv, yv = let
    dummy = zeros(n, n) 
    xv  = TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
    yv  = TA(backend)([y_g(i, dx, dummy) for i in axes(dummy, 2)])
    xv, yv
end
# nodal centers
xci = xc, yc = let
    dummy = zeros(nx, ny) 
    xc  = @zeros(nx) 
    xc .= TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
    yc  = TA(backend)([y_g(i, dx, dummy) for i in axes(dummy, 2)])
    xc, yc
end
# staggered grid for the velocity components
grid_vx = xv, add_ghost_nodes(yc, dy, (0.0, Ly))
grid_vy = add_ghost_nodes(xc, dx, (0.0, Lx)), yv
```

And we continue with business as usual

```julia
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
)
```

and the velocity and field we want to advect (on the staggered grid)

```julia
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
T  = TA(backend)([y for x in xv, y in yv]); # defined at the cell vertices
V  = Vx, Vy;
dt = min(
    dx / MPI.Allreduce(maximum(abs.(Vx)), MPI.MAX, MPI.COMM_WORLD),
    dy / MPI.Allreduce(maximum(abs.(Vy)), MPI.MAX, MPI.COMM_WORLD)
)
```

Note that now we need to reduce over all the MPI ranks to compute the time-step.

We finally initialize the field `T` on the particles

```julia
particle_args = pT, = init_cell_arrays(particles, Val(1));
```

and we use the function `grid2particle!` to interpolate the field `T` to the particles

```julia
grid2particle!(pT, T, particles);
```

Now start the simulation

```julia
niter = 250
for it in 1:niter
    # advect particles
    advection!(particles, RungeKutta2(), V, dt)
    # exchange MPI halos before shuffling particles between parent cells
    update_cell_halo!(particles.coords...)
    update_cell_halo!(particle_args...)
    update_cell_halo!(particles.index)
    # move particles in memory
    move_particles!(particles, particle_args)
    # inject particles if needed
    inject_particles!(particles, (pT, ))
    # refresh halos again so reinjected particles are visible across ranks
    update_cell_halo!(particles.coords...)
    update_cell_halo!(particle_args...)
    update_cell_halo!(particles.index)
    # interpolate particles to the grid
    particle2grid!(T, pT, particles)
end
```

## Periodic Boundary Conditions

For periodic MPI runs, there are two separate mechanisms working together:

- `init_particles` extends the particle vertex and center grids with periodic
  ghost nodes.
- `update_cell_halo!` exchanges the overlapping `CellArray` halos between MPI
  ranks, including across periodic rank boundaries configured by
  `init_global_grid`.

The usual pattern is:

1. advect particles locally
2. exchange halos with `update_cell_halo!`
3. call `move_particles!` so particles that crossed a rank boundary are
   reassigned on the neighboring rank
4. reinject particles if needed
5. exchange halos again before `particle2grid!`

In code:

```julia
advection!(particles, RungeKutta2(), V, dt)
update_cell_halo!(particles.coords...)
update_cell_halo!(particle_args...)
update_cell_halo!(particles.index)
move_particles!(particles, particle_args)
inject_particles!(particles, (pT,))
update_cell_halo!(particles.coords...)
update_cell_halo!(particle_args...)
update_cell_halo!(particles.index)
particle2grid!(T, pT, particles)
```

If the second halo refresh is skipped after reinjection, `particle2grid!` can
reconstruct grid nodes from incomplete cross-rank neighborhoods. If the first
halo refresh is skipped before `move_particles!`, particles that leave a rank
can fail to appear on the neighboring rank.

## Visualization
To visualize the results, we need to allocate a global array `T_v` and buffer arrays `T_nohalo` without the overlapping halo (here with `width = 1`)
```julia
nx_v     = (size(T, 1) - 2) * dims[1] # global size of `T` without halos
ny_v     = (size(T, 2) - 2) * dims[2] # global size of `T` without halos
T_v      = zeros(nx_v, ny_v)          # initialize global `T`
T_nohalo = @zeros(size(T).-2)         # local `T` without overlapping halo
```

Visualization with GLMakie.jl
```julia
using GLMakie
x_global = range(0, Lx, length=size(T_v,1))
y_global = range(0, Ly, length=size(T_v,2))
heatmap(x_global, y_global, T_v)
```

## Going 3D
A 3D example using MPI is found in `scripts/temperature_advection3D_MPI.jl`.
