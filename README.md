[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliageodynamics.github.io/JustPIC.jl/dev/)
[![DOI](https://zenodo.org/badge/507905159.svg)](https://zenodo.org/doi/10.5281/zenodo.10212675)
[![CPU UnitTests](https://github.com/JuliaGeodynamics/JustPIC.jl/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/JuliaGeodynamics/JustPIC.jl/actions/workflows/UnitTests.yml)
[![GPU UnitTests](https://badge.buildkite.com/bb05ed7ef3b43f843a5ba4a976c27a724064d67955193accea.svg)](https://buildkite.com/julialang/justpic-dot-jl)
[![codecov](https://codecov.io/gh/JuliaGeodynamics/JustPIC.jl/graph/badge.svg?token=PN0AJZXK13)](https://codecov.io/gh/JuliaGeodynamics/JustPIC.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# JustPIC.jl

Particle-in-Cell advection ready to rock the GPU  :rocket:

# Example:
The first step is to chose whether we want to run our simulation on the CPU or Nvidia or AMD GPUs. This is done by setting the `backend` variable to either `CUDABackend`, `AMDGPUBackend` or `CPUBackend`. In the following we will assume that we are running on a Nvidia GPU.

```julia
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
```

and load the required packages:

```julia
using JustPIC, JustPIC._2D
using GLMakie
```

Define domain and grids of the domain:

```julia
# number of grid points
n            = 257
# number of cells
nx           = ny = n-1
# domain size
Lx           = Ly = 1.0
# nodal vertices
xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
# grid spacing
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
# staggered grid velocity nodal locations
grid_vx      = xv, expand_range(yc)
grid_vy      = expand_range(xc), yv
```
where `expand_range` is a helper function that expands a range by one element on each side:
```julia
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end
```

Now we can initialize the particles object
```julia
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 12 # minimum number of particles per cell
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, (nx, ny)
    )
```

The velocity field is defined by the stream function $\psi=\frac{250}{\pi}\sin(\pi x)\cos(\pi y)$, so that the analytical velocity field at the particle $p=p(x,y)$ is given by
```julia
vx_stream(x, y) =   250 * sin(π*x) * cos(π*y)
vy_stream(x, y) =  -250 * cos(π*x) * sin(π*y)
```
and therefore the velocity field (__with ghost nodes__) on the staggered grid is given by
```julia
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
V  = Vx, Vy
dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy))) # time step
```
where `TA(backend)` is a type alias for either `Array`, `CuArray` or `ROCArray`.

We save the initial particle positions:
```julia
pxv = particles.coords[1].data;
pyv = particles.coords[2].data;
idxv = particles.index.data;
p = [(pxv[idxv][1], pyv[idxv][1])]
```

chose the advection scheme
```julia
advection_scheme = RungeKutta2()
```

and finally perform the time iterations:
```julia
 # Advection test
particle_args = ()
niter = 750
for iter in 1:niter
    # advect particles
    advection!(particles, advection_scheme, V, (grid_vx, grid_vy), dt)
    # shuffle particles in the memory to keep the spatial locality tight
    move_particles!(particles, xvi, particle_args)
    # save particle position
    pxv = particles.coords[1].data;
    pyv = particles.coords[2].data;
    idxv = particles.index.data;
    p_i = (pxv[idxv][1], pyv[idxv][1])
    push!(p, p_i)
end
```

where `particle_args` is an empty tuple, but typically it contains the fields that are advected with the particles (e.g. temperature). At last, we plot the particle trajectory on top of the stream function:
```julia
using GLMakie
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)
f, ax, = streamplot(g, xvi...)
lines!(ax, p, color=:red)
f
```


## Funding
The development of this package is supported by the [GPU4GEO](https://ptsolvers.github.io/GPU4GEO/) [PASC](https://www.pasc-ch.org) project.
