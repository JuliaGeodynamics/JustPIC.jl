# Particles 

## Memory layout

Particles stored in `CellArray`s objects from [`CellArrays.jl`](https://github.com/omlins/CellArrays.jl) and are constantly sorted by their parent cell to avoid a loss of spatial locality with time.

## Particle objects

There are three type of `AbstractParticles` types:

- `Particles` is the basic type used to advect and track information on the whole domain of our model. The dimension of the `CellArrays` has to match the dimension of the model.
- `PassiveMarkers` is a set of particles where their initial position is defined by the user. These particles are only advected and are not meant to have any feedback with the simulation; their purpose is to track the history of any arbitrary field(s) throughout the simulation.
- `MarkerChain` is a one or two dimensional chain of particles, used to track surfaces / interfaces.

### Simulation particles
```julia
struct Particles{Backend,N,I,T1,T2,D,V} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I
    di::D
    _di::D
    xci::NTuple{N,V}
    xvi::NTuple{N,V}
    xi_vel::NTuple{N,NTuple{N,V}}
end
```
Where `coords` is a tuple containing the particle coordinates; `index` marks active slots in each cell; `nxcell`, `min_xcell`, and `max_xcell` are the initial, minimum, and maximum number of particles per cell; and `np` is the total number of particle slots. The container also stores grid metadata:

- `di` and `_di`: direct and inverse grid spacings for center, vertex, and velocity grids.
- `xci`: cell-center coordinates.
- `xvi`: vertex coordinates.
- `xi_vel`: staggered velocity-grid coordinates.

These extra fields are what allow high-level helpers such as `move_particles!`, `grid2particle!`, `particle2grid!`, `inject_particles!`, `update_phase_ratios!`, and `subgrid_diffusion!` to use the short `(..., particles, ...)` call forms without passing the grids explicitly every time.
    
### Marker chain
```julia
struct MarkerChain{Backend,N,M,I,T1,T2,TV} <: AbstractParticles
    coords::NTuple{N,T1}
    coords0::NTuple{N,T1}
    h_vertices::T2
    h_vertices0::T2
    cell_vertices::TV 
    index::T2
    max_xcell::I
    min_xcell::I
end
```
Where `coords` and `coords0` store the current and previous marker coordinates, `h_vertices` and `h_vertices0` store the current and previous topography sampled at grid vertices, `cell_vertices` holds the horizontal vertex coordinates of the chain grid, `index` marks active marker slots, and `min_xcell`/`max_xcell` define the allowed markers per cell.

### Passive markers
```julia
struct PassiveMarkers{Backend,T} <: AbstractParticles
    coords::T
    np::Int64
end
```
Where `coords` contains the tracer coordinates and `np` is the total number of passive markers.

## Particle initialization

Particles can be initialized as randomly distributed, or regularly spaced. If `nxcell` is a scalar integer, particles are initialized randomly within each cell. If `nxcell` is a tuple of integers, particles are initialized on a regular layout, with one entry per coordinate direction.

After construction, the returned `Particles` object already contains the derived center, vertex, and staggered velocity grids needed by the higher-level APIs.

### Randomly distributed particles

```julia
backend   = JustPIC.CPUBackend # device backend
nxcell    = 24  # initial number of randomly distributed particles
max_xcell = 48  # maximum number of particles per cell
min_xcell = 12  # minimum number of particles per cell
n         = 32  # number of cells per dimension
Lx   = Ly = 1.0 # domain size
xvi       = LinRange(0, Lx, n), LinRange(0, Ly, n) # nodal vertices
## initialize particles object with randomly distributed coordinates
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi...,
)
```

### Regularly spaces particles

```julia
backend   = JustPIC.CPUBackend # device backend
nxcell    = (5, 5)  # number of evenly spaced particles in the x- and y- dimensions
max_xcell = 48      # maximum number of particles per cell
min_xcell = 12      # minimum number of particles per cell
n         = 32      # number of cells per dimension
Lx   = Ly = 1.0     # domain size
xvi       = LinRange(0, Lx, n), LinRange(0, Ly, n) # nodal vertices
## initialize particles object with randomly distributed coordinates
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi...,
)
```

## Convenience APIs

Once `particles` has been initialized, most particle-grid transfer and maintenance routines use the geometry stored in the container directly:

```julia
grid2particle!(Fp, F, particles)
particle2grid!(F, Fp, particles)
move_particles!(particles, particle_args)
inject_particles!(particles, particle_args)
update_phase_ratios!(phase_ratios, particles, phases)
```

This is the preferred high-level style for simulation code, tests, and examples.
