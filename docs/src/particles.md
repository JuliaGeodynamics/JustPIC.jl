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
struct Particles{Backend,N,M,I,T1,T2} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I
end
```
Where `coords` is a tuple containing the coordinates of the particles; `index` is a `BitArray` where `true` if in the correspondent `CellArray` there is an active particle, otherwise false; `nxcell`, `min_xcell`, `max_xcell` are the initial, minimum and maximum number of particles per cell; and, `np` is the initial number of particles.
    
### Passive markers
```julia
struct MarkerChain{Backend,N,M,I,T1,T2,TV} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    cell_vertices::TV 
    max_xcell::I
    min_xcell::I
end
```
Where `coords` is a tuple containing the coordinates of the particles; `index` is an `BitArray` where `true` if in the correspondent `CellArray` there is an active particle, otherwise false; `cell_vertices` is the lower-left corner (`(x,)` in 2D, `(x,y)` in 3D) of the cell containing those particles;and, `min_xcell`, `max_xcell` are theminimum and maximum number of particles per cell.

### Marker chain
```julia
struct PassiveMarkers{Backend,T} <: AbstractParticles
    coords::T
    np::Int64
end
```
Where `coords` is a tuple containing the coordinates of the particles; and `np` is the number of passive markers.

## Particle initialization

Particles can be initialized as randomly distributed, or regularly spaced. As seen in the examples below, if `nxcell` is a scalar integer, particles will be randomly initialized. If `nxcell` is a tuple integers (with length 2 in 2D, and length 3 in 3D), particles will regularly spaced, with the elements of the tuple being the number of particles per dimension.

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
