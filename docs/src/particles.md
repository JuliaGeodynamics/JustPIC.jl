# Particles 

## Memory layout

Particles stored in `CellArray`s objects from [`CellArrays.jl`](https://github.com/omlins/CellArrays.jl) and are constantly sorted by their parent cell to avoid a loss of spatial locality with time.

## Particle objects

There are three type of `AbstractParticles` types:

- `Particles` is the basic type used to advect and track information on the whole domain of our model. The dimension of the `CellArrays` has to match the dimension of the model.
- `PassiveMarkers` is a set of particles where their initial position is defined by the user. These particles are only advected and are not meant to have any feedback with the simulation; their purpose is to track the history of any arbitray field(s) throughout the simulation.
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
