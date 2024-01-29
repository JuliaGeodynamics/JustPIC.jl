"""
    init_particle_fields(particles, ::Val{N})

Returns `N` particle fields with the same size as `particles`
"""
@inline function init_cell_arrays(particles::Particles, ::Val{N}) where {N}
    return ntuple(
        _ -> @fill(
            0.0, size(particles.coords[1])..., celldims = (cellsize(particles.index))
        ),
        Val(N),
    )
end

@inline function cell_array(x::T, ncells::NTuple{N,Integer}, ni::Vararg{Any,N}) where {T,N}
    @fill(x, ni..., celldims = ncells, eltype = T)
end

## random particles initialization 

function init_particles(backend, nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
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
        backend, (px, py), index, inject, nxcell, max_xcell, min_xcell, np,
    )
end

function init_particles(backend, nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, ni)
    ncells     = prod(ni)
    np         = max_xcell * ncells
    px, py, pz = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(3))
    inject     = @fill(false, ni..., eltype=Bool)
    index      = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j, k) function fill_coords_index(px, py, pz, index)    
        # lower-left corner of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j, k]    = x0 + dx * rand(0.05:1e-5:0.95)
            @cell py[l, i, j, k]    = y0 + dy * rand(0.05:1e-5:0.95)
            @cell pz[l, i, j, k]    = z0 + dz * rand(0.05:1e-5:0.95)
            @cell index[l, i, j, k] = true
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(px, py, pz, index)    

    return Particles(
        backend,(px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np
    )
end