"""
    force_injection!(particles, p_new, fields, values)

Insert particles from `p_new` directly into free particle slots.

# Arguments
- `particles`: destination `Particles` container.
- `p_new`: per-cell collection of coordinates to inject; `NaN` marks empty input slots.
- `fields`: tuple of particle fields to initialize together with the coordinates.
- `values`: values written into each corresponding entry of `fields`.

# Notes
- This is a low-level routine: it does not search for nearest-neighbor values.
- Injection only happens into currently inactive particle slots.
"""
function force_injection!(particles::Particles{Backend}, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {Backend, N}
    (; coords, index) = particles
    ni = size(index)
    @parallel (@idx ni) force_injection!(coords, index, p_new, fields, values)
    return nothing
end

"""
    force_injection!(particles, p_new)

Convenience method for `force_injection!` when no companion particle fields need
to be initialized.
"""
force_injection!(particles::Particles{Backend}, p_new) where {Backend} = force_injection!(particles, p_new, (), ())


@parallel_indices (I...) function force_injection!(coords::NTuple{2}, index, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {N}

    # check whether there are new particles to inject in the ij-th cell
    if !isnan(p_new[I..., begin])
        c = 0 # helper counter
        # iterate over particles in the cell
        for ip in cellaxes(index)
            c += 1
            c > cellnum(index)  && continue
            doskip(index, ip, I...) || continue
            pᵢ = p_new[I..., c]
            @index coords[1][ip, I...] = pᵢ[1]
            @index coords[2][ip, I...] = pᵢ[2]
            @index index[ip, I...] = true

            # force fields to have a given value
            for (value, field) in zip(values, fields)
                @index field[ip, I...] = value
            end
        end
    end

    return nothing
end
