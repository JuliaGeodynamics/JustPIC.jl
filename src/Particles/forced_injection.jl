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
    launch!(ka_backend(index), force_injection_kernel!, ni, coords, index, p_new, fields, values)
    return nothing
end

"""
    force_injection!(particles, p_new)

Convenience method for `force_injection!` when no companion particle fields need
to be initialized.
"""
force_injection!(particles::Particles{Backend}, p_new) where {Backend} = force_injection!(particles, p_new, (), ())


@kernel function force_injection_kernel!(coords::NTuple{2}, index, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {N}
    I = @index(Global, NTuple)

    # check whether there are new particles to inject in the ij-th cell
    if !isnan(p_new[I..., begin])
        c = 0 # helper counter
        # iterate over particles in the cell
        for ip in cellaxes(index)
            c += 1
            c > cellnum(index)  && continue
            doskip(index, ip, I...) || continue
            pᵢ = p_new[I..., c]
            CAI.@index coords[1][ip, I...] = pᵢ[1]
            CAI.@index coords[2][ip, I...] = pᵢ[2]
            CAI.@index index[ip, I...] = true

            # force fields to have a given value
            for (value, field) in zip(values, fields)
                CAI.@index field[ip, I...] = value
            end
        end
    end
end

@kernel function force_injection_kernel!(coords::NTuple{3}, index, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {N}
    I = @index(Global, NTuple)

    # check whether there are new particles to inject in the ij-th cell
    if !isnan(p_new[I..., begin])
        c = 0 # helper counter
        # iterate over particles in the cell
        for ip in cellaxes(index)
            c += 1
            c > cellnum(index)  && continue
            doskip(index, ip, I...) || continue
            pᵢ = p_new[I..., c]
            CAI.@index coords[1][ip, I...] = pᵢ[1]
            CAI.@index coords[2][ip, I...] = pᵢ[2]
            CAI.@index coords[3][ip, I...] = pᵢ[3]
            CAI.@index index[ip, I...] = true

            # force fields to have a given value
            for (value, field) in zip(values, fields)
                CAI.@index field[ip, I...] = value
            end
        end
    end
end
