
"""
    force_injection!(particles::Particles{Backend}, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {Backend, N}

Forcefully injects new particles into the `particles` object. This function modifies the `particles` object in place.

# Arguments
- `particles::Particles{Backend}`: The particles object to be modified.
- `p_new`: The new particles to be injected.
- `fields::NTuple{N, Any}`: A tuple containing the fields to be updated.
- `values::NTuple{N, Any}`: A tuple containing the values corresponding to the fields.

# Returns
- Nothing. This function modifies the `particles` object in place.
"""
function force_injection!(particles::Particles{Backend}, p_new, fields::NTuple{N, Any}, values::NTuple{N, Any}) where {Backend, N}
    (; coords, index) = particles;
    ni = size(index)
    @parallel (@idx ni) force_injection!(coords, index, p_new, fields, values)
    return nothing
end

"""
    force_injection!(particles::Particles{Backend}, p_new) where {Backend}

Forces the injection of new particles into the existing `particles` collection. This function modifies the `particles` in place by adding the new particles specified in `p_new`.

# Arguments
- `particles::Particles{Backend}`: The existing collection of particles to which new particles will be added. The type of backend is specified by the `Backend` parameter.
- `p_new`: The new particles to be injected into the existing collection.
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