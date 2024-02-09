using JLD2

function save_particles(particles::AbstractParticles, filename)
    (; coords) = particles.coords
    pxi = ntuple(Val(length(coords))) do i 
        tmp = Array(coords[i].x.data)[:]
        filter(!isnan(x), tmp)
    end

    f(x, y)    = jldsave(filename * ".jld2"; x, y)
    f(x, y, z) = jldsave(filename * ".jld2"; x, y, z)

    f(pxi...)
end

