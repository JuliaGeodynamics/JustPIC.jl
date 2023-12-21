## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
function grid2particle_naive!(Fp, xvi, F, particles::ClassicParticles)
    (; coords, parent_cell, grid_step) = particles
    np = nparticles(particles)

    @parallel (@idx np) grid2particle_classic_naive!(
        Fp, F, parent_cell, xvi, grid_step, coords
    )

    return nothing
end

@parallel_indices (ipart) function grid2particle_classic_naive!(
    Fp, F, parent_cell, xvi, di, coords
)
    @inbounds begin
        pᵢ = coords[ipart]
        icell = parent_cell[ipart]
        # Interpolate field F onto particle
        Fp[ipart] = _grid2particle(pᵢ, xvi, di, F, icell)
    end
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

# @inline function _grid2particle_classic_naive!(Fp, p, xvi, di::NTuple{N,T}, F, ipart) where {N,T}
#     pᵢ = p[ipart]
#     icell = get_cell(p[ipart], di)
#     # Interpolate field F onto particle
#     Fp[ipart] = _grid2particle(pᵢ, xvi, di, F, icell)
# end

# @inline function _grid2particle_classic_naive!(
#     Fp::NTuple{N1,T1}, p, xvi, di::NTuple{N2,T2}, F::NTuple{N1,T3}, idx
# ) where {N1,T1,N2,T2,T3}
#     # iterate over all the particles within the cells of index `idx` 
#     for ip in cellaxes(Fp[1])
#         # cache particle coordinates 
#         pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N2))

#         # skip lines below if there is no particle in this pice of memory
#         any(isnan, pᵢ) && continue

#         # Interpolate field F onto particle
#         ntuple(Val(N1)) do i
#             Base.@_inline_meta
#             @cell Fp[i][ip, idx...] = _grid2particle(pᵢ, xvi, di, F[i], idx)
#         end
#     end
# end

#  Interpolation from grid corners to particle positions

## FULL PARTICLE PIC ------------------------------------------------------------------------------------------

# LAUNCHERS

# function grid2particle_flip!(Fp, xvi, F, F0, particle_coords; α=0.0)
#     di = grid_size(xvi)
#     grid2particle_flip!(Fp, xvi, F, F0, particle_coords, di; α=α)

#     return nothing
# end

# function grid2particle_flip!(
#     Fp, xvi, F, F0, particle_coords, di::NTuple{N,T}; α=0.0
# ) where {N,T}
#     ni = length.(xvi)

#     @parallel (@idx ni .- 1) grid2particle_naive!(Fp, F, F0, xvi, di, particle_coords, α)

#     return nothing
# end

# @parallel_indices (I...) function grid2particle_naive!(
#     Fp, F, F0, xvi, di, particle_coords, α
# )
#     _grid2particle_naive!(Fp, particle_coords, xvi, di, F, F0, I, α)
#     return nothing
# end

# # INNERMOST INTERPOLATION KERNEL

# @inline function _grid2particle_naive!(
#     Fp, p, xvi, di::NTuple{N,T}, F, F0, idx, α
# ) where {N,T}
#     # iterate over all the particles within the cells of index `idx` 
#     for ip in cellaxes(Fp)
#         # cache particle coordinates 
#         pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))

#         # skip lines below if there is no particle in this pice of memory
#         any(isnan, pᵢ) && continue

#         Fᵢ = @cell Fp[ip, idx...]
#         F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (F, F0), idx)
#         ΔF = F_pic - F0_pic
#         F_flip = Fᵢ + ΔF
#         # Interpolate field F onto particle
#         @cell Fp[ip, idx...] = muladd(F_pic, α, F_flip * (1.0 - α))
#     end
# end

# @inline function _grid2particle_naive!(
#     Fp::NTuple{N1,T1},
#     p,
#     xvi,
#     di::NTuple{N2,T2},
#     F::NTuple{N1,T3},
#     F0::NTuple{N1,T3},
#     idx,
#     α,
# ) where {N1,T1,N2,T2,T3}
#     # iterate over all the particles within the cells of index `idx` 
#     for ip in cellaxes(Fp)
#         # cache particle coordinates 
#         pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N2))

#         # skip lines below if there is no particle in this pice of memory
#         any(isnan, pᵢ) && continue

#         ntuple(Val(N1)) do i
#             Base.@_inline_meta
#             Fᵢ = @cell Fp[i][ip, idx...]
#             F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (F[i], F0[i]), idx)
#             ΔF = F_pic - F0_pic
#             F_flip = Fᵢ + ΔF
#             # Interpolate field F onto particle
#             @cell Fp[i][ip, idx...] = muladd(F_pic, α, F_flip * (1.0 - α))
#         end
#     end
# end
