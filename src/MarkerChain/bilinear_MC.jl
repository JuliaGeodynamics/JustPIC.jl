function compute_topography_vertex!(chain::MarkerChain)
    (; coords, index, cell_vertices, h_vertices) = chain
    chain_x, chain_y = coords

    _dx = inv(cell_vertices[2] - cell_vertices[1])

    @parallel (1:length(cell_vertices)) _compute_h_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index, _dx
    )

    return nothing
end

@parallel_indices (ivertex) function _compute_h_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index, _dx
    )
    _compute_h_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, _dx, ivertex
    )
    return nothing
end

function _compute_h_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, _dx::T, ivertex
    ) where {T}
    h = zero(T)
    ω = zero(T)
    xcorner = cell_vertices[ivertex]

    # iterate over cells on the left and right hand sides of the vertex
    multiplier = -one(T)
    for j in (ivertex - 1):ivertex
        if 0 < j < length(cell_vertices)
            for ip in cellaxes(index)
                @index(index[ip, j]) || continue

                x_m = @index chain_x[ip, j]
                y_m = @index chain_y[ip, j]
                dx_m = multiplier * (x_m - xcorner)
                ωᵢ = @muladd one(T) - dx_m * _dx
                h += y_m * ωᵢ
                ω += ωᵢ
            end
        end
        multiplier = one(T)
    end
    h_vertices[ivertex] = h / ω

    return nothing
end

######################################

function reconstruct_chain_from_vertices!(chain::MarkerChain)
    (; coords, index, cell_vertices, h_vertices) = chain
    chain_x, chain_y = coords

    @parallel (1:length(index)) _reconstruct_h_from_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index
    )

    return nothing
end

@parallel_indices (ivertex) function _reconstruct_h_from_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index
    )
    _reconstruct_h_from_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
    return nothing
end

function _reconstruct_h_from_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
    xcorner_left = cell_vertices[ivertex]
    xcorner_right = cell_vertices[ivertex + 1]
    lx = xcorner_right - xcorner_left
    ycorner_left = h_vertices[ivertex]
    ycorner_right = h_vertices[ivertex + 1]
    ly = ycorner_right - ycorner_left

    # count active particles
    np = 0
    for ip in cellaxes(index)
        @index(index[ip, ivertex]) || break
        np += 1
    end

    Δx = lx / (np + 1)
    Δy = ly / (np + 1)
    xp_new = xcorner_left
    yp_new = ycorner_left

    # fill cell arrays with new particle coordinates
    for ip in cellaxes(index)
        @index(index[ip, ivertex]) || break

        xp_new += Δx
        yp_new += Δy
        @index chain_x[ip, ivertex] = xp_new
        @index chain_y[ip, ivertex] = yp_new
    end

    return nothing
end
