@inline _interp1D(xq, x0, x1, y0, y1) = muladd((xq - x0), (y1 - y0) * inv(x1 - x0), y0)

function interp1D_extremas(xq, x, y)
    last_I = 1
    for i in length(x):-1:2
        if !isnan(x[i])
            last_I = i
            break
        end
    end
    x_lo, x_hi = x[1], x[last_I]
    @inbounds for j in eachindex(x)[1:(end - 1)]
        x0, x1 = x[j], x[j + 1]

        # interpolation
        if x0 ≤ xq ≤ x1
            y0, y1 = y[j], y[j + 1]
            return _interp1D(xq, x0, x1, y0, y1)
        end

        # extrapolation
        if xq ≤ x_lo
            x0, x1 = x[1], x[2]
            y0, y1 = y[1], y[2]
            return _interp1D(xq, x0, x1, y0, y1)
        end

        if xq ≥ x_hi
            x0, x1 = x[last_I], x[last_I - 1]
            y0, y1 = y[last_I], y[last_I - 1]
            return _interp1D(xq, x0, x1, y0, y1)
        end
    end
    # return error("xq outside domain")
    return NaN
end

function interp1D_inner(xq, x, y, coords, I::Integer)
    last_I = 1
    for i in length(x):-1:2
        if !isnan(x[i])
            last_I = i
            break
        end
    end
    x_lo, x_hi = x[1], x[last_I]
    @inbounds for j in 1:(last_I - 1)
        x0, x1 = x[j], x[j + 1]

        # interpolate using the last particle of left-neighbouring cell
        if xq ≤ x_lo
            x0, y0 = left_cell_right_particle(coords, I)
            x1, y1 = x[1], y[1]
            return _interp1D(xq, x0, x1, y0, y1)
        end

        # interpolate using the first particle of right-neighbouring cell
        if xq ≥ x_hi
            x0, y0 = x[last_I], y[last_I]
            x1, y1 = right_cell_left_particle(coords, I)
            return _interp1D(xq, x0, x1, y0, y1)
        end

        # interpolation
        if x0 ≤ xq ≤ x1
            y0, y1 = y[j], y[j + 1]
            return _interp1D(xq, x0, x1, y0, y1)
        end
    end
    # return error("xq outside domain")
    return NaN
end

@inline right_cell_left_particle(coords, I::Int) =
    @index(coords[1][1, I + 1]), @index(coords[2][1, I + 1])

@inline function left_cell_right_particle(coords, I)
    px = coords[1]
    # px = @cell coords[1][I - 1]
    ip = 1
    for i in cellnum(px):-1:2
        if !isnan(@index px[i, I - 1])
            ip = i
            break
        end
    end

    return @index(px[ip, I - 1]), @index(coords[2][ip, I - 1])
end

@inline function is_above_surface(xq, yq, coords, cell_vertices)
    I = cell_index(xq, cell_vertices)
    x_cell, y_cell = coords[1][I], coords[2][I]
    return yq > interp1D_inner(xq, x_cell, y_cell, coords, I)
end
