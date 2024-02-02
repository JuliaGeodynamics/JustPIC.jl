@inline _interp1D(xq, x0, x1, y0, y1) = fma((xq - x0), (y1 - y0) * inv(x1 - x0), y0)

function interp1D_extremas(xq, x, y)
    x_lo, x_hi = x[1], x[end]
    @inbounds for j in eachindex(x)[1:(end - 1)]
        x0, x1 = x[j], x[j + 1]

        # interpolation
        if x0 < xq < x1
            y0, y1 = y[j], y[j + 1]
            _interp1D(xq, x0, x1, y0, y1)
        end

        # extrapolation
        if xq < x_lo
            x0, x1 = x[1], x[2]
            y0, y1 = y[1], y[2]
            _interp1D(xq, x0, x1, y0, y1)
        end

        if xq > x_hi
            x0, x1 = x[end], x[end - 1]
            y0, y1 = y[end], y[end - 1]
            _interp1D(xq, x0, x1, y0, y1)
        end
    end
end

function interp1D_inner(xq, x, y, cell_coords, I::Integer)
    x_lo, x_hi = x[1], x[end]
    @inbounds for j in eachindex(x)[1:(end - 1)]
        x0, x1 = x[j], x[j + 1]

        # interpolation
        if x0 < xq < x1
            y0, y1 = y[j], y[j + 1]
            _interp1D(xq, x0, x1, y0, y1)
        end

        # interpolate using the last particle of left-neighbouring cell
        if xq < x_lo
            x0, y0 = left_cell_right_particle(cell_coords, I)
            x1, y1 = x[1], y[1]
            _interp1D(xq, x0, x1, y0, y1)
        end
        # interpolate using the first particle of right-neighbouring cell
        if xq > x_hi
            x0, y0 = x[end], y[end]
            x1, y1 = right_cell_left_particle(cell_coords, I)
            _interp1D(xq, x0, x1, y0, y1)
        end
    end
end
