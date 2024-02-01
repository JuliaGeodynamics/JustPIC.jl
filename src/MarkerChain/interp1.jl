# function interp1D(xq::Number, x::AbstractVector{T}, y::AbstractVector{T}) where T
#     @inbounds for j in eachindex(x)[1:end-1]
#         x0, x1 = x[j], x[j+1]
#         if x0 < xq < x1
#             y0, y1 = y[j], y[j+1]
#             return fma(
#                 (xq-x0),
#                 (y1-y0)*inv(x1-x0), 
#                 y0
#             )
#         end
#     end
#     error("xq out of range")
# end

function interp1D(xq::Number, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    x_lo, x_hi = x[1], x[end]
    @inbounds for j in eachindex(x)
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
