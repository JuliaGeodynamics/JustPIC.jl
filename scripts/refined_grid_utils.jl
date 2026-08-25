# Helpers for building graded (non-uniformly spaced) coordinate grids.

"""
    expand_range(x::AbstractVector)

Extend a coordinate vector by one node on each side, preserving the spacing of the first
and last cells. Used to build the ghost-node rows of a staggered velocity grid.
"""
function expand_range(x::AbstractVector)
    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    return vcat(x[1] - dx_left, x, x[end] + dx_right)
end

"""
    checkGridLength(n, d0, f)

Total length spanned by `n` cells whose widths grow geometrically from `d0` by the factor
`f`.
"""
function checkGridLength(n, d0, f)
    if f < 1
        error("Growth factor cannot be smaller than 1!")
    elseif isone(f)
        return n * d0
    else
        return d0 * (f^n - 1) / (f - 1)
    end
end

"""
    findGrowthFactor(L, n, d0)

Growth factor for which `n` geometrically graded cells starting at width `d0` span `L`,
found by bisection.
"""
function findGrowthFactor(L, n, d0)
    a = 1.0
    b = 2.0
    for i in 1:20
        c = (a + b) / 2.0
        err = checkGridLength(n, d0, c) - L
        if abs(err) < L / 1.0e3
            return c
        elseif err > 0
            b = c
        else
            a = c
        end
    end
    return error("No growth factor spans L = $L with $n cells of minimum width $d0")
end

"""
    makeExpoGrid(L, n, d0, x0)

Build a grid of `n` cells spanning `[x0, x0 + L]`, refined to width `d0` at the center and
coarsening geometrically towards both ends.

Returns `(xn, xc, dx)`: the `n + 1` vertices, the `n` cell centers, and the `n` cell widths.
"""
function makeExpoGrid(L, n, d0, x0)
    dx = zeros(n)
    if mod(n, 2) == 0
        L2 = L / 2.0
        n2 = Int64(n / 2)
        f = findGrowthFactor(L2, n2, d0)
        dx[n2:(n2 + 1)] .= d0
        dn = 2
    else
        L2 = L / 2.0 + d0 / 2.0
        n2 = Int64((n + 1) / 2)
        f = findGrowthFactor(L2, n2, d0)
        dx[n2] = d0
        dn = 1
    end
    for i in (n2 + dn):(n - 1)
        dx[i] = dx[i - 1] * f
    end
    for i in (n2 - 1):-1:2
        dx[i] = dx[i + 1] * f
    end

    dx[1] = (L - sum(dx)) / 2.0
    dx[end] = dx[1]

    xn = zeros(n + 1)
    xc = zeros(n + 2) # with ghost cells
    xn[1] = x0
    xc[1] = x0 - dx[1] / 2.0
    xc[end] = x0 + L + dx[end] / 2.0
    for i in 1:n
        xn[i + 1] = xn[i] + dx[i]
        xc[i + 1] = (xn[i] + xn[i + 1]) / 2.0
    end

    return xn, xc[2:(end - 1)], dx
end
