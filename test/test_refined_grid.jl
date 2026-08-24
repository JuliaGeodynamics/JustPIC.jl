using Test
using JustPIC
using LinearAlgebra
import JustPIC: lerp
import KernelAbstractions: CPU

const backend = CPU

function expand_range(x::AbstractVector)
    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx_left
    xF = x2 + dx_right
    return vcat(xI, x, xF)
end

# checks if grid options are reasonable
function checkGridLength(n, d0, f)
    if f < 1
        error("Growth factor cannot be smaller than 1!")
    elseif isone(f)
        return n * d0
    else
        return d0 * (f^n - 1) / (f - 1)
    end
end

# finds the correct growth factor by bisection
function findGrowthFactor(L, n, d0)
    a = 1.0
    b = 2.0
    for i in 1:20
        c = (a + b) / 2.0
        err = checkGridLength(n, d0, c) - L
        if abs(err) < L / 1.0e3
            # println("Grid growth factor: $(c)")
            return c
        elseif err > 0
            b = c
        else
            a = c
        end
        #println("c: $(c)")
    end
    return println("Grid seems impossible!")
end

# make exponential grid
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

    # dx from the vertices
    return xn, xc[2:(end - 1)], dx
end


# Initialize particles -------------------------------
nxcell, max_xcell, min_xcell = 4, 4, 4
n = 9
nx = ny = n
Lx = Ly = 1.0

dx0 = Lx / nx
dy0 = Ly / ny

# refined coordinates
xv, xc, dx = makeExpoGrid(Lx, nx, dx0 / 2, 0.0e0)
yv, yc, dy = makeExpoGrid(Ly, ny, dy0 / 1, 0.0e0)

xvi = xv, yv
xci = xc, yc

# staggered grid velocity nodal locations
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv
grid_vi = grid_vx, grid_vy

xvi_device = TA(backend).(xvi)
grid_vi_device = (
    TA(backend).(grid_vi[1]),
    TA(backend).(grid_vi[2]),
)

particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vi_device...,
)

# init_particles pads the vertex/center grids with periodic ghost nodes; use its
# stored (padded) grids rather than the pre-padding local xvi/xci so field arrays
# line up with particles.coords.
xvi_p = particles.xvi
xci_p = particles.xci

# Linear field at the vertices
T = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
T0 = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
# Linear field at the centroids
Tc = TA(backend)([y for x in xci_p[1], y in xci_p[2]])

pT, = init_cell_arrays(particles, Val(1))
active = Array(particles.index.data)

@testset "Interpolations 2D on refined grid" begin

    # Grid to particle test
    JustPIC.grid2particle!(pT, T, particles)
    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Grid to particle test
    JustPIC.grid2particle_flip!(pT, xvi_p, T, T0, particles)
    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to grid test
    T2 = similar(T)
    fill!(T2, NaN)
    JustPIC.particle2grid!(T2, pT, particles)
    finite_mask = isfinite.(T2)
    @test norm(T2[finite_mask] .- T[finite_mask]) / count(finite_mask) < 1.0e-1

    # Grid to centroid test
    JustPIC.centroid2particle!(pT, Tc, particles)
    @test Array(pT.data)[active] ≈ Array(particles.coords[2].data)[active]

    # Particle to centroid test
    Tc2 = similar(Tc)
    fill!(Tc2, NaN)
    JustPIC.particle2centroid!(Tc2, pT, particles)
    finite_mask_c = isfinite.(Tc2)
    @test norm(Tc2[finite_mask_c] .- Tc[finite_mask_c]) / count(finite_mask_c) < 1.0e-1

end
