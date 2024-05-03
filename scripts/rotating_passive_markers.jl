using JustPIC
using JustPIC._2D
using GLMakie

const backend = JustPIC.CPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n+2)
end

# Analytical flow solution
vi_stream(x) =  π*1e-5 * (x - 0.5)

function main()

    # Initialize particles -------------------------------
    n = 51
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);

    T   = TA(backend)([y for x in xv, y in yv]);
    P   = TA(backend)([x for x in xv, y in yv]);
    V   = Vx, Vy;

    w      = π*1e-5  # angular velocity
    period = 1  # revolution number
    tmax   = period / (w/(2*π))
    dt     = 200.0

    np = 256 # number of passive markers
    passive_coords = ntuple(Val(2)) do i
        (rand(np) .+ 1) .* Lx/4
    end

    passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = TA(backend)(zeros(np))
    P_marker = TA(backend)(zeros(np))

    f = Figure()
    ax = Axis(f[1, 1])
    # vector of shapes
    poly!(
        ax,
        Rect(0, 0, 1, 1),
        color=:lightgray,
    )
    px = passive_markers.coords[1].data[:];
    py = passive_markers.coords[2].data[:];
    scatter!(px, py, color=:black)
    
    for _ in 1:325
        advection!(passive_markers, RungeKutta2(), V, (grid_vx, grid_vy), dt)
    end

    # interpolate grid fields T and P onto the marker locations
    grid2particle!((T_marker, P_marker), xvi, (T, P), passive_markers)

    px = passive_markers.coords[1].data[:];
    py = passive_markers.coords[2].data[:];
    scatter!(px, py, color = _marker)
    display(f)

end