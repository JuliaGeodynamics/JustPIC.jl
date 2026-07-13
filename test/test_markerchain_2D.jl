const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif BACKEND_NAME == "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustPIC, Test, Statistics
using CellArrays: field
import KernelAbstractions: CPU

const backend = @static if BACKEND_NAME == "AMDGPU"
    AMDGPU.ROCBackend
elseif BACKEND_NAME == "CUDA"
    CUDA.CUDABackend
else
    CPU
end

host_data(A) = dropdims(Array(A).data; dims = 1)
host_grid(x) = Array(x)

function host_chain(chain)
    px = host_data(chain.coords[1])
    py = host_data(chain.coords[2])
    index = host_data(chain.index)
    cell_vertices = host_grid(chain.cell_vertices)
    return px, py, index, cell_vertices
end

chain_tol(chain) = eltype(Array(chain.h_vertices)) <: Float32 ? 1.0f-5 : 1.0e-10

# set slot `ip` of cell `cell` without assuming the backend's CellArray data layout
function set_cell_slot!(A, ip, cell, val)
    f = field(A, ip)
    h = Array(f)
    h[cell] = val
    copyto!(f, h)
    return nothing
end
active_counts(index) = [count(@view index[:, i]) for i in axes(index, 2)]

function markerchain_expand_range(x)
    dx = x[2] - x[1]
    return range(first(x) - dx, last(x) + dx; length = length(x) + 2)
end

function markerchain_velocity_grid(n = 17)
    xv = range(0.0, 1.0; length = n)
    yv = range(0.0, 1.0; length = n)
    dx = xv[2] - xv[1]
    dy = yv[2] - yv[1]
    xc = range(dx / 2, 1.0 - dx / 2; length = n - 1)
    yc = range(dy / 2, 1.0 - dy / 2; length = n - 1)
    grid_vx = xv, markerchain_expand_range(yc)
    grid_vy = markerchain_expand_range(xc), yv
    return xv, yv, grid_vx, grid_vy
end

function constant_markerchain_velocity(grid_vx, grid_vy, vx, vy)
    Vx = TA(backend)(fill(vx, length(grid_vx[1]), length(grid_vx[2])))
    Vy = TA(backend)(fill(vy, length(grid_vy[1]), length(grid_vy[2])))
    return Vx, Vy
end

function assert_chain_invariants(chain)
    px, py, index, cell_vertices = host_chain(chain)
    @test size(px) == size(py) == size(index)
    for i in axes(index, 2), ip in axes(index, 1)
        if index[ip, i]
            @test !isnan(px[ip, i])
            @test !isnan(py[ip, i])
            @test cell_vertices[i] < px[ip, i] < cell_vertices[i + 1]
        else
            @test isnan(px[ip, i])
            @test isnan(py[ip, i])
        end
    end
    return nothing
end

function assert_markers_on_line(chain, a, b)
    px, py, index, _ = host_chain(chain)
    atol = chain_tol(chain)
    for i in axes(index, 2), ip in axes(index, 1)
        if index[ip, i]
            @test isapprox(py[ip, i], a * px[ip, i] + b; atol, rtol = atol)
        end
    end
    return nothing
end

@testset "MarkerChain initialization 2D" begin
    nxcell, min_xcell, max_xcell = 3, 2, 5

    for T in (Float64, Float32)
        xv_cpu = collect(range(T(0), T(1); length = 9))
        xv = TA(backend)(xv_cpu)
        elevation = T(0.25)
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
        px, py, index, cell_vertices = host_chain(chain)

        @test size(px) == (max_xcell, length(xv_cpu) - 1)
        # marker storage must inherit the grid/elevation precision, not be forced to
        # Float64 by a bare NaN literal (regression guard for the allocation bug)
        @test eltype(eltype(chain.coords[1])) === T
        @test eltype(eltype(chain.coords[2])) === T
        @test eltype(chain.h_vertices) === T
        @test active_counts(index) == fill(nxcell, length(xv_cpu) - 1)
        @test count(index) == nxcell * (length(xv_cpu) - 1)
        @test all(Array(chain.h_vertices) .≈ elevation)
        @test Array(chain.h_vertices0) == Array(chain.h_vertices)
        assert_chain_invariants(chain)

        for i in axes(index, 2)
            @test all(diff(px[1:nxcell, i]) .> 0)
            @test all(py[1:nxcell, i] .≈ elevation)
            @test all((cell_vertices[i] .< px[1:nxcell, i]) .& (px[1:nxcell, i] .< cell_vertices[i + 1]))
            @test !any(index[(nxcell + 1):end, i])
        end

        topo_y = collect(range(T(0.1), T(0.3); length = length(xv_cpu)))
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, TA(backend)(topo_y))
        _, py, index, _ = host_chain(chain)
        @test isapprox(Array(chain.h_vertices), topo_y; atol = chain_tol(chain), rtol = chain_tol(chain))
        for i in axes(index, 2)
            @test all(py[1:nxcell, i] .≈ topo_y[i])
        end
        assert_chain_invariants(chain)
    end
end

@testset "MarkerChain topography reconstruction 2D" begin
    nxcell, min_xcell, max_xcell = 4, 2, 6
    xv_cpu = collect(range(0.0, 1.0; length = 18))
    xv = TA(backend)(xv_cpu)
    flat_y = 0.35
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, flat_y)

    compute_topography_vertex!(chain)
    @test all(Array(chain.h_vertices) .≈ flat_y)
    assert_chain_invariants(chain)

    a, b = 0.2, 0.15
    topo_y = a .* xv_cpu .+ b
    fill_chain_from_vertices!(chain, TA(backend)(topo_y))
    @test isapprox(Array(chain.h_vertices), topo_y; atol = chain_tol(chain), rtol = chain_tol(chain))
    @test Array(chain.h_vertices0) == Array(chain.h_vertices)
    @test isequal(host_data(chain.coords0[1]), host_data(chain.coords[1]))
    @test isequal(host_data(chain.coords0[2]), host_data(chain.coords[2]))
    assert_chain_invariants(chain)
    assert_markers_on_line(chain, a, b)

    compute_topography_vertex!(chain)
    h_vertices = Array(chain.h_vertices)
    @test isapprox(h_vertices[2:(end - 1)], topo_y[2:(end - 1)]; atol = chain_tol(chain), rtol = chain_tol(chain))
end

@testset "MarkerChain reconstruct non-contiguous index 2D" begin
    # move_particles! can leave interior holes in a cell's occupancy mask;
    # reconstruct_chain_from_vertices! must refill every active slot, not stop at the
    # first inactive one (regression guard for the contiguity assumption)
    xv_cpu = collect(range(0.0, 1.0; length = 6))
    xv = TA(backend)(xv_cpu)
    chain = init_markerchain(backend, 4, 2, 8, xv, 0.3)

    # punch a hole at slot 2 of cell 2 -> [T, F, T, T, ...]
    set_cell_slot!(chain.index, 2, 2, false)
    set_cell_slot!(chain.coords[1], 2, 2, NaN)
    set_cell_slot!(chain.coords[2], 2, 2, NaN)

    JustPIC.reconstruct_chain_from_vertices!(chain)

    px2, py2, index2, cell_vertices = host_chain(chain)
    active2 = index2[:, 2]
    @test count(active2) == 3
    @test all(.!isnan.(px2[active2, 2]))
    @test all(.!isnan.(py2[active2, 2]))
    @test all(cell_vertices[2] .< px2[active2, 2] .< cell_vertices[3])
    @test all(diff(px2[active2, 2]) .> 0)
    assert_chain_invariants(chain)
end

@testset "MarkerChain fill from chain 2D" begin
    nxcell, min_xcell, max_xcell = 3, 1, 4
    xv_cpu = collect(range(0.0, 1.0; length = 8))
    xv = TA(backend)(xv_cpu)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, 0.0)

    topo_x = Float64[]
    topo_y = Float64[]
    flat_y = 0.42
    for i in 1:(length(xv_cpu) - 1)
        dx = xv_cpu[i + 1] - xv_cpu[i]
        push!(topo_x, xv_cpu[i] + dx / 3)
        push!(topo_x, xv_cpu[i] + 2 * dx / 3)
        push!(topo_y, flat_y)
        push!(topo_y, flat_y)
    end

    fill_chain_from_chain!(chain, TA(backend)(topo_x), TA(backend)(topo_y))
    px, py, index, _ = host_chain(chain)
    @test active_counts(index) == fill(2, length(xv_cpu) - 1)
    @test all(Array(chain.h_vertices) .≈ flat_y)
    assert_chain_invariants(chain)

    for i in axes(index, 2)
        j = 2 * i - 1
        @test px[1:2, i] ≈ topo_x[j:(j + 1)]
        @test py[1:2, i] ≈ topo_y[j:(j + 1)]
    end
end

@testset "MarkerChain resample 2D" begin
    xv_cpu = collect(range(0.0, 1.0; length = 17))
    xv = TA(backend)(xv_cpu)

    chain = init_markerchain(backend, 2, 4, 6, xv, 0.2)
    resample!(chain)
    _, py, index, _ = host_chain(chain)
    @test active_counts(index) == fill(4, length(xv_cpu) - 1)
    @test all(py[index] .≈ 0.2)
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, 4, 2, 6, xv, 0.2)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    resample!(chain)
    px, py, index, _ = host_chain(chain)
    @test isequal(px, px0)
    @test isequal(py, py0)
    @test index == index0
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, 2, 4, 6, xv, 0.0)
    a, b = 0.15, 0.1
    fill_chain_from_vertices!(chain, TA(backend)(a .* xv_cpu .+ b))
    resample!(chain)
    _, _, index, _ = host_chain(chain)
    @test active_counts(index) == fill(4, length(xv_cpu) - 1)
    assert_chain_invariants(chain)
    assert_markers_on_line(chain, a, b)
end

@testset "MarkerChain advection 2D" begin
    xv, _, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    nxcell, min_xcell, max_xcell = 3, 2, 6
    elevation = 0.45
    dt = 0.1

    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, 0.0, 0.0)
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test isequal(px, px0)
    @test isequal(py, py0)
    @test index == index0
    assert_chain_invariants(chain)

    vx = 0.05
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, vx, 0.0)
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test isapprox(px[index], px0[index0] .+ vx * dt; atol = chain_tol(chain), rtol = chain_tol(chain))
    @test py[index] ≈ py0[index0]
    @test index == index0
    assert_chain_invariants(chain)

    vy = 0.04
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, 0.0, vy)
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test px[index] ≈ px0[index0]
    @test isapprox(py[index], py0[index0] .+ vy * dt; atol = chain_tol(chain), rtol = chain_tol(chain))
    @test index == index0
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    V = constant_markerchain_velocity(grid_vx, grid_vy, 0.0, vy)
    advect_markerchain!(chain, Euler(), V, grid_vi, dt)
    @test mean(Array(chain.h_vertices)) ≈ mean(h0)
    assert_chain_invariants(chain)
end

@testset "MarkerChain velocity interpolation 2D" begin
    xv, _, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    chain = init_markerchain(backend, 3, 2, 6, xv, 0.45)
    vx, vy = 0.03, -0.02
    V = constant_markerchain_velocity(grid_vx, grid_vy, vx, vy)
    chain_V = ntuple(_ -> cell_array(backend, 0.0, (chain.max_xcell,), size(chain.index)), Val(2))

    interpolate_velocity_to_markerchain!(chain, chain_V, V, grid_vi)
    Vx_chain = host_data(chain_V[1])
    Vy_chain = host_data(chain_V[2])
    _, _, index, _ = host_chain(chain)

    @test all(Vx_chain[index] .≈ vx)
    @test all(Vy_chain[index] .≈ vy)
end

@testset "MarkerChain semi-Lagrangian advection 2D" begin
    xv, yv, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    grid = xv, yv
    nxcell, min_xcell, max_xcell = 3, 2, 6
    elevation = 0.5
    dt = 0.1

    # zero velocity leaves the topography untouched
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    V0 = constant_markerchain_velocity(grid_vx, grid_vy, 0.0, 0.0)
    semilagrangian_advection_markerchain!(chain, RungeKutta2(), V0, grid_vi, grid, dt)
    @test isapprox(Array(chain.h_vertices), h0; atol = chain_tol(chain), rtol = chain_tol(chain))
    assert_chain_invariants(chain)

    # low-level backtracking step: a uniform vertical velocity lifts a flat surface by vy*dt
    vy = 0.03
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    Vup = constant_markerchain_velocity(grid_vx, grid_vy, 0.0, vy)
    JustPIC.semilagrangian_advection!(chain, RungeKutta2(), Vup, grid_vi, grid, dt)
    @test isapprox(Array(chain.h_vertices), fill(elevation + vy * dt, length(xv)); atol = 1.0e-6)

    # the full wrapper reapplies mass conservation, so the mean height is preserved
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    semilagrangian_advection_markerchain!(chain, RungeKutta2(), Vup, grid_vi, grid, dt)
    @test mean(Array(chain.h_vertices)) ≈ mean(h0)
    assert_chain_invariants(chain)

    # Float32 grid: the SL path must recast its grids/integrator and keep the topography
    # precision (guards against Float64 promotion, which breaks Metal)
    n32 = length(xv)
    xv32 = range(0.0f0, 1.0f0; length = n32)
    yv32 = range(0.0f0, 1.0f0; length = n32)
    dx32 = xv32[2] - xv32[1]
    dy32 = yv32[2] - yv32[1]
    xc32 = range(dx32 / 2, 1.0f0 - dx32 / 2; length = n32 - 1)
    yc32 = range(dy32 / 2, 1.0f0 - dy32 / 2; length = n32 - 1)
    grid_vx32 = xv32, markerchain_expand_range(yc32)
    grid_vy32 = markerchain_expand_range(xc32), yv32
    grid_vi32 = grid_vx32, grid_vy32
    grid32 = xv32, yv32
    Vup32 = constant_markerchain_velocity(grid_vx32, grid_vy32, 0.0f0, Float32(vy))
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv32, 0.5f0)
    JustPIC.semilagrangian_advection!(chain, RungeKutta2(), Vup32, grid_vi32, grid32, 0.1f0)
    h32 = Array(chain.h_vertices)
    @test eltype(h32) === Float32
    @test all(isapprox.(h32[2:(end - 1)], 0.5f0 + Float32(vy) * 0.1f0; atol = 1.0f-5))
end

@testset "MarkerChain multi-step advection pipeline 2D" begin
    # A flat interface stays single-valued under a uniform horizontal flow, so the full
    # advect → move → resample → reconstruct pipeline can be driven for many steps. The
    # markers drift several cells (0.05 * 0.1 * 50 ≈ 4 cells), exercising move_particles!
    # and the resampling that refills the emptied leading cells.
    xv, yv, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    elevation = 0.5
    V = constant_markerchain_velocity(grid_vx, grid_vy, 0.05, 0.0)

    chain = init_markerchain(backend, 6, 3, 12, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    for _ in 1:50
        advect_markerchain!(chain, RungeKutta2(), V, grid_vi, 0.1)
    end

    h = Array(chain.h_vertices)
    @test all(isfinite, h)
    @test all(h .≈ elevation)
    @test mean(h) ≈ mean(h0)
    assert_chain_invariants(chain)
end

@testset "MarkerChain rock fraction 2D" begin
    n = 9
    xv = range(0.0, 1.0; length = n)
    yv = range(0.0, 1.0; length = n)
    dx = xv[2] - xv[1]
    dy = yv[2] - yv[1]
    xvi = TA(backend)(collect(xv)), TA(backend)(collect(yv))
    dxi = dx, dy

    make_ratios() = (
        center = TA(backend)(zeros(n - 1, n - 1)),
        vertex = TA(backend)(zeros(n, n)),
        Vx = TA(backend)(zeros(n, n - 1)),
        Vy = TA(backend)(zeros(n - 1, n)),
    )

    chain = init_markerchain(backend, 3, 2, 6, xv, 0.5)

    # chain above the whole domain: every control volume is fully rock
    copyto!(chain.h_vertices, TA(backend)(fill(2.0, n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 1)
    end

    # chain below the whole domain: every control volume is fully air
    copyto!(chain.h_vertices, TA(backend)(fill(-1.0, n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 0)
    end

    # flat interface straddling one row of cell centres: exact area fraction
    h = 0.42
    copyto!(chain.h_vertices, TA(backend)(fill(h, n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    center = Array(ratios.center)
    for j in axes(center, 2)
        y_bottom = yv[j]
        y_top = yv[j] + dy
        expected = if h ≥ y_top
            1.0
        elseif h ≤ y_bottom
            0.0
        else
            (h - y_bottom) / dy
        end
        @test all(center[:, j] .≈ expected)
    end
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        data = Array(field)
        @test all(0 .≤ data .≤ 1)
    end
end

@testset "MarkerChain slope smoothing 2D" begin
    n = 7
    xv = TA(backend)(collect(range(0.0, 1.0; length = n)))
    chain = init_markerchain(backend, 3, 2, 6, xv, 0.0)

    # a single steep interior spike is redistributed onto its neighbours
    H = 0.3
    spike = zeros(n)
    spike[4] = H
    copyto!(chain.h_vertices, TA(backend)(spike))
    JustPIC.smooth_slopes!(chain, deg2rad(5.0))
    expected = [0.0, 0.0, 0.25H, 0.5H, 0.25H, 0.0, 0.0]
    @test isapprox(Array(chain.h_vertices), expected; atol = 1.0e-12)

    # a gentle slope stays below the limiter and is left untouched
    a = 0.05
    linear = a .* collect(range(0.0, 1.0; length = n))
    copyto!(chain.h_vertices, TA(backend)(linear))
    JustPIC.smooth_slopes!(chain, deg2rad(45.0))
    @test Array(chain.h_vertices) ≈ linear

    # fewer than three vertices: no-op
    xv2 = TA(backend)(collect(range(0.0, 1.0; length = 2)))
    chain2 = init_markerchain(backend, 1, 1, 2, xv2, 0.0)
    copyto!(chain2.h_vertices, TA(backend)([0.1, 0.9]))
    JustPIC.smooth_slopes!(chain2, deg2rad(5.0))
    @test Array(chain2.h_vertices) == [0.1, 0.9]
end

@testset "MarkerChain interpolation helpers 2D" begin
    # linear interpolation kernel is exact on a line
    @test JustPIC._interp1D(0.5, 0.0, 1.0, 0.0, 2.0) ≈ 1.0
    @test JustPIC._interp1D(0.25, 0.0, 1.0, 3.0, 7.0) ≈ 4.0

    # interp1D_extremas interpolates inside and extrapolates past both ends (y = 2x)
    x = [0.0, 1.0, 2.0]
    y = [0.0, 2.0, 4.0]
    @test JustPIC.interp1D_extremas(0.5, x, y) ≈ 1.0
    @test JustPIC.interp1D_extremas(1.5, x, y) ≈ 3.0
    @test JustPIC.interp1D_extremas(-0.5, x, y) ≈ -1.0
    @test JustPIC.interp1D_extremas(2.5, x, y) ≈ 5.0

    # isdistorded flags gaps larger than 2*dx_ideal, tolerates trailing NaNs
    @test !JustPIC.isdistorded([0.1, 0.2, 0.3], 0.1)
    @test JustPIC.isdistorded([0.1, 0.5], 0.1)
    @test !JustPIC.isdistorded([0.1, 0.2, NaN, NaN], 0.1)

    # first_last_particle_incell bins an interior cell of an open polyline
    topo_x = [0.5, 1.5, 1.6, 2.5]
    cell_vertices = [0.0, 1.0, 2.0, 3.0]
    @test JustPIC.first_last_particle_incell(topo_x, cell_vertices, 2) == (2, 3)
end
