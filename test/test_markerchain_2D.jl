const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif BACKEND_NAME == "CUDA"
    using CUDA
    CUDA.allowscalar(true)
elseif BACKEND_NAME == "Metal"
    using Metal
    Metal.allowscalar(true)
end

using JustPIC, Test, Statistics
using CellArrays: field
using GridGeometryUtils: Point, Segment
import KernelAbstractions: CPU

const backend = @static if BACKEND_NAME == "AMDGPU"
    AMDGPU.ROCBackend
elseif BACKEND_NAME == "CUDA"
    CUDA.CUDABackend
elseif BACKEND_NAME == "Metal"
    Metal.MetalBackend
else
    CPU
end

# Metal has no Float64; JULIA_JUSTPIC_PRECISION=Float32 runs the same paths on CPU
const FT = if BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32"
    Float32
else
    Float64
end
const TEST_PRECISIONS = FT === Float32 ? (Float32,) : (Float64, Float32)

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
chain_mean(h) = (sum(h) - (first(h) + last(h)) / 2) / (length(h) - 1)

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
    xv = range(FT(0), FT(1); length = n)
    yv = range(FT(0), FT(1); length = n)
    dx = xv[2] - xv[1]
    dy = yv[2] - yv[1]
    xc = range(dx / 2, FT(1) - dx / 2; length = n - 1)
    yc = range(dy / 2, FT(1) - dy / 2; length = n - 1)
    grid_vx = xv, markerchain_expand_range(yc)
    grid_vy = markerchain_expand_range(xc), yv
    return xv, yv, grid_vx, grid_vy
end

# strictly increasing, with cell widths growing left to right by a factor of `ratio`
function markerchain_graded_grid(n, ratio = FT(4))
    widths = FT[1 + (ratio - 1) * (i - 1) / (n - 2) for i in 1:(n - 1)]
    xv = cumsum(vcat(zero(FT), widths))
    return xv ./ last(xv)
end

function markerchain_expand_vector(x)
    return vcat(x[1] - (x[2] - x[1]), x, x[end] + (x[end] - x[end - 1]))
end

# Staggered velocity grids on graded meshes refined in opposite directions, so a kernel
# that reuses the x-component spacing for the y component gives a wrong answer.
function markerchain_refined_velocity_grid(n = 17)
    xv = markerchain_graded_grid(n)
    yv = reverse(one(FT) .- markerchain_graded_grid(n))
    xc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2
    yc = (yv[1:(end - 1)] .+ yv[2:end]) ./ 2
    grid_vx = TA(backend)(xv), TA(backend)(markerchain_expand_vector(yc))
    grid_vy = TA(backend)(markerchain_expand_vector(xc)), TA(backend)(yv)
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

    for T in TEST_PRECISIONS
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
        @test chain.coords0[1].data !== chain.coords[1].data
        @test chain.coords0[2].data !== chain.coords[2].data
        previous_x = host_data(chain.coords0[1])
        set_cell_slot!(chain.coords[1], 1, 1, px[1, 1] + eps(T))
        @test isequal(host_data(chain.coords0[1]), previous_x)
        set_cell_slot!(chain.coords[1], 1, 1, px[1, 1])
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
            expected_y = range(topo_y[i], topo_y[i + 1]; length = nxcell + 2)[2:(end - 1)]
            @test py[1:nxcell, i] ≈ expected_y
        end
        compute_topography_vertex!(chain)
        @test isapprox(Array(chain.h_vertices), topo_y; atol = chain_tol(chain), rtol = chain_tol(chain))
        assert_chain_invariants(chain)
    end

    # the initialization kernel indexes both the grid and a vertex-wise elevation, so host
    # inputs have to reach the device
    host_xv = collect(range(FT(0), FT(1); length = 7))
    host_elevation = collect(range(FT(0.1), FT(0.3); length = 7))
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, host_xv, host_elevation)
    @test chain.cell_vertices isa TA(backend)
    @test chain.h_vertices isa TA(backend)
    @test chain.h_vertices !== chain.h_vertices0
    @test isapprox(
        Array(chain.h_vertices), host_elevation; atol = chain_tol(chain), rtol = chain_tol(chain)
    )
    assert_chain_invariants(chain)

    @static if BACKEND_NAME == "CPU"
        xv = collect(range(0.0, 1.0; length = 6))
        original = init_markerchain(CPU, 3, 2, 5, xv, 0.4)
        reconstructed = MarkerChain(original.coords, original.index, xv, 2, 5)
        @test Array(reconstructed.h_vertices) ≈ fill(0.4, length(xv))
        @test reconstructed.coords0[1].data !== reconstructed.coords[1].data
    end

    # refined grids are supported; the grid only has to be strictly increasing
    nonuniform_xv = TA(backend)(FT[0, 0.2, 0.5, 1])
    refined = init_markerchain(backend, 2, 1, 4, nonuniform_xv, FT(0.4))
    assert_chain_invariants(refined)
    @test_throws ArgumentError init_markerchain(
        backend, 2, 1, 4, TA(backend)(FT[0, 0.5, 0.2, 1]), FT(0.4)
    )
    @test_throws ArgumentError init_markerchain(
        backend, 2, 1, 4, TA(backend)(FT[0, 0.5, 0.5, 1]), FT(0.4)
    )
    @test_throws ArgumentError init_markerchain(
        backend, 2, 1, 4, TA(backend)(FT[0.0]), FT(0.4)
    )
end

@testset "MarkerChain topography reconstruction 2D" begin
    nxcell, min_xcell, max_xcell = 4, 2, 6
    xv_cpu = collect(range(FT(0), FT(1); length = 18))
    xv = TA(backend)(xv_cpu)
    flat_y = FT(0.35)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, flat_y)

    compute_topography_vertex!(chain)
    @test all(Array(chain.h_vertices) .≈ flat_y)
    assert_chain_invariants(chain)

    a, b = FT(0.2), FT(0.15)
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
    @test isapprox(h_vertices, topo_y; atol = chain_tol(chain), rtol = chain_tol(chain))
end

@testset "MarkerChain reconstruct non-contiguous index 2D" begin
    # move_particles! can leave interior holes in a cell's occupancy mask;
    # reconstruct_chain_from_vertices! must refill every active slot, not stop at the
    # first inactive one (regression guard for the contiguity assumption)
    xv_cpu = collect(range(FT(0), FT(1); length = 6))
    xv = TA(backend)(xv_cpu)
    chain = init_markerchain(backend, 4, 2, 8, xv, FT(0.3))

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

@testset "MarkerChain multi-cell movement 2D" begin
    xv_cpu = collect(range(FT(0), FT(1); length = 9))
    chain = init_markerchain(backend, 1, 1, 4, TA(backend)(xv_cpu), FT(0.4))

    # Two distant sources converge on cell 4. This exercises both arbitrary-distance
    # jumps and collision-free destination writes on threaded/GPU backends.
    set_cell_slot!(chain.coords[1], 1, 1, FT(0.4))
    set_cell_slot!(chain.coords[1], 1, 7, FT(0.46))
    move_particles!(chain)

    _, _, index, _ = host_chain(chain)
    @test count(index) == length(xv_cpu) - 1
    @test active_counts(index) == [0, 1, 1, 3, 1, 1, 0, 1]
    assert_chain_invariants(chain)
end

@testset "MarkerChain fill from chain 2D" begin
    nxcell, min_xcell, max_xcell = 3, 1, 4
    xv_cpu = collect(range(FT(0), FT(1); length = 8))
    xv = TA(backend)(xv_cpu)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, FT(0))

    topo_x = FT[]
    topo_y = FT[]
    flat_y = FT(0.42)
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
    xv_cpu = collect(range(FT(0), FT(1); length = 17))
    xv = TA(backend)(xv_cpu)

    chain = init_markerchain(backend, 2, 4, 6, xv, FT(0.2))
    resample!(chain)
    _, py, index, _ = host_chain(chain)
    @test active_counts(index) == fill(4, length(xv_cpu) - 1)
    @test all(py[index] .≈ FT(0.2))
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, 4, 2, 6, xv, FT(0.2))
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    resample!(chain)
    px, py, index, _ = host_chain(chain)
    @test isequal(px, px0)
    @test isequal(py, py0)
    @test index == index0
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, 2, 4, 6, xv, FT(0))
    a, b = FT(0.15), FT(0.1)
    fill_chain_from_vertices!(chain, TA(backend)(a .* xv_cpu .+ b))
    resample!(chain)
    _, _, index, _ = host_chain(chain)
    @test active_counts(index) == fill(4, length(xv_cpu) - 1)
    assert_chain_invariants(chain)
    assert_markers_on_line(chain, a, b)

    chain = init_markerchain(backend, 4, 2, 6, xv, FT(0.2))
    for ip in 1:4
        set_cell_slot!(chain.index, ip, 1, false)
        set_cell_slot!(chain.coords[1], ip, 1, NaN)
        set_cell_slot!(chain.coords[2], ip, 1, NaN)
    end
    resample!(chain)
    px, py, index, _ = host_chain(chain)
    @test count(index[:, 1]) == 2
    @test all(isfinite, px[index])
    @test all(isfinite, py[index])
    assert_chain_invariants(chain)
end

@testset "MarkerChain advection 2D" begin
    xv, _, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    nxcell, min_xcell, max_xcell = 3, 2, 6
    elevation = FT(0.45)
    dt = FT(0.1)

    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), FT(0))
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test isequal(px, px0)
    @test isequal(py, py0)
    @test index == index0
    assert_chain_invariants(chain)

    vx = FT(0.05)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, vx, FT(0))
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test isapprox(px[index], px0[index0] .+ vx * dt; atol = chain_tol(chain), rtol = chain_tol(chain))
    @test py[index] ≈ py0[index0]
    @test index == index0
    assert_chain_invariants(chain)

    vy = FT(0.04)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    px0, py0, index0, _ = host_chain(chain)
    px0, py0, index0 = copy(px0), copy(py0), copy(index0)
    V = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), vy)
    advection!(chain, Euler(), V, grid_vi, dt)
    px, py, index, _ = host_chain(chain)
    @test px[index] ≈ px0[index0]
    @test isapprox(py[index], py0[index0] .+ vy * dt; atol = chain_tol(chain), rtol = chain_tol(chain))
    @test index == index0
    assert_chain_invariants(chain)

    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    V = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), vy)
    advect_markerchain!(chain, Euler(), V, grid_vi, dt)
    @test chain_mean(Array(chain.h_vertices)) ≈ chain_mean(h0)
    assert_chain_invariants(chain)

    profile = @. FT(0.35) + FT(0.08) * sin(FT(2pi) * xv) + FT(0.03) * xv^2
    chain = init_markerchain(backend, 4, 2, 8, xv, FT(0))
    fill_chain_from_vertices!(chain, TA(backend)(collect(profile)))
    h0 = copy(Array(chain.h_vertices))
    V0 = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), FT(0))
    for _ in 1:20
        advect_markerchain!(chain, RungeKutta2(), V0, grid_vi, dt)
    end
    h = Array(chain.h_vertices)
    tol = FT === Float32 ? 2.0f-4 : 1.0e-9
    @test isapprox(h, h0; atol = tol, rtol = tol)
    @test isapprox(chain_mean(h), chain_mean(h0); atol = tol, rtol = tol)
    assert_chain_invariants(chain)
end

@testset "MarkerChain velocity interpolation 2D" begin
    xv, _, grid_vx, grid_vy = markerchain_velocity_grid()
    grid_vi = grid_vx, grid_vy
    chain = init_markerchain(backend, 3, 2, 6, xv, FT(0.45))
    vx, vy = FT(0.03), FT(-0.02)
    V = constant_markerchain_velocity(grid_vx, grid_vy, vx, vy)
    chain_V = ntuple(_ -> cell_array(backend, FT(0), (chain.max_xcell,), size(chain.index)), Val(2))

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
    elevation = FT(0.5)
    dt = FT(0.1)

    # zero velocity leaves the topography untouched
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    V0 = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), FT(0))
    semilagrangian_advection_markerchain!(chain, RungeKutta2(), V0, grid_vi, grid, dt)
    @test isapprox(Array(chain.h_vertices), h0; atol = chain_tol(chain), rtol = chain_tol(chain))
    assert_chain_invariants(chain)

    # low-level backtracking step: a uniform vertical velocity lifts a flat surface by vy*dt
    vy = FT(0.03)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    Vup = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), vy)
    JustPIC.semilagrangian_advection!(chain, RungeKutta2(), Vup, grid_vi, grid, dt)
    @test isapprox(Array(chain.h_vertices), fill(elevation + vy * dt, length(xv)); atol = (FT === Float32 ? 1.0f-5 : 1.0e-6))

    # the full wrapper reapplies mass conservation, so the mean height is preserved
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    semilagrangian_advection_markerchain!(chain, RungeKutta2(), Vup, grid_vi, grid, dt)
    @test chain_mean(Array(chain.h_vertices)) ≈ chain_mean(h0)
    assert_chain_invariants(chain)

    slope = FT(0.1)
    profile = @. FT(0.3) + slope * xv
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, FT(0))
    fill_chain_from_vertices!(chain, TA(backend)(collect(profile)))
    vx = FT(0.05)
    Vright = constant_markerchain_velocity(grid_vx, grid_vy, vx, FT(0))
    JustPIC.semilagrangian_advection!(chain, RungeKutta2(), Vright, grid_vi, grid, dt)
    expected = @. profile - slope * vx * dt
    @test isapprox(
        Array(chain.h_vertices)[2:(end - 1)], expected[2:(end - 1)];
        atol = (FT === Float32 ? 1.0f-5 : 1.0e-12), rtol = chain_tol(chain)
    )

    steep = fill(FT(0.4), length(xv))
    steep[2] += FT(0.2)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, FT(0))
    fill_chain_from_vertices!(chain, TA(backend)(steep))
    mean0 = chain_mean(steep)
    semilagrangian_advection_markerchain!(
        chain, RungeKutta2(), V0, grid_vi, grid, dt; max_slope_angle = 5
    )
    @test isapprox(
        chain_mean(Array(chain.h_vertices)), mean0;
        atol = (FT === Float32 ? 1.0f-5 : 1.0e-12), rtol = chain_tol(chain)
    )

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
    elevation = FT(0.5)
    V = constant_markerchain_velocity(grid_vx, grid_vy, FT(0.05), FT(0))

    chain = init_markerchain(backend, 6, 3, 12, xv, elevation)
    h0 = copy(Array(chain.h_vertices))
    for _ in 1:50
        advect_markerchain!(chain, RungeKutta2(), V, grid_vi, FT(0.1))
    end

    h = Array(chain.h_vertices)
    @test all(isfinite, h)
    @test all(h .≈ elevation)
    @test chain_mean(h) ≈ chain_mean(h0)
    assert_chain_invariants(chain)
end

@testset "MarkerChain cell rock area 2D" begin
    for T in TEST_PRECISIONS
        min_corner = T(2), T(-1)
        di = T(4), T(2)
        r = JustPIC.rectangle_from_min_corner(min_corner, di)

        @test r.origin == JustPIC.Point(T(4), T(0))

        horizontal(y) = JustPIC.Segment(
            JustPIC.Point(T(2), T(y)), JustPIC.Point(T(6), T(y))
        )
        @test JustPIC.cell_rock_area(horizontal(-1), r) == T(0)
        @test JustPIC.cell_rock_area(horizontal(0), r) == T(0.5)
        @test JustPIC.cell_rock_area(horizontal(1), r) == T(1)

        rising = JustPIC.Segment(JustPIC.Point(T(2), T(-2)), JustPIC.Point(T(6), T(0)))
        falling = JustPIC.Segment(JustPIC.Point(T(2), T(0.5)), JustPIC.Point(T(6), T(-0.5)))
        @test JustPIC.cell_rock_area(rising, r) == T(0.125)
        @test JustPIC.cell_rock_area(falling, r) == T(0.5)

        clipped = JustPIC.clip_chain_to_cell(rising, r)
        clipped_reversed = JustPIC.Segment(clipped.p2, clipped.p1)
        @test JustPIC.intersecting_area(clipped, r) / JustPIC.area(r) == T(0.125)
        @test JustPIC.intersecting_area(clipped_reversed, r) / JustPIC.area(r) == T(0.875)
    end
end

@testset "MarkerChain rock fraction 2D" begin
    # A 2x2 cell spanning x ∈ [1, 3], y ∈ [2, 4], repeated at offsets of many cell widths:
    # the geometry is evaluated relative to the cell, so the answers must not degrade as the
    # cell moves away from the coordinate origin.
    for T in TEST_PRECISIONS, offset in (T(0), T(1.0e2), T(1.0e4))
        r = JustPIC.rectangle_from_min_corner((T(1) + offset, T(2) + offset), (T(2), T(2)))
        chain_area(y1, y2) = JustPIC.cell_rock_area(
            Segment(Point(T(1) + offset, y1 + offset), Point(T(3) + offset, y2 + offset)), r
        )

        @test chain_area(T(3), T(3)) ≈ T(0.5)          # flat, mid-height
        @test chain_area(T(1), T(3)) ≈ T(0.125)        # in through the floor, out at the NE corner
        @test chain_area(T(3), T(1)) ≈ T(0.125)        # mirror image
        @test chain_area(T(1), T(5)) ≈ T(0.5)          # in through the floor, out through the ceiling
        @test chain_area(T(5), T(1)) ≈ T(0.5)          # mirror image
        @test chain_area(T(2), T(4)) ≈ T(0.5)          # corner to corner
        @test chain_area(T(2), T(2)) ≈ T(0)            # flat, on the floor
        @test chain_area(T(4), T(4)) ≈ T(1)            # flat, on the ceiling
        @test chain_area(T(-10), T(-10)) ≈ T(0)        # far below
        @test chain_area(T(10), T(10)) ≈ T(1)          # far above
        @test chain_area(T(2), T(3)) ≈ T(0.25)         # floor corner to the right edge
        @test chain_area(T(3), T(4)) ≈ T(0.75)         # left edge to the ceiling corner
    end

    n = 9
    xv = range(FT(0), FT(1); length = n)
    yv = range(FT(0), FT(1); length = n)
    dx = xv[2] - xv[1]
    dy = yv[2] - yv[1]
    xvi = TA(backend)(collect(xv)), TA(backend)(collect(yv))
    dxi = dx, dy

    make_ratios() = (
        center = TA(backend)(zeros(FT, n - 1, n - 1)),
        vertex = TA(backend)(zeros(FT, n, n)),
        Vx = TA(backend)(zeros(FT, n, n - 1)),
        Vy = TA(backend)(zeros(FT, n - 1, n)),
    )

    chain = init_markerchain(backend, 3, 2, 6, xv, FT(0.5))

    # chain above the whole domain: every control volume is fully rock
    copyto!(chain.h_vertices, TA(backend)(fill(FT(2), n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 1)
    end

    # chain below the whole domain: every control volume is fully air
    copyto!(chain.h_vertices, TA(backend)(fill(FT(-1), n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 0)
    end

    # flat interface straddling one row of cell centres: exact area fraction
    h = FT(0.42)
    copyto!(chain.h_vertices, TA(backend)(fill(h, n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    rock_fraction(y_bottom, height) = clamp(
        (h - y_bottom) / height, zero(FT), one(FT)
    )

    center = Array(ratios.center)
    for j in axes(center, 2)
        expected = rock_fraction(yv[j], dy)
        @test all(center[:, j] .≈ expected)
    end

    vx = Array(ratios.Vx)
    for j in axes(vx, 2)
        expected = rock_fraction(yv[j], dy)
        @test all(vx[:, j] .≈ expected)
    end

    vertex = Array(ratios.vertex)
    vy = Array(ratios.Vy)
    for j in axes(vertex, 2)
        expected = if j == firstindex(yv)
            rock_fraction(yv[j], dy / 2)
        elseif j == lastindex(yv)
            rock_fraction(yv[j] - dy / 2, dy / 2)
        else
            (
                rock_fraction(yv[j] - dy / 2, dy / 2) +
                    rock_fraction(yv[j], dy / 2)
            ) / 2
        end
        @test all(vertex[:, j] .≈ expected)
        @test all(vy[:, j] .≈ expected)
    end

    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        data = Array(field)
        @test all(0 .≤ data .≤ 1)
    end

    topo = @. FT(0.25) + FT(0.3) * xv
    copyto!(chain.h_vertices, TA(backend)(collect(topo)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    @test isapprox(Array(ratios.vertex)[2, 3], FT(0.8); atol = (FT === Float32 ? 1.0f-5 : 1.0e-12))

    # the same flat interface on a grid sitting hundreds of cell widths from the origin
    shift = FT(500)
    xv_far = range(shift, shift + FT(1); length = n)
    xvi_far = TA(backend)(collect(xv_far)), TA(backend)(collect(range(shift, shift + FT(1); length = n)))
    chain_far = init_markerchain(backend, 3, 2, 6, xv_far, shift + FT(0.42))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain_far, xvi_far, dxi)
    center = Array(ratios.center)
    for j in axes(center, 2)
        y_bottom = xv_far[j]
        expected = clamp((shift + FT(0.42) - y_bottom) / dy, FT(0), FT(1))
        @test all(isapprox.(center[:, j], expected; atol = (FT === Float32 ? 1.0f-2 : 1.0e-6)))
    end
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        data = Array(field)
        @test all(isfinite, data)
        @test all(0 .≤ data .≤ 1)
    end
end

@testset "MarkerChain slope smoothing 2D" begin
    n = 7
    xv = TA(backend)(collect(range(FT(0), FT(1); length = n)))
    chain = init_markerchain(backend, 3, 2, 6, xv, FT(0))

    # a single steep interior spike is redistributed onto its neighbours
    H = FT(0.3)
    spike = zeros(FT, n)
    spike[4] = H
    copyto!(chain.h_vertices, TA(backend)(spike))
    JustPIC.smooth_slopes!(chain, FT(deg2rad(5)))
    expected = FT[0, 0, 0.25H, 0.5H, 0.25H, 0, 0]
    @test isapprox(Array(chain.h_vertices), expected; atol = (FT === Float32 ? 1.0f-6 : 1.0e-12))

    # a gentle slope stays below the limiter and is left untouched
    a = FT(0.05)
    linear = a .* collect(range(FT(0), FT(1); length = n))
    copyto!(chain.h_vertices, TA(backend)(linear))
    JustPIC.smooth_slopes!(chain, FT(deg2rad(45)))
    @test Array(chain.h_vertices) ≈ linear

    # fewer than three vertices: no-op
    xv2 = TA(backend)(collect(range(FT(0), FT(1); length = 2)))
    chain2 = init_markerchain(backend, 1, 1, 2, xv2, FT(0))
    copyto!(chain2.h_vertices, TA(backend)(FT[0.1, 0.9]))
    JustPIC.smooth_slopes!(chain2, FT(deg2rad(5)))
    @test Array(chain2.h_vertices) == FT[0.1, 0.9]
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
    @test JustPIC.first_last_particle_incell([0.1, 0.2], cell_vertices, 2) == (1, 0)
end

@testset "MarkerChain refined grid initialization 2D" begin
    nxcell, min_xcell, max_xcell = 3, 2, 6
    xv_cpu = markerchain_graded_grid(9)
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, TA(backend)(xv_cpu), FT(0.4))

    px, _, index, _ = host_chain(chain)
    @test active_counts(index) == fill(nxcell, length(xv_cpu) - 1)
    assert_chain_invariants(chain)

    # each column is populated according to its own width
    for i in axes(index, 2)
        dx = xv_cpu[i + 1] - xv_cpu[i]
        expected = xv_cpu[i] .+ dx .* (1:nxcell) ./ (nxcell + 1)
        @test px[1:nxcell, i] ≈ expected
    end
end

@testset "MarkerChain cell_length 2D" begin
    xv_cpu = markerchain_graded_grid(9)
    refined = init_markerchain(backend, 2, 2, 4, TA(backend)(xv_cpu), FT(0.4))
    for i in 1:(length(xv_cpu) - 1)
        @test cell_length(refined, i) ≈ xv_cpu[i + 1] - xv_cpu[i]
    end
    @test_throws "cell_vertices are not uniformly spaced" cell_length(refined)
    @test_throws "cell_length(chain, i)" cell_length(refined)

    xv_uniform = collect(range(FT(0), FT(1); length = 9))
    dx = xv_uniform[2] - xv_uniform[1]
    uniform = init_markerchain(backend, 2, 2, 4, TA(backend)(xv_uniform), FT(0.4))
    @test cell_length(uniform) ≈ dx
    @test all(cell_length(uniform, i) ≈ dx for i in 1:(length(xv_uniform) - 1))

    ranged = init_markerchain(backend, 2, 2, 4, range(FT(0), FT(1); length = 9), FT(0.4))
    @test cell_length(ranged) ≈ dx
    @test cell_length(ranged, 1) ≈ dx
end

@testset "MarkerChain refined grid movement 2D" begin
    xv_cpu = markerchain_graded_grid(9)
    ncells = length(xv_cpu) - 1
    chain = init_markerchain(backend, 1, 1, 4, TA(backend)(xv_cpu), FT(0.4))

    # a marker crossing several columns of differing width lands in the right one, and one
    # pushed past the right edge of the domain is deleted
    set_cell_slot!(chain.coords[1], 1, 1, (xv_cpu[6] + xv_cpu[7]) / 2)
    set_cell_slot!(chain.coords[1], 1, ncells, FT(1.5))
    move_particles!(chain)

    _, _, index, _ = host_chain(chain)
    counts = active_counts(index)
    @test counts[1] == 0
    @test counts[6] == 2
    @test counts[ncells] == 0
    @test count(index) == ncells - 1
    assert_chain_invariants(chain)
end

@testset "MarkerChain refined grid resample 2D" begin
    xv_cpu = markerchain_graded_grid(9)
    ncells = length(xv_cpu) - 1
    min_xcell, max_xcell = 4, 6
    # the widest column is several times the narrowest, so a single global spacing cannot
    # satisfy both
    @test (xv_cpu[end] - xv_cpu[end - 1]) > 3 * (xv_cpu[2] - xv_cpu[1])

    chain = init_markerchain(backend, 2, min_xcell, max_xcell, TA(backend)(xv_cpu), FT(0.2))
    resample!(chain)
    px, py, index, _ = host_chain(chain)
    @test active_counts(index) == fill(min_xcell, ncells)
    @test all(py[index] .≈ FT(0.2))
    assert_chain_invariants(chain)

    for i in (1, ncells)
        dx = (xv_cpu[i + 1] - xv_cpu[i]) / (min_xcell + 1)
        @test diff(px[1:min_xcell, i]) ≈ fill(dx, min_xcell - 1)
    end
end

@testset "MarkerChain refined grid velocity interpolation 2D" begin
    xv, _, grid_vx, grid_vy = markerchain_refined_velocity_grid()
    grid_vi = grid_vx, grid_vy
    chain = init_markerchain(backend, 3, 2, 6, TA(backend)(xv), FT(0.45))
    chain_V = ntuple(_ -> cell_array(backend, FT(0), (chain.max_xcell,), size(chain.index)), Val(2))

    vx_field(x, y) = FT(0.1) * x + FT(0.2) * y
    vy_field(x, y) = FT(-0.3) * x + FT(0.05) * y
    gvx, gvy = Array.(grid_vx), Array.(grid_vy)
    V = (
        TA(backend)([vx_field(x, y) for x in gvx[1], y in gvx[2]]),
        TA(backend)([vy_field(x, y) for x in gvy[1], y in gvy[2]]),
    )

    interpolate_velocity_to_markerchain!(chain, chain_V, V, grid_vi)
    px, py, index, _ = host_chain(chain)
    Vx_chain = host_data(chain_V[1])
    Vy_chain = host_data(chain_V[2])

    # bilinear interpolation of a linear field is exact only when each component is
    # normalized by the spacing of its own staggered grid
    atol = FT === Float32 ? 1.0f-5 : 1.0e-12
    @test all(isapprox.(Vx_chain[index], vx_field.(px[index], py[index]); atol, rtol = atol))
    @test all(isapprox.(Vy_chain[index], vy_field.(px[index], py[index]); atol, rtol = atol))
end

@testset "MarkerChain refined grid advection 2D" begin
    xv, yv, grid_vx, grid_vy = markerchain_refined_velocity_grid()
    grid_vi = grid_vx, grid_vy
    grid = TA(backend)(xv), TA(backend)(yv)
    nxcell, min_xcell, max_xcell = 3, 2, 6
    elevation = FT(0.45)
    dt = FT(0.1)
    vx, vy = FT(0.03), FT(-0.02)
    V = constant_markerchain_velocity(grid_vx, grid_vy, vx, vy)

    reference = init_markerchain(backend, nxcell, min_xcell, max_xcell, TA(backend)(xv), elevation)
    px0, py0, index0, _ = host_chain(reference)

    for method in (Euler(), RungeKutta2(), RungeKutta4())
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, TA(backend)(xv), elevation)
        advection!(chain, method, V, grid_vi, dt)
        px, py, index, _ = host_chain(chain)
        atol = chain_tol(chain)
        @test index == index0
        @test all(isapprox.(px[index], px0[index0] .+ vx * dt; atol, rtol = atol))
        @test all(isapprox.(py[index], py0[index0] .+ vy * dt; atol, rtol = atol))
    end

    # a uniform vertical velocity lifts a flat surface by vy*dt
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, TA(backend)(xv), elevation)
    Vup = constant_markerchain_velocity(grid_vx, grid_vy, FT(0), vy)
    JustPIC.semilagrangian_advection!(chain, RungeKutta2(), Vup, grid_vi, grid, dt)
    @test isapprox(
        Array(chain.h_vertices), fill(elevation + vy * dt, length(xv));
        atol = (FT === Float32 ? 1.0f-5 : 1.0e-6)
    )

    # both wrappers conserve the mean height
    Vright = constant_markerchain_velocity(grid_vx, grid_vy, FT(0.05), FT(0))
    for advect! in (
            (c) -> advect_markerchain!(c, RungeKutta2(), Vright, grid_vi, FT(0.05)),
            (c) -> semilagrangian_advection_markerchain!(c, RungeKutta2(), Vright, grid_vi, grid, FT(0.05)),
        )
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, TA(backend)(xv), elevation)
        h0 = chain_mean(Array(chain.h_vertices))
        for _ in 1:10
            advect!(chain)
        end
        h = Array(chain.h_vertices)
        @test all(isfinite, h)
        @test isapprox(chain_mean(h), h0; atol = chain_tol(chain), rtol = chain_tol(chain))
        assert_chain_invariants(chain)
    end
end

@testset "MarkerChain refined grid rock fraction 2D" begin
    n = 9
    xv = markerchain_graded_grid(n)
    yv = reverse(one(FT) .- markerchain_graded_grid(n))
    dx = diff(xv)
    dy = diff(yv)
    xvi = TA(backend)(xv), TA(backend)(yv)
    dxi = TA(backend)(dx), TA(backend)(dy)

    make_ratios() = (
        center = TA(backend)(zeros(FT, n - 1, n - 1)),
        vertex = TA(backend)(zeros(FT, n, n)),
        Vx = TA(backend)(zeros(FT, n, n - 1)),
        Vy = TA(backend)(zeros(FT, n - 1, n)),
    )

    chain = init_markerchain(backend, 3, 2, 6, TA(backend)(xv), FT(0.5))

    copyto!(chain.h_vertices, TA(backend)(fill(FT(2), n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 1)
    end

    copyto!(chain.h_vertices, TA(backend)(fill(FT(-1), n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(Array(field) .≈ 0)
    end

    # flat interface: every control volume gets the fraction of its own height below `h`
    h = FT(0.42)
    copyto!(chain.h_vertices, TA(backend)(fill(h, n)))
    ratios = make_ratios()
    compute_rock_fraction!(ratios, chain, xvi, dxi)
    rock_fraction(y_bottom, height) = clamp((h - y_bottom) / height, zero(FT), one(FT))

    center = Array(ratios.center)
    for j in axes(center, 2)
        @test all(center[:, j] .≈ rock_fraction(yv[j], dy[j]))
    end

    vx = Array(ratios.Vx)
    for j in axes(vx, 2)
        @test all(vx[:, j] .≈ rock_fraction(yv[j], dy[j]))
    end

    vertex = Array(ratios.vertex)
    vy = Array(ratios.Vy)
    for j in axes(vertex, 2)
        # the two halves straddling vertex `j` are cut from rows of different height, so
        # the control-volume fraction weights them by area, not by count
        expected = if j == firstindex(yv)
            rock_fraction(yv[j], dy[j] / 2)
        elseif j == lastindex(yv)
            rock_fraction(yv[j] - dy[j - 1] / 2, dy[j - 1] / 2)
        else
            (
                dy[j - 1] * rock_fraction(yv[j] - dy[j - 1] / 2, dy[j - 1] / 2) +
                    dy[j] * rock_fraction(yv[j], dy[j] / 2)
            ) / (dy[j - 1] + dy[j])
        end
        @test all(vertex[:, j] .≈ expected)
        @test all(vy[:, j] .≈ expected)
    end

    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        data = Array(field)
        @test all(isfinite, data)
        @test all(0 .≤ data .≤ 1)
    end
end

# Exact fraction of the rectangle [x0, x1] x [y0, y1] lying below the line y = h0 + s * x.
# The integrand is piecewise linear with breakpoints where the line crosses the floor and
# the ceiling, so the trapezoid rule over those nodes is exact.
function markerchain_rock_fraction(h0, s, x0, x1, y0, y1)
    g(x) = clamp((h0 + s * x - y0) / (y1 - y0), zero(FT), one(FT))
    breaks = iszero(s) ? FT[] : FT[(y0 - h0) / s, (y1 - h0) / s]
    nodes = sort(vcat(FT[x0, x1], filter(x -> x0 < x < x1, breaks)))
    total = zero(FT)
    for k in 1:(length(nodes) - 1)
        total += (g(nodes[k]) + g(nodes[k + 1])) * (nodes[k + 1] - nodes[k]) / 2
    end
    return total / (x1 - x0)
end

@testset "MarkerChain sloping interface rock fraction 2D" begin
    n = 9
    xv = markerchain_graded_grid(n)
    yv = reverse(one(FT) .- markerchain_graded_grid(n))
    dx = diff(xv)
    dy = diff(yv)
    xvi = TA(backend)(xv), TA(backend)(yv)
    dxi = TA(backend)(dx), TA(backend)(dy)

    h0, slope = FT(0.25), FT(0.5)
    chain = init_markerchain(backend, 3, 2, 6, TA(backend)(xv), FT(0.5))
    copyto!(chain.h_vertices, TA(backend)(h0 .+ slope .* xv))

    ratios = (
        center = TA(backend)(zeros(FT, n - 1, n - 1)),
        vertex = TA(backend)(zeros(FT, n, n)),
        Vx = TA(backend)(zeros(FT, n, n - 1)),
        Vy = TA(backend)(zeros(FT, n - 1, n)),
    )
    compute_rock_fraction!(ratios, chain, xvi, dxi)

    # extents of the staggered control volumes, truncated at the domain boundary; the
    # sub-cells tiling them have unequal areas on a graded grid, so a count-weighted average
    # of their fractions is wrong
    left(i) = i == 1 ? xv[1] : xv[i] - dx[i - 1] / 2
    right(i) = i == n ? xv[n] : xv[i] + dx[i] / 2
    bottom(j) = j == 1 ? yv[1] : yv[j] - dy[j - 1] / 2
    top(j) = j == n ? yv[n] : yv[j] + dy[j] / 2

    atol = FT === Float32 ? 1.0f-4 : 1.0e-9

    center = Array(ratios.center)
    for j in axes(center, 2), i in axes(center, 1)
        expected = markerchain_rock_fraction(h0, slope, xv[i], xv[i + 1], yv[j], yv[j + 1])
        @test isapprox(center[i, j], expected; atol, rtol = atol)
    end

    vertex = Array(ratios.vertex)
    for j in axes(vertex, 2), i in axes(vertex, 1)
        expected = markerchain_rock_fraction(h0, slope, left(i), right(i), bottom(j), top(j))
        @test isapprox(vertex[i, j], expected; atol, rtol = atol)
    end

    vx = Array(ratios.Vx)
    for j in axes(vx, 2), i in axes(vx, 1)
        expected = markerchain_rock_fraction(h0, slope, left(i), right(i), yv[j], yv[j + 1])
        @test isapprox(vx[i, j], expected; atol, rtol = atol)
    end

    vy = Array(ratios.Vy)
    for j in axes(vy, 2), i in axes(vy, 1)
        expected = markerchain_rock_fraction(h0, slope, xv[i], xv[i + 1], bottom(j), top(j))
        @test isapprox(vy[i, j], expected; atol, rtol = atol)
    end
end
