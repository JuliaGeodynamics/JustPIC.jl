using JustPIC, Test, StaticArrays

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function expand_range(x::AbstractVector)
    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    x1, x2 = extrema(x)
    xI = x1 - dx_left
    xF = x2 + dx_right
    return vcat(xI, x, xF)
end
@testset "CellArrays - 2D" begin
    x = 1.0e0
    ni = (2, 2)

    ## Test a 2x2 grid with 2x1 CellArrays per grid cell
    ncells = (2,)
    # instantiate CellArray object
    CA = JustPIC._2D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2)
    # mutate and read 2nd element in grid cell [1, 1]
    JustPIC._2D.@index CA[2, 1, 1] = 2.0
    @test JustPIC._2D.@index(CA[2, 1, 1]) == 2.0

    ## Test a 2x2 grid with 2x2 CellArrays per grid cell
    ncells = (2, 2)
    # instantiate CellArray object
    CA = JustPIC._2D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2, 2)
    # mutate and read [2,2] element in grid cell [1, 1]
    JustPIC._2D.@index CA[2, 2, 1, 1] = 2.0
    @test JustPIC._2D.@index(CA[2, 2, 1, 1]) == 2.0
end

@testset "Phase ratios - 2D" begin
    nxcell, max_xcell, min_xcell = 50, 50, 50
    n = 256
    nx = ny = n - 1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = JustPIC._2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, (grid_vx, grid_vy)...,
    )

    nphases = 5
    phases, = JustPIC._2D.init_cell_arrays(particles, Val(1))
    T = typeof(phases.data)
    phases.data .= T(rand(1:nphases, size(phases.data)))

    phase_ratios = JustPIC._2D.PhaseRatios(backend, nphases, ni)

    JustPIC._2D.update_phase_ratios!(phase_ratios, particles, phases)

    @test all(extrema([sum(p) for p in phase_ratios.vertex]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.center]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vx]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vy]) .≈ 1)
end

@testset "CellArrays - 3D" begin
    x = 1.0e0
    ni = (2, 2, 2)

    ## Test a 2x2x2 grid with 2x1 CellArrays per grid cell
    ncells = (2,)
    # instantiate CellArray object
    CA = JustPIC._3D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2)
    # mutate and read 2nd element in grid cell [1, 1, 1]
    JustPIC._3D.@index CA[2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@index(CA[2, 1, 1, 1]) == 2.0

    ## Test a 2x2x2 grid with 2x2 CellArrays per grid cell
    ncells = (2, 2)
    # instantiate CellArray object
    CA = JustPIC._3D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2, 2)
    # mutate and read [2,2] element in grid cell [1, 1, 1]
    JustPIC._3D.@index CA[2, 2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@index(CA[2, 2, 1, 1, 1]) == 2.0
end

@testset "Phase ratios - 3D" begin
    n = 32
    nx = ny = nz = n - 1
    Lx = Ly = Lz = 1.0
    ni = nx, ny, nz
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz

    nxcell, max_xcell, min_xcell = 125, 125, 125
    particles = JustPIC._3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...,
    )

    nphases = 5
    phases, = JustPIC._3D.init_cell_arrays(particles, Val(1))
    T = typeof(phases.data)
    phases.data .= T(rand(1:nphases, size(phases.data)))

    phase_ratios = JustPIC._3D.PhaseRatios(backend, nphases, ni);|

    JustPIC._3D.update_phase_ratios!(phase_ratios, particles, phases)

    @test all(extrema([sum(p) for p in phase_ratios.vertex]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.center]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vx]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vy]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.xz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.yz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.xy]) .≈ 1)
end
