using JustPIC, Test, StaticArrays

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
    xvi = xv, yv = range(0, Lx, length = n), range(0, Ly, length = n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0 + dx / 2, Lx - dx / 2, length = n - 1), range(0 + dy / 2, Ly - dy / 2, length = n - 1)
    # staggered grid velocity nodal locations

    particles = JustPIC._2D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    nphases = 5
    phases, = JustPIC._2D.init_cell_arrays(particles, Val(1))
    T = typeof(phases.data)
    phases.data .= T(rand(1:nphases, size(phases.data)))

    phase_ratios = JustPIC._2D.PhaseRatios(backend, nphases, ni)

    JustPIC._2D.update_phase_ratios!(phase_ratios, particles, xci, xvi, phases)

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
    xvi = xv, yv, zv = ntuple(i -> range(0, Li[i], length = n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> range(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, length = ni[i]), Val(3))

    nxcell, max_xcell, min_xcell = 125, 125, 125
    particles = JustPIC._3D.init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    nphases = 5
    phases, = JustPIC._3D.init_cell_arrays(particles, Val(1))
    T = typeof(phases.data)
    phases.data .= T(rand(1:nphases, size(phases.data)))

    phase_ratios = JustPIC._3D.PhaseRatios(backend, nphases, ni);|

    JustPIC._3D.update_phase_ratios!(phase_ratios, particles, xci, xvi, phases)

    @test all(extrema([sum(p) for p in phase_ratios.vertex]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.center]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vx]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vy]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.Vz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.xz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.yz]) .≈ 1)
    @test all(extrema([sum(p) for p in phase_ratios.xy]) .≈ 1)
end
