using JustPIC, Test, StaticArrays

@testset "CellArrays - 2D" begin
    x = 1e0
    ni = (2, 2)

    ## Test a 2x2 grid with 2x1 CellArrays per grid cell
    ncells = (2,)
    # instantiate CellArray object
    CA =  JustPIC._2D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2)
    # mutate and read 2nd element in grid cell [1, 1]
    JustPIC._2D.@index CA[2, 1, 1] = 2.0
    @test JustPIC._2D.@index(CA[2, 1, 1]) == 2.0

    ## Test a 2x2 grid with 2x2 CellArrays per grid cell
    ncells = (2,2)
    # instantiate CellArray object
    CA =  JustPIC._2D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2,2)
    # mutate and read [2,2] element in grid cell [1, 1]
    JustPIC._2D.@index CA[2, 2, 1, 1] = 2.0
    @test JustPIC._2D.@index(CA[2, 2, 1, 1]) == 2.0
end

@testset "CellArrays - 3D" begin
    x = 1e0
    ni = (2, 2, 2)
    
    ## Test a 2x2x2 grid with 2x1 CellArrays per grid cell
    ncells = (2,)
    # instantiate CellArray object
    CA =  JustPIC._3D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2)
    # mutate and read 2nd element in grid cell [1, 1, 1]
    JustPIC._3D.@index CA[2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@index(CA[2, 1, 1, 1]) == 2.0

    ## Test a 2x2x2 grid with 2x2 CellArrays per grid cell
    ncells = (2,2)
    # instantiate CellArray object
    CA =  JustPIC._3D.cell_array(x, ncells, ni)
    # test all the data is one
    @test all(isone, CA.data)
    # create empty cell
    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2,2)
    # mutate and read [2,2] element in grid cell [1, 1, 1]
    JustPIC._3D.@index CA[2, 2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@index(CA[2, 2, 1, 1, 1]) == 2.0
end