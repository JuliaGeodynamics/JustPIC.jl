using JustPIC, Test, StaticArrays

@testset "CellArrays - 2D" begin
    x = 1e0
    ncells = (2,)
    ni = (2, 2)
    CA =  JustPIC._2D.cell_array(x, ncells, ni)
    @test all(isone, CA.data)

    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2)

    JustPIC._2D.@cell CA[2, 1, 1] = 2.0
    @test JustPIC._2D.@cell(CA[2, 1, 1]) == 2.0

    ncells = (2,2)
    CA =  JustPIC._2D.cell_array(x, ncells, ni)
    @test all(isone, CA.data)

    @test JustPIC._2D.new_empty_cell(CA) == @SArray zeros(2,2)

    JustPIC._2D.@cell CA[2, 2, 1, 1] = 2.0
    @test JustPIC._2D.@cell(CA[2, 2, 1, 1]) == 2.0
end

@testset "CellArrays - 3D" begin
    x = 1e0
    ncells = (2,)
    ni = (2, 2, 2)
    CA =  JustPIC._3D.cell_array(x, ncells, ni)
    @test all(isone, CA.data)

    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2)

    JustPIC._3D.@cell CA[2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@cell(CA[2, 1, 1, 1]) == 2.0

    ncells = (2,2)
    CA =  JustPIC._3D.cell_array(x, ncells, ni)
    @test all(isone, CA.data)

    @test JustPIC._3D.new_empty_cell(CA) == @SArray zeros(2,2)

    JustPIC._3D.@cell CA[2, 2, 1, 1, 1] = 2.0
    @test JustPIC._3D.@cell(CA[2, 2, 1, 1, 1]) == 2.0
end