@testset "Runge-Kutta 2" begin
    rk2 = JustPIC._2D.RungeKutta2()
    @test rk2.α == 0.5
    rk2 = JustPIC._2D.RungeKutta2(2 / 3)
    @test rk2.α == 2 / 3
    @test_throws ArgumentError RungeKutta2(1.1)
    @test_throws ArgumentError RungeKutta2(-0.1)

    rk2 = JustPIC._2D.RungeKutta2()
    dt  = 0.1
    p   = 1.0, 2.0
    v0  = 3.0, 4.0
    v1  = 3.1, 4.1

    p1 = JustPIC._2D.first_stage(rk2, dt, v0, p)
    @test p1 == (1.15, 2.2)
    @test all(JustPIC._2D.second_stage(rk2, dt, v0, v1, p1) .≈ (1.46, 2.61))

    rk2 = JustPIC._2D.RungeKutta2(2 / 3)
    p1  = JustPIC._2D.first_stage(rk2, dt, v0, p)
    @test all(p1 .≈ (1.2, 2.266666666))
    @test all(JustPIC._2D.second_stage(rk2, dt, v0, v1, p1) .≈ (1.5075, 2.67416666))
end

@testset "Euler" begin
    @test Euler()         isa Euler
    @test Euler(1)        isa Euler
    @test Euler("potato") isa Euler
end