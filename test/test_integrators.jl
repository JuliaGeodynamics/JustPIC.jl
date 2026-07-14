using JustPIC, Test

@testset "Runge-Kutta 2 - 2D" begin
    rk2 = JustPIC.RungeKutta2()
    @test rk2.α == 0.5
    rk2 = JustPIC.RungeKutta2(2 / 3)
    @test rk2.α == 2 / 3
    @test_throws ArgumentError RungeKutta2(1.1)
    @test_throws ArgumentError RungeKutta2(-0.1)

    rk2 = JustPIC.RungeKutta2()
    dt = 0.1
    p = 1.0, 2.0
    v0 = 3.0, 4.0
    v1 = 3.1, 4.1

    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test p1 == (1.15, 2.2)
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.46, 2.61))

    rk2 = JustPIC.RungeKutta2(2 / 3)
    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test all(p1 .≈ (1.2, 2.266666666))
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.5075, 2.67416666))
end

@testset "Euler - 2D" begin
    @test JustPIC.Euler() isa Euler
    @test JustPIC.Euler(1) isa Euler
    @test JustPIC.Euler("potato") isa Euler

    euler = JustPIC.Euler()
    dt = 0.1
    p = 1.0, 2.0
    v0 = 3.0, 4.0
    v1 = 3.1, 4.1

    p1 = JustPIC.first_stage(euler, dt, v0, p)
    @test p1 == (1.3, 2.4)
end

@testset "Runge-Kutta 2 - 3D" begin
    rk2 = JustPIC.RungeKutta2()
    @test rk2.α == 0.5
    rk2 = JustPIC.RungeKutta2(2 / 3)
    @test rk2.α == 2 / 3
    @test_throws ArgumentError RungeKutta2(1.1)
    @test_throws ArgumentError RungeKutta2(-0.1)

    rk2 = JustPIC.RungeKutta2()
    dt = 0.1
    p = 1.0, 2.0, 3.0
    v0 = 3.0, 4.0, 5.0
    v1 = 3.1, 4.1, 5.1

    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test p1 == (1.15, 2.2, 3.25)
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.46, 2.61, 3.76))

    rk2 = JustPIC.RungeKutta2(2 / 3)
    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test all(p1 .≈ (1.2, 2.266666666, 3.3333333333333335))
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.5075, 2.67416666, 3.8408333333333333))
end

@testset "Euler - 3D" begin
    @test JustPIC.Euler() isa Euler
    @test JustPIC.Euler(1) isa Euler
    @test JustPIC.Euler("potato") isa Euler

    euler = JustPIC.Euler()
    dt = 0.1
    p = 1.0, 2.0, 3.0
    v0 = 3.0, 4.0, 5.0
    v1 = 3.1, 4.1, 5.1

    p1 = JustPIC.first_stage(euler, dt, v0, p)
    @test p1 == (1.3, 2.4, 3.5)

end
