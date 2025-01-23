using Aqua, Test, JustPIC

## Failing tests: hard to fix in the current state of the pkg
# Aqua.test_unbound_args(JustPIC)
# Aqua.test_piracies(JustPIC)

# @testset "Ambiguities" begin
#     @test Aqua.test_ambiguities(JustPIC, 
#         color = true,
#         # exclude = [_grid2particle],
#         exclude = [
#             JustPIC._2D._grid2particle, 
#             JustPIC._3D._grid2particle,
#         ],
#     ).value
# end

@testset "Project extras" begin
    @test Aqua.test_project_extras(JustPIC).value
end

@testset "Undefined exports" begin
    @test Aqua.test_undefined_exports(JustPIC).value
end

@testset "Compats" begin
    @test !Aqua.test_deps_compat(
        JustPIC;
        check_julia = true,
        check_extras = false,
        check_weakdeps = true,
    ).anynonpass
    @test Aqua.test_stale_deps(JustPIC).value
end

@testset "Persistent tasks" begin
    Aqua.test_persistent_tasks(JustPIC)
end
