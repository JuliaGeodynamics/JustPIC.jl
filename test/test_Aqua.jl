using Aqua, Test, JustPIC

## Failing tests: hard to fix in the current state of the pkg
# Aqua.test_unbound_args(JustPIC)
# Aqua.test_piracies(JustPIC)

@testset "Ambiguities" begin
    @test Aqua.test_ambiguities(
        JustPIC,
        color = true,
        # exclude = [_grid2particle],
        exclude = [
            JustPIC._grid2particle,
            JustPIC._grid2particle,
        ],
    ).value
end

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
    # GPU stacks are weakdeps that cluster CI may promote to hard deps (see runtests.jl)
    @test Aqua.test_stale_deps(JustPIC; ignore = [:CUDA, :AMDGPU, :Metal]).value
end

@testset "Persistent tasks" begin
    Aqua.test_persistent_tasks(JustPIC)
end

@testset "Migration lint" begin
    root = normpath(joinpath(@__DIR__, ".."))
    files = String[]
    for dir in ("src", "ext", "scripts", "test")
        for (subdir, _, names) in walkdir(joinpath(root, dir))
            append!(files, joinpath.(Ref(subdir), filter(endswith(".jl"), names)))
        end
    end

    stale_ps = Tuple{String, Int, String}[]
    bare_index = Tuple{String, Int, String}[]
    stale_tokens = (
        "using " * "ParallelStencil",
        "using " * "Atomix",
        "@" * "init_parallel_stencil",
        "@" * "parallel_indices",
        "@" * "parallel",
        "@" * "fill",
        "@" * "myatomic",
    )
    index_token = "@" * "index"

    for file in files
        for (lineno, line) in enumerate(eachline(file))
            stripped = strip(line)
            isempty(stripped) && continue
            startswith(stripped, "#") && continue

            if any(token -> occursin(token, stripped), stale_tokens)
                push!(stale_ps, (file, lineno, stripped))
            end

            if occursin(index_token, stripped)
                allowed =
                    occursin(index_token * "(Global", stripped) ||
                    occursin(index_token * "(Local", stripped) ||
                    occursin(index_token * "(Group", stripped) ||
                    occursin("CAI." * index_token, stripped) ||
                    occursin("JustPIC." * index_token, stripped) ||
                    startswith(stripped, "import KernelAbstractions:") ||
                    startswith(stripped, "export " * "@" * "cell, " * index_token) ||
                    startswith(stripped, "export " * index_token)
                allowed || push!(bare_index, (file, lineno, stripped))
            end
        end
    end

    @test stale_ps == []
    @test bare_index == []
end
