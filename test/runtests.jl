pushfirst!(LOAD_PATH, dirname(@__DIR__))

using JustPIC

using Pkg

istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")

function parse_flags!(args, flag; default = nothing, type = typeeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(type ≡ nothing || type <: AbstractString)
                @show type val
                val = parse(type, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

function runtests()
    testdir = @__DIR__
    projectdir = dirname(testdir)
    load_path = string("@:", projectdir, ":@v#.#:@stdlib")
    testfiles = sort(
        filter(
            istest,
            vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...),
        ),
    )
    nfail = 0
    printstyled("Testing package JustPIC.jl\n"; bold = true, color = :white)

    if get(ENV, "JULIA_JUSTPIC_BACKEND", "") === "CPU"

        try
            printstyled("Running 2D tests\n"; bold = true, color = :white)
            include(joinpath(testdir, "test_Aqua.jl"))
            include(joinpath(testdir, "test_2D.jl"))
            include(joinpath(testdir, "test_integrators.jl"))
            include(joinpath(testdir, "test_CellArrays.jl"))
            include(joinpath(testdir, "test_save_load.jl"))
        catch
            nfail += 1
        end
        try
            printstyled("Running 3D tests\n"; bold = true, color = :white)
            include(joinpath(testdir, "test_3D.jl"))
        catch
            nfail += 1
        end
    else
        gpu_testfiles = (
            "test_2D.jl",
            "test_3D.jl",
            "test_CellArrays.jl",
            "test_interpolation_kernels.jl",
            "test_save_load.jl",
        )
        for f in gpu_testfiles
            println("\n Running tests from $f")
            try
                cmd = addenv(
                    `$(Base.julia_cmd()) --project=$(testdir) --startup-file=no $(joinpath(testdir, f))`,
                    "JULIA_LOAD_PATH" => load_path,
                )
                run(cmd)
            catch ex
                nfail += 1
            end
        end
    end

    return nfail
end

_, backend_name = parse_flags!(ARGS, "--backend"; default = "CPU", type = String)

@static if backend_name == "AMDGPU"
    ENV["JULIA_JUSTPIC_BACKEND"] = "AMDGPU"
elseif backend_name == "CUDA"
    ENV["JULIA_JUSTPIC_BACKEND"] = "CUDA"
elseif backend_name == "CPU"
    ENV["JULIA_JUSTPIC_BACKEND"] = "CPU"
end

exit(runtests())
