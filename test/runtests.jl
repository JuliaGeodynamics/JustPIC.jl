using JustPIC

using Pkg

push!(LOAD_PATH, "..")

istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")

function parse_flags!(args, flag; default=nothing, typ=typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing || typ <: AbstractString)
                @show typ val
                val = parse(typ, val)
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
    testdir = pwd()
    testfiles = sort(
        filter(
            istest,
            vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...),
        ),
    )
    nfail = 0
    printstyled("Testing package JustPIC.jl\n"; bold=true, color=:white)

    # 2D tests --------------------------------------------------
    printstyled("Running 2D tests\n"; bold=true, color=:white)

    # if get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "AMDGPU"
    #     # set_backend("AMDGPU_Float64_2D")
    #     using AMDGPU
    # elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CUDA"
    #     # set_backend("CUDA_Float64_2D")
    #     using CUDA
    # # elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CPU"
    # #     set_backend("Threads_Float64_2D")
    # end

    for f in testfiles
        if occursin("2D", f)
            println("\n Running tests from $f")
            try
                run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                nfail += 1
            end
        end
    end

    # 3D tests --------------------------------------------------
    printstyled("Running 3D tests\n"; bold=true, color=:white)

    # if get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "AMDGPU"
    #     # set_backend("AMDGPU_Float64_3D")
    #     using AMDGPU

    # elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CUDA"
    #     # set_backend("CUDA_Float64_3D")
    #     using CUDA

    # # elseif get(ENV, "JULIA_JUSTPIC_BACKEND", "") == "CPU"
    # #     set_backend("Threads_Float64_3D")
    # end

    for f in testfiles
        if occursin("3D", f)
            println("\n Running tests from $f")
            try
                run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                nfail += 1
            end
        end
    end

    return nfail
end

_, backend_name = parse_flags!(ARGS, "--backend"; default="CPU", typ=String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    ENV["JULIA_JUSTPIC_BACKEND"] = "AMDGPU"
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    ENV["JULIA_JUSTPIC_BACKEND"] = "CUDA"
elseif backend_name == "CPU"
    ENV["JULIA_JUSTPIC_BACKEND"] = "CPU"
end

exit(runtests())
