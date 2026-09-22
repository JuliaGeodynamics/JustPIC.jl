pushfirst!(LOAD_PATH, dirname(@__DIR__))

using JustPIC
using Pkg

function parse_flags!(args, flag; default = nothing)
    for (i, value) in pairs(args)
        startswith(value, flag) || continue
        parsed = value == flag ? default : split(value, "="; limit = 2)[2]
        deleteat!(args, i)
        return true, parsed
    end
    return false, default
end

const FAST_SUITES = ("test_fast.jl",)
const FULL_SUITES = (
    "test_Aqua.jl", "test_2D.jl", "test_3D.jl", "test_integrators.jl",
    "test_CellArrays.jl", "test_markerchain_2D.jl", "test_refined_grid.jl",
    "test_save_load.jl", "test_interpolation_kernels.jl", "test_semilagrangian.jl",
    "test_marker_surface.jl",
)

function run_suite(testdir, load_path, filename, tier)
    path = joinpath(testdir, filename)
    printstyled("\nRunning $filename\n"; bold = true, color = :white)
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no $path`,
        "JULIA_LOAD_PATH" => load_path,
        "JULIA_JUSTPIC_ALLOW_SCALAR" => tier == "full" ? "true" : "false",
    )
    try
        run(cmd)
        printstyled("PASS $filename\n"; color = :green)
        return true
    catch ex
        printstyled("FAIL $filename\n"; color = :red)
        showerror(stderr, ex, catch_backtrace())
        println(stderr)
        return false
    end
end

function runtests(tier)
    testdir = @__DIR__
    load_path = join(LOAD_PATH, Sys.iswindows() ? ';' : ':')
    suites = tier == "fast" ? FAST_SUITES : tier == "full" ? FULL_SUITES : (FAST_SUITES..., FULL_SUITES...)
    failures = count(!run_suite(testdir, load_path, suite, tier) for suite in suites)
    println("\n$(length(suites) - failures)/$(length(suites)) test suites passed")
    return failures
end

_, backend_name = parse_flags!(ARGS, "--backend"; default = "CPU")
_, tier = parse_flags!(ARGS, "--tier"; default = "full")
backend_name in ("CPU", "CUDA", "AMDGPU", "Metal") ||
    error("Unknown backend $(repr(backend_name)); use --backend=CPU|CUDA|AMDGPU|Metal")
tier in ("fast", "full", "all") || error("Unknown tier $(repr(tier)); use --tier=fast|full|all")
isempty(ARGS) || error("Unrecognised test arguments $(ARGS)")

backend_name == "CPU" || Pkg.add(backend_name)
ENV["JULIA_JUSTPIC_BACKEND"] = backend_name

exit(runtests(tier))
