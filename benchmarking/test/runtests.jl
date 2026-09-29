using JSON: JSON
using JustPIC
using JustPICBenchmarks
using Test

@testset "benchmark harness" begin
    cases = benchmark_cases(; particle_size = 8, particle_size_3d = 4, surface_size = 8)
    results = run_benchmarks(; samples = 1, cases)

    @test length(results) == 21
    @test count(result -> result["parameters"]["dimension"] == 3, results) == 11
    directional = filter(result -> haskey(result["parameters"], "direction"), results)
    @test length(directional) == 16
    @test sort(unique(result["parameters"]["direction"] for result in directional)) == [
        "centroid_to_particle", "grid_to_particle", "particle_to_centroid", "particle_to_grid",
    ]
    @test all(result -> result["value"] > 0, results)
    @test all(result -> result["sanity_check"] == "passed", results)
    @test all(result -> result["samples"] == 1, results)
    @test all(result -> result["modeled_flops"] > 0, results)
    @test all(result -> result["modeled_memory_bytes"] > 0, results)
    @test all(result -> result["arithmetic_intensity_flops_per_byte"] > 0, results)
    @test all(result -> result["effective_gflops_per_second"] > 0, results)
    @test all(result -> result["performance_metric_source"] == "algorithmic_model", results)
    @test all(
        result -> isapprox(
            result["effective_flops_per_second"],
            result["modeled_flops"] / result["time_median_seconds"],
        ),
        results,
    )
    @test all(
        result -> result["metadata"]["justpic_source"] == ".",
        results,
    )
    @test all(result -> !occursin(".local", result["metadata"]["hardware_fingerprint"]), results)
    @test all(result -> haskey(result["metadata"], "commit_subject"), results)
    @test all(result -> result["metadata"]["backend"] == "CPU", results)
    @test all(result -> result["metadata"]["float_type"] == "Float64", results)
    @test all(result -> result["parameters"]["float_type"] == "Float64", results)
    @test all(result -> endswith(result["name"], "Float64)"), results)
    @test results[1]["metadata"]["peak_memory_bandwidth_gb_per_second"] > 0
    @test results[1]["metadata"]["peak_compute_gflops"] > 0
    @test_throws "device description is required" run_benchmarks(;
        backend_name = "CUDA", samples = 1, cases,
    )

    cases32 = benchmark_cases(; particle_size = 8, particle_size_3d = 4, surface_size = 8, precision = Float32)
    results32 = run_benchmarks(; samples = 1, precision = Float32, cases = cases32)
    @test all(result -> endswith(result["name"], "Float32)"), results32)
    @test all(result -> result["parameters"]["float_type"] == "Float32", results32)
    @test all(
        i -> results32[i]["modeled_memory_bytes"] < results[i]["modeled_memory_bytes"],
        eachindex(results32, results),
    )
    @test_throws "unknown precision" JustPICBenchmarks.parse_commandline(["--precision=Float16"])
    sized = JustPICBenchmarks.parse_commandline(["--size=64", "--size-3d=16"])
    @test (sized.particle_size, sized.particle_size_3d) == (64, 16)
    @test_throws "must be positive" JustPICBenchmarks.parse_commandline(["--size=0"])

    comparison = sprint(io -> @test all(==(1), print_comparison(io, results, results)))
    @test occursin(results[1]["name"], comparison)

    # a baseline case that failed is reported without a ratio; a failed candidate is fatal
    failing = JustPICBenchmarks.BenchmarkCase(
        "failing case", "Interpolation", () -> error("baseline lacks this API"), identity, identity,
        1, "unit", Dict{String, Any}(), cases[1].performance_model,
    )
    @test_throws "baseline lacks this API" run_benchmarks(; samples = 1, cases = (failing,))
    failed = only(run_benchmarks(; samples = 1, cases = (failing,), allow_case_errors = true))
    @test failed["error"] == "baseline lacks this API"
    baseline = [failed; collect(results[2:end])]
    candidate = [merge(results[1], Dict("name" => "failing case")); collect(results[2:end])]
    partial = sprint() do io
        @test length(print_comparison(io, baseline, candidate)) == length(results) - 1
    end
    @test occursin("Baseline failed failing case: `baseline lacks this API`", partial)
    @test_throws "candidate benchmarks failed: failing case" print_comparison(stdout, candidate, baseline)
    @test JustPICBenchmarks.parse_commandline(["--allow-case-errors"]).allow_case_errors
    @test !occursin("🔴", comparison)
    slower = deepcopy(results)
    slower[1]["time_median_seconds"] *= 2
    @test occursin("🔴", sprint(print_comparison, results, slower))
    elsewhere = deepcopy(results)
    elsewhere[1]["metadata"]["hardware_fingerprint"] = "another machine"
    @test_throws "different hardware_fingerprint" print_comparison(devnull, results, elsewhere)
    @test_throws "different float_type" print_comparison(devnull, results, results32)

    mktempdir() do dir
        path = write_results(joinpath(dir, "results.json"), results)
        decoded = JSON.parsefile(path)
        @test getindex.(decoded, "name") == collect(getindex.(results, "name"))

        history = write_dashboard_data(joinpath(dir, "history.json"), [path])
        payload = JSON.parsefile(history)
        @test payload["schema_version"] == 1
        @test payload["repository_url"] == "https://github.com/JuliaGeodynamics/JustPIC.jl"
        @test length(payload["runs"]) == 1
        @test length(only(payload["runs"])["benchmarks"]) == 21
        @test only(payload["runs"])["source"] == "results.json"
        @test_throws "more than one run for the same commit" write_dashboard_data(
            joinpath(dir, "duplicate-history.json"),
            [path, path],
        )
    end
end
