module JustPICBenchmarks

using Chairmarks: @be
using Dates: UTC, now
using JSON: JSON
using JustPIC
using JustPIC.KernelAbstractions: @Const, @index, @kernel, allocate
using Statistics: mean, median, quantile

export benchmark_cases, dashboard_main, main, print_comparison, run_benchmarks, write_dashboard_data,
    write_results

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PERFORMANCE_MODEL_VERSION = "justpic_cpu_v2"
const REPOSITORY_URL = "https://github.com/JuliaGeodynamics/JustPIC.jl"

struct PerformanceModel
    flops::Int
    memory_bytes::Int
    description::String
end

struct BenchmarkCase{S, R, V}
    name::String
    group::String
    setup::S
    run::R
    validate::V
    work_units::Int
    work_unit::String
    parameters::Dict{String, Any}
    performance_model::PerformanceModel
end

function extended_centers(x)
    dx = step(x)
    centers = range(first(x) + dx / 2, last(x) - dx / 2; length = length(x) - 1)
    return range(first(centers) - dx, last(centers) + dx; length = length(centers) + 2)
end

synchronize(backend) = JustPIC.KernelAbstractions.synchronize(backend())

# Byte counts whose split into field scalars and index/flag bytes is not recorded
# are treated as `Float32` scalars, so they scale with a wider element type.
scale_float32_bytes(bytes, ::Type{FT}) where {FT} = bytes * sizeof(FT) ÷ sizeof(Float32)

function particle_state(backend, n, ::Type{FT}, ::Val{D}) where {FT, D}
    xv = ntuple(_ -> range(FT(0), FT(1); length = n + 1), Val(D))
    grid_vi = ntuple(i -> ntuple(j -> j == i ? xv[j] : extended_centers(xv[j]), Val(D)), Val(D))
    particles = init_particles(backend, ntuple(_ -> 2, Val(D)), 2^(D + 1), 2, grid_vi...)
    particle_field, = init_cell_arrays(particles, Val(1))
    xvi = Array.(particles.xvi)
    nodal_field = TA(backend)(FT[sum(x) for x in Iterators.product(xvi...)])
    grid2particle!(particle_field, nodal_field, particles)

    velocities = FT.((0.1, -0.05, 0.025))
    V = ntuple(i -> TA(backend)(fill(velocities[i], length.(grid_vi[i]))), Val(D))
    dt = FT(step(xv[1]) / FT(0.4))
    initial_count = count(Array(particles.index.data))
    return (; particles, particle_field, nodal_field, V, dt, initial_count)
end

function validate_particle_state(state)
    active = Array(state.particles.index.data)
    count(active) == state.initial_count || error("particle count changed during benchmark")
    for coordinate in state.particles.coords
        values = Array(coordinate.data)[active]
        all(isfinite, values) || error("particle benchmark produced nonfinite coordinates")
        all(x -> 0 <= x <= 1, values) || error("particle benchmark moved a particle outside the periodic domain")
    end
    all(isfinite, Array(state.particle_field.data)[active]) ||
        error("particle benchmark produced a nonfinite particle field")
    return nothing
end

# Performance-model coefficients per particle (`Np`), slot (`Ns`), vertex (`Nv`), and cell
# (`Nc`); benchmarking/README.md lists them.
advection_move_model(::Val{2}, Np, Ns, ::Type{FT}) where {FT} = PerformanceModel(
    80 * Np,
    26 * sizeof(FT) * Np + 2 * Ns,
    "two-stage 2D RK2 velocity interpolation and one logical particle-payload movement pass",
)
advection_move_model(::Val{3}, Np, Ns, ::Type{FT}) where {FT} = PerformanceModel(
    234 * Np,
    62 * sizeof(FT) * Np + 2 * Ns,
    "two-stage 3D RK2 trilinear velocity interpolation and one logical particle-payload movement pass",
)

interpolation_model(::Val{2}, Np, Nv, Nc, ::Type{FT}) where {FT} = PerformanceModel(
    62 * Np + Nv,
    15 * sizeof(FT) * Np + sizeof(FT) * Nv + 8 * sizeof(FT) * Nc,
    "one inverse-distance particle-to-grid pass followed by one bilinear grid-to-particle pass",
)
interpolation_model(::Val{3}, Np, Nv, Nc, ::Type{FT}) where {FT} = PerformanceModel(
    149 * Np + 2 * Nv,
    36 * sizeof(FT) * Np + sizeof(FT) * Nv + 14 * sizeof(FT) * Nc,
    "one inverse-distance particle-to-grid pass followed by one trilinear grid-to-particle pass",
)

grid_label(n, D) = join(fill(n, D), "×")

function advection_move_case(backend, n, ::Type{FT}, dims::Val{D} = Val(2)) where {FT, D}
    setup() = particle_state(backend, n, FT, dims)
    periodic = (; periodic_1 = true, periodic_2 = true, periodic_3 = D == 3)
    function run(state)
        advection!(state.particles, RungeKutta2(), state.V, state.dt; periodic...)
        move_particles!(state.particles, (state.particle_field,); periodic...)
        synchronize(backend)
        return state
    end
    ppc = 2^D
    work_units = ppc * n^D
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(FT),
        "grid_cells" => fill(n, D),
        "particles_per_cell" => ppc,
        "periodic" => fill(true, D),
        "integrator" => "RungeKutta2",
    )
    return BenchmarkCase(
        "Particle advection + move ($(D)D, $(grid_label(n, D)), $ppc ppc, $FT)",
        "Particle workflow",
        setup,
        run,
        validate_particle_state,
        work_units,
        "particles",
        parameters,
        advection_move_model(dims, work_units, 2 * work_units, FT),
    )
end

function interpolation_case(backend, n, ::Type{FT}, dims::Val{D} = Val(2)) where {FT, D}
    setup() = particle_state(backend, n, FT, dims)
    function run(state)
        particle2grid!(state.nodal_field, state.particle_field, state.particles)
        grid2particle!(state.particle_field, state.nodal_field, state.particles)
        synchronize(backend)
        return state
    end
    ppc = 2^D
    work_units = ppc * n^D
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(FT),
        "grid_cells" => fill(n, D),
        "particles_per_cell" => ppc,
        "directions" => ["particle_to_grid", "grid_to_particle"],
    )
    return BenchmarkCase(
        "Particle ↔ grid interpolation ($(D)D, $(grid_label(n, D)), $ppc ppc, $FT)",
        "Interpolation",
        setup,
        run,
        validate_particle_state,
        work_units,
        "particle values",
        parameters,
        interpolation_model(dims, work_units, (n + 1)^D, n^D, FT),
    )
end

function marker_surface_state(backend, n, ::Type{FT}) where {FT}
    xv = range(FT(0), FT(1); length = n + 1)
    yv = range(FT(0), FT(1); length = n + 1)
    zv = range(FT(0), FT(1); length = 3)
    grid_vxi = map(
        grid -> TA(backend).(collect.(grid)),
        (
            (xv, extended_centers(yv), extended_centers(zv)),
            (extended_centers(xv), yv, extended_centers(zv)),
            (extended_centers(xv), extended_centers(yv), zv),
        ),
    )
    V = (
        TA(backend)(fill(FT(0.01), length.(grid_vxi[1]))),
        TA(backend)(fill(FT(-0.005), length.(grid_vxi[2]))),
        TA(backend)(fill(FT(0), length.(grid_vxi[3]))),
    )
    surface = init_marker_surface(backend, xv, yv, FT(0.5))
    return (; surface, V, grid_vxi, dt = FT(0.01))
end

function validate_marker_surface(state)
    topo = Array(state.surface.topo)
    all(isfinite, topo) || error("MarkerSurface benchmark produced nonfinite topography")
    maximum(abs, topo .- eltype(topo)(0.5)) <= 100 * eps(eltype(topo)) ||
        error("a flat MarkerSurface changed under horizontal translation")
    return nothing
end

function marker_surface_case(backend, n, ::Type{FT}) where {FT}
    setup() = marker_surface_state(backend, n, FT)
    function run(state)
        advect_marker_surface!(state.surface, state.V, state.grid_vxi, state.dt)
        synchronize(backend)
        return state
    end
    work_units = (n + 1)^2
    cells = n^2
    performance_model = PerformanceModel(
        317 * work_units + 14 * cells,
        scale_float32_bytes(294 * work_units + 37 * cells, FT),
        "trilinear velocity interpolation, fixed-case triangle advection, and flat-surface smoothing scan",
    )
    parameters = Dict{String, Any}(
        "dimension" => 3,
        "float_type" => string(FT),
        "surface_cells" => [n, n],
        "velocity_z_cells" => 2,
        "max_slope_angle_degrees" => 45,
    )
    return BenchmarkCase(
        "MarkerSurface update ($(n)×$(n), $FT)",
        "MarkerSurface",
        setup,
        run,
        validate_marker_surface,
        work_units,
        "surface nodes",
        parameters,
        performance_model,
    )
end

function benchmark_cases(
        backend = JustPIC.CPU; particle_size = 128, particle_size_3d = 32, surface_size = 256,
        precision::Type = Float64,
    )
    return (
        advection_move_case(backend, particle_size, precision),
        interpolation_case(backend, particle_size, precision),
        advection_move_case(backend, particle_size_3d, precision, Val(3)),
        interpolation_case(backend, particle_size_3d, precision, Val(3)),
        marker_surface_case(backend, surface_size, precision),
    )
end

function git_metadata()
    commit = readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`)
    dirty = !isempty(read(`git -C $REPOSITORY_ROOT status --porcelain`, String))
    subject = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%s`)
    author = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%an`)
    return (; commit, dirty, subject, author)
end

function justpic_source()
    source = realpath(dirname(dirname(pathof(JustPIC))))
    source == realpath(REPOSITORY_ROOT) ||
        error("benchmark loaded JustPIC from $source instead of $REPOSITORY_ROOT")
    return source
end

function loaded_package_versions()
    tracked = Set(
        (
            "AMDGPU", "CUDA", "CellArrays", "CellArraysIndexing", "ImplicitGlobalGrid",
            "KernelAbstractions", "Metal",
        ),
    )
    versions = Dict{String, String}()
    for (id, package) in Base.loaded_modules
        id.name in tracked || continue
        version = pkgversion(package)
        isnothing(version) || (versions[id.name] = string(version))
    end
    return versions
end

@kernel function triad!(a, @Const(b), @Const(c), s)
    i = @index(Global)
    a[i] = b[i] + s * c[i]
end

# Eight independent chains hide FMA latency; summing them into `out` keeps them live.
@kernel function fma_chains!(out, s, c, iterations)
    i = @index(Global)
    T = eltype(out)
    x1, x2, x3, x4 = T(i), T(i + 1), T(i + 2), T(i + 3)
    x5, x6, x7, x8 = T(i + 4), T(i + 5), T(i + 6), T(i + 7)
    for _ in 1:iterations
        x1 = muladd(x1, s, c); x2 = muladd(x2, s, c)
        x3 = muladd(x3, s, c); x4 = muladd(x4, s, c)
        x5 = muladd(x5, s, c); x6 = muladd(x6, s, c)
        x7 = muladd(x7, s, c); x8 = muladd(x8, s, c)
    end
    out[i] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
end

# Fastest of `repeats` launches, after `warmup` seconds of launches that let the
# device reach its sustained clock.
function best_time(backend, kernel!, ndrange, args...; repeats = 5, warmup = 1.0)
    start = time()
    while time() - start < warmup
        JustPIC.launch!(backend, kernel!, ndrange, args...)
    end
    return minimum(1:repeats) do _
        t = time_ns()
        JustPIC.launch!(backend, kernel!, ndrange, args...)
        (time_ns() - t) / 1.0e9
    end
end

"""
    measure_peaks(backend, T; n = 2^26, fma_items = 2^22, fma_iterations = 256)

Attainable `T` memory bandwidth (STREAM triad over `n` elements, counting
3 transfers per element) and compute rate (8 FMA chains of `fma_iterations`
per work item), in GB/s and GFLOP/s. Kernels on the CPU backend do not
vectorize across work items, so the CPU compute rate is a lower bound.
"""
function measure_peaks(
        backend, ::Type{T} = Float64; n = 2^26, fma_items = 2^22, fma_iterations = 256
    ) where {T}
    be = backend()
    a, b, c = (fill!(allocate(be, T, n), one(T)) for _ in 1:3)
    bandwidth = 3 * n * sizeof(T) / best_time(be, triad!, n, a, b, c, T(3)) / 1.0e9

    out = allocate(be, T, fma_items)
    flops = 2 * 8 * fma_iterations * fma_items
    compute = flops / best_time(be, fma_chains!, fma_items, out, T(0.999), T(0.001), fma_iterations) / 1.0e9
    all(isfinite, Array(out)) || error("FMA probe produced non-finite values")
    return (; bandwidth, compute)
end

function benchmark_metadata(backend, backend_name, device, ::Type{FT}) where {FT}
    backend_name == "CPU" || !isnothing(device) ||
        throw(ArgumentError("a device description is required for the $backend_name backend"))
    git = git_metadata()
    source = justpic_source()
    cpu = first(Sys.cpu_info())
    cpu_model = "$(cpu.model) ($(Sys.CPU_NAME))"
    device = something(device, cpu_model)
    peaks = measure_peaks(backend, FT)
    return Dict{String, Any}(
        "timestamp_utc" => string(now(UTC)),
        "commit" => git.commit,
        "commit_subject" => git.subject,
        "commit_author" => git.author,
        "dirty" => git.dirty,
        "julia_version" => string(VERSION),
        "justpic_version" => string(pkgversion(JustPIC)),
        "justpic_source" => relpath(source, REPOSITORY_ROOT),
        "backend" => backend_name,
        "float_type" => string(FT),
        "device" => device,
        "cpu_model" => cpu_model,
        "hardware_fingerprint" =>
            "$(Sys.KERNEL) | $(Sys.ARCH) | $device | $cpu_model | $(Threads.nthreads()) threads",
        "threads" => Threads.nthreads(),
        "package_versions" => loaded_package_versions(),
        "peak_memory_bandwidth_gb_per_second" => peaks.bandwidth,
        "peak_compute_gflops" => peaks.compute,
        "peak_source" => "measured: $FT STREAM triad and FMA-chain kernels",
    )
end

function measure(case::BenchmarkCase, samples, metadata)
    state = case.setup()
    case.validate(case.run(state))

    setup = case.setup
    run_case = case.run
    validate = case.validate
    trial = @be setup run_case validate evals = 1 samples = samples seconds = Inf
    timings = [sample.time for sample in trial.samples]
    median_time = median(timings)
    model = case.performance_model
    arithmetic_intensity = model.flops / model.memory_bytes
    flops_per_second = model.flops / median_time

    return Dict{String, Any}(
        "name" => case.name,
        "group" => case.group,
        "unit" => "s",
        "value" => median_time,
        "time_min_seconds" => minimum(timings),
        "time_median_seconds" => median_time,
        "time_mean_seconds" => mean(timings),
        "time_max_seconds" => maximum(timings),
        "time_iqr_seconds" => quantile(timings, 0.75) - quantile(timings, 0.25),
        "samples" => length(timings),
        "allocations" => round(Int, minimum(sample -> sample.allocs, trial.samples)),
        "bytes" => round(Int, minimum(sample -> sample.bytes, trial.samples)),
        "work_units" => case.work_units,
        "work_unit" => case.work_unit,
        "throughput_per_second" => case.work_units / median_time,
        "modeled_flops" => model.flops,
        "modeled_memory_bytes" => model.memory_bytes,
        "arithmetic_intensity_flops_per_byte" => arithmetic_intensity,
        "effective_flops_per_second" => flops_per_second,
        "effective_gflops_per_second" => flops_per_second / 1.0e9,
        "effective_bandwidth_gb_per_second" => model.memory_bytes / median_time / 1.0e9,
        "performance_metric_source" => "algorithmic_model",
        "performance_model_version" => PERFORMANCE_MODEL_VERSION,
        "performance_model_description" => model.description,
        "sanity_check" => "passed",
        "parameters" => case.parameters,
        "metadata" => metadata,
    )
end

function run_benchmarks(;
        backend = JustPIC.CPU, backend_name = "CPU", device = nothing,
        samples = 10, group = "all", precision::Type = Float64,
        cases = benchmark_cases(backend; precision),
    )
    samples > 0 || throw(ArgumentError("samples must be positive"))
    selected = group == "all" ? cases : filter(case -> case.group == group, cases)
    isempty(selected) && throw(ArgumentError("unknown or empty benchmark group: $group"))
    metadata = benchmark_metadata(backend, backend_name, device, precision)
    return map(selected) do case
        @info "Benchmarking $(case.name)" samples
        measure(case, samples, metadata)
    end
end

function write_results(path, results)
    output = abspath(path)
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, results, 2)
        write(io, '\n')
    end
    return output
end

function dashboard_run(path)
    results = JSON.parsefile(path)
    results isa Vector || error("dashboard input $path must contain a JSON array")
    isempty(results) && error("dashboard input $path contains no benchmark results")

    names = getindex.(results, "name")
    allunique(names) || error("dashboard input $path contains duplicate benchmark names")
    metadata = results[1]["metadata"]
    required = ("commit", "timestamp_utc", "backend", "hardware_fingerprint", "dirty")
    for key in required
        haskey(metadata, key) || error("dashboard input $path is missing metadata.$key")
    end
    for result in results
        result["metadata"] == metadata ||
            error("dashboard input $path mixes results from different benchmark runs")
    end

    return Dict{String, Any}(
        "source" => basename(normpath(path)),
        "commit" => metadata["commit"],
        "timestamp_utc" => metadata["timestamp_utc"],
        "backend" => metadata["backend"],
        "hardware_fingerprint" => metadata["hardware_fingerprint"],
        "dirty" => metadata["dirty"],
        "benchmarks" => results,
    )
end

function dashboard_data(inputs)
    isempty(inputs) && throw(ArgumentError("at least one dashboard input is required"))
    runs = dashboard_run.(inputs)
    identities = [(run["commit"], run["backend"], run["hardware_fingerprint"]) for run in runs]
    allunique(identities) || error(
        "dashboard inputs contain more than one run for the same commit, backend, and hardware",
    )
    sort!(runs; by = run -> run["timestamp_utc"])
    return Dict{String, Any}(
        "schema_version" => 1,
        "generated_at_utc" => string(now(UTC)),
        "repository_url" => REPOSITORY_URL,
        "runs" => runs,
    )
end

write_dashboard_data(path, inputs) = write_results(path, dashboard_data(inputs))

format_seconds(t) = t < 1.0e-3 ? "$(round(t * 1.0e6; sigdigits = 3)) μs" :
    t < 1 ? "$(round(t * 1.0e3; sigdigits = 3)) ms" : "$(round(t; sigdigits = 3)) s"

relative_spread(result) = result["time_iqr_seconds"] / result["time_median_seconds"]

format_timing(result) =
    "$(format_seconds(result["time_median_seconds"])) ±$(round(Int, 100 * relative_spread(result)))%"

"""
    print_comparison(io, baseline, candidate)

Print a markdown table of the median time of each benchmark in `candidate` relative to
`baseline`, both result vectors as written by [`write_results`](@ref). The spread is the
interquartile range as a percentage of the median. A ratio that differs from 1 by more than
the two spreads combined is marked 🔴 (slower) or 🟢 (faster). Returns the
candidate-to-baseline median ratios.
"""
function print_comparison(io::IO, baseline, candidate)
    base_meta, cand_meta = baseline[1]["metadata"], candidate[1]["metadata"]
    for key in ("backend", "float_type", "hardware_fingerprint")
        base_meta[key] == cand_meta[key] || error(
            "cannot compare runs with different $key: $(repr(base_meta[key])) vs $(repr(cand_meta[key]))",
        )
    end
    getindex.(baseline, "name") == getindex.(candidate, "name") ||
        error("baseline and candidate ran different benchmarks")

    ratios = map((b, c) -> c["time_median_seconds"] / b["time_median_seconds"], baseline, candidate)
    rows = map(baseline, candidate, ratios) do base, cand, ratio
        noise = relative_spread(base) + relative_spread(cand)
        marker = ratio > 1 + noise ? " 🔴" : ratio < 1 - noise ? " 🟢" : ""
        [
            cand["name"], format_timing(base), format_timing(cand),
            "$(base["allocations"]) → $(cand["allocations"])", "$(round(ratio; digits = 2))$marker",
        ]
    end
    header = ["Benchmark", "Baseline", "Candidate", "Allocations", "Ratio"]
    widths = [maximum(length, getindex.([[header]; collect(rows)], j)) for j in eachindex(header)]
    cells(row) = join((j == 1 ? rpad(x, w) : lpad(x, w) for (j, (x, w)) in enumerate(zip(row, widths))), " | ")
    label(meta) = "`$(first(meta["commit"], 8))`" * (meta["dirty"] ? " (dirty)" : "")

    println(io, "Baseline $(label(base_meta)) vs candidate $(label(cand_meta)) on `$(cand_meta["hardware_fingerprint"])`")
    println(io)
    println(io, "| ", cells(header), " |")
    println(io, "| :", "-"^(widths[1] - 1), " | ", join(("-"^(w - 1) * ":" for w in widths[2:end]), " | "), " |")
    foreach(row -> println(io, "| ", cells(row), " |"), rows)
    return ratios
end

const PRECISIONS = Dict("Float64" => Float64, "Float32" => Float32)

function parse_commandline(args)
    output = "benchmark_results.json"
    samples = 10
    group = "all"
    precision = Float64
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--group=")
            group = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--precision=")
            name = split(arg, '='; limit = 2)[2]
            precision = get(PRECISIONS, name) do
                throw(ArgumentError("unknown precision $(repr(name)); use --precision=Float64|Float32"))
            end
        else
            throw(
                ArgumentError(
                    "unknown argument $arg; use --output=PATH, --samples=N, --group=NAME, or --precision=TYPE"
                )
            )
        end
    end
    return (; output, samples, group, precision)
end

function main(args = ARGS; backend = JustPIC.CPU, backend_name = "CPU", device = nothing)
    options = parse_commandline(args)
    results = run_benchmarks(;
        backend, backend_name, device, options.samples, options.group, options.precision
    )
    output = write_results(options.output, results)
    @info "Wrote benchmark results" output
    return results
end

function parse_dashboard_commandline(args)
    output = "benchmark_history.json"
    inputs = String[]
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--input=")
            push!(inputs, split(arg, '='; limit = 2)[2])
        else
            throw(ArgumentError("unknown argument $arg; use --input=PATH or --output=PATH"))
        end
    end
    isempty(inputs) && push!(inputs, "benchmark_results.json")
    return (; output, inputs)
end

function dashboard_main(args = ARGS)
    options = parse_dashboard_commandline(args)
    output = write_dashboard_data(options.output, options.inputs)
    @info "Wrote benchmark history" output
    return output
end

end
