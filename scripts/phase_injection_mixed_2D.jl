# Run: julia --project=. scripts/phase_injection_mixed_2D.jl
# Requires the mixed-layout inject_particles_phase! prototype.
using JustPIC, CellArrays
using GLMakie

# For a GPU, load CUDA, AMDGPU or Metal and select its KernelAbstractions backend.
const backend = JustPIC.CPU
const FT = Float32
include(joinpath(@__DIR__, "refined_grid_utils.jl"))

function main()
    xv = collect(range(zero(FT), one(FT); length = 6))
    yv = collect(range(zero(FT), one(FT); length = 8))
    xc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2
    yc = (yv[1:(end - 1)] .+ yv[2:end]) ./ 2
    particles = init_particles(backend, 4, 16, 8, (xv, expand_range(yc)), (expand_range(xc), yv))

    # Keep one donor per cell; empty slots force injection throughout the domain.
    for ip in 2:particles.max_xcell
        fill!(field(particles.index, ip), false)
        foreach(A -> fill!(field(A, ip), FT(NaN)), particles.coords)
    end
    n_before = count(Array(particles.index.data))
    phases, = init_cell_arrays(particles, Val(1))
    fill!(phases.data, one(FT))
    args = pT, pTghost, pC, pCghost = init_cell_arrays(particles, Val(4))
    foreach(A -> fill!(A.data, FT(-1)), args)

    xcg, ycg = map(to_cpu, particles.xci)
    xvg, yvg = map(to_cpu, particles.xvi)
    # Each source has its own shape. No layout keyword is needed.
    T = TA(backend)([FT(2) + x for x in xc, y in yc])                  # physical centers
    Tghost = TA(backend)([FT(2) + x for x in xcg, y in ycg])           # ghosted centers
    C = TA(backend)([FT(3) + y for x in xv, y in yv])                  # physical vertices
    Cghost = TA(backend)([FT(4) + x + y for x in xvg, y in yvg])       # ghosted vertices
    fields = (T, Tghost, C, Cghost)
    inject_particles_phase!(particles, phases, args, fields)

    index = to_cpu(particles.index)
    px, py = map(to_cpu, particles.coords)
    values = map(to_cpu, args)
    phase_cpu = to_cpu(phases)
    added = 0
    for I in CartesianIndices(size(index)), ip in 2:particles.max_xcell
        index[I][ip] || continue
        x, y = px[I][ip], py[I][ip]
        # Without ghost centers, boundary interpolation is bounded by the samples.
        expected = (FT(2) + clamp(x, first(xc), last(xc)), FT(2) + x, FT(3) + y, FT(4) + x + y)
        @assert all(j -> isapprox(values[j][I][ip], expected[j]; atol = 32eps(FT)), 1:4)
        @assert phase_cpu[I][ip] == one(FT)
        added += 1
    end
    @assert added > 0
    @assert count(Array(particles.index.data)) == n_before + added
    println("2D field sizes: ", map(size, fields))
    println("Injected $added particles; all four properties match the analytic fields.")

    # Donors (slot 1) keep the -1 sentinel and are drawn as black squares;
    # injected particles are colored by their interpolated value.
    active = [(I, ip) for I in CartesianIndices(size(index)), ip in 1:particles.max_xcell if index[I][ip]]
    x = [px[I][ip] for (I, ip) in active]
    y = [py[I][ip] for (I, ip) in active]
    donor = [ip == 1 for (_, ip) in active]
    titles = ("T (centers)", "T (ghosted centers)", "C (vertices)", "C (ghosted vertices)")
    fig = Figure(size = (1000, 800))
    for j in 1:4
        v = [values[j][I][ip] for (I, ip) in active]
        row, col = fldmod1(j, 2)
        ax = Axis(fig[row, 2col - 1]; title = titles[j], aspect = DataAspect())
        vlines!(ax, xv; color = :gray70)
        hlines!(ax, yv; color = :gray70)
        s = scatter!(ax, x[.!donor], y[.!donor]; color = v[.!donor], colormap = :viridis, markersize = 6)
        scatter!(ax, x[donor], y[donor]; color = :black, marker = :rect, markersize = 10)
        Colorbar(fig[row, 2col], s)
    end
    display(fig)
    return particles, phases, args, fig
end

main()
