# Run: julia --project=. scripts/phase_injection_mixed_3D.jl
# Requires the mixed-layout inject_particles_phase! prototype.
using JustPIC, CellArrays
using GLMakie

# For a GPU, load CUDA, AMDGPU or Metal and select its KernelAbstractions backend.
const backend = JustPIC.CPU
const FT = Float32
include(joinpath(@__DIR__, "refined_grid_utils.jl"))

function main()
    # A graded mesh exercises interpolation with a different spacing in every cell.
    xv = collect(range(zero(FT), one(FT); length = 5)) .^ 2
    yv = collect(range(zero(FT), one(FT); length = 6)) .^ 2
    zv = collect(range(zero(FT), one(FT); length = 7)) .^ 2
    xc, yc, zc = map(x -> (x[1:(end - 1)] .+ x[2:end]) ./ 2, (xv, yv, zv))
    grid_vx = (xv, expand_range(yc), expand_range(zc))
    grid_vy = (expand_range(xc), yv, expand_range(zc))
    grid_vz = (expand_range(xc), expand_range(yc), zv)
    particles = init_particles(backend, 8, 24, 16, grid_vx, grid_vy, grid_vz)

    for ip in 2:particles.max_xcell
        fill!(field(particles.index, ip), false)
        foreach(A -> fill!(field(A, ip), FT(NaN)), particles.coords)
    end
    n_before = count(Array(particles.index.data))
    phases, = init_cell_arrays(particles, Val(1))
    fill!(phases.data, FT(2))
    args = pT, pTghost, pC, pCghost, pPartial = init_cell_arrays(particles, Val(5))
    foreach(A -> fill!(A.data, FT(-1)), args)

    xcg, ycg, zcg = map(to_cpu, particles.xci)
    xvg, yvg, zvg = map(to_cpu, particles.xvi)
    T = TA(backend)([FT(2) + x for x in xc, y in yc, z in zc])
    Tghost = TA(backend)([FT(2) + x for x in xcg, y in ycg, z in zcg])
    C = TA(backend)([FT(3) + y for x in xv, y in yv, z in zv])
    Cghost = TA(backend)([FT(4) + x + y + z for x in xvg, y in yvg, z in zvg])
    # Ghost centers only along x; y and z contain physical centers only.
    Partial = TA(backend)([FT(5) + z for x in xcg, y in yc, z in zc])
    fields = (T, Tghost, C, Cghost, Partial)
    inject_particles_phase!(particles, phases, args, fields)

    index = to_cpu(particles.index)
    px, py, pz = map(to_cpu, particles.coords)
    values = map(to_cpu, args)
    phase_cpu = to_cpu(phases)
    added = 0
    for I in CartesianIndices(size(index)), ip in 2:particles.max_xcell
        index[I][ip] || continue
        x, y, z = px[I][ip], py[I][ip], pz[I][ip]
        # Periodic ghost centers on a graded mesh can also fall inside the domain;
        # bounded interpolation clips to each field's actual sample range.
        expected = (
            FT(2) + clamp(x, first(xc), last(xc)), FT(2) + clamp(x, first(xcg), last(xcg)),
            FT(3) + y, FT(4) + x + y + z, FT(5) + clamp(z, first(zc), last(zc)),
        )
        @assert all(j -> isapprox(values[j][I][ip], expected[j]; atol = 32eps(FT)), 1:5)
        @assert phase_cpu[I][ip] == FT(2)
        added += 1
    end
    @assert added > 0
    @assert count(Array(particles.index.data)) == n_before + added
    println("3D field sizes: ", map(size, fields))
    println("Injected $added particles; all five properties match the analytic fields.")

    # Donors (slot 1) keep the -1 sentinel and are drawn as black squares;
    # injected particles are colored by their interpolated value.
    active = [(I, ip) for I in CartesianIndices(size(index)), ip in 1:particles.max_xcell if index[I][ip]]
    x = [px[I][ip] for (I, ip) in active]
    y = [py[I][ip] for (I, ip) in active]
    z = [pz[I][ip] for (I, ip) in active]
    donor = [ip == 1 for (_, ip) in active]
    titles = ("T (centers)", "T (ghosted centers)", "C (vertices)", "C (ghosted vertices)", "Partial (x-ghosted centers)")
    fig = Figure(size = (1500, 900))
    for j in 1:5
        v = [values[j][I][ip] for (I, ip) in active]
        row, col = fldmod1(j, 3)
        ax = Axis3(fig[row, 2col - 1]; title = titles[j], aspect = :data)
        s = scatter!(ax, x[.!donor], y[.!donor], z[.!donor]; color = v[.!donor], colormap = :viridis, markersize = 5)
        scatter!(ax, x[donor], y[donor], z[donor]; color = :black, marker = :rect, markersize = 9)
        Colorbar(fig[row, 2col], s)
    end
    display(fig)
    return particles, phases, args, fig
end

main()
