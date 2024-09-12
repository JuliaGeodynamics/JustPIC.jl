using GLMakie
using CairoMakie; CairoMakie.activate!()

cx = [0, 1, 0, 1]
cy = [0, 0, 1, 1]

# Cell boundaries
f = Figure()
ax = Axis(f[1, 1], aspect = DataAspect())

lines!(
    ax,
    [cx[1], cx[2]], [cy[1], cy[2]],
    color = :black,
    strokewidth = 2,
    strokecolor = :black,
)

lines!(
    [1, 1], [0, 1],
    color = :black,
    strokewidth = 2,
    strokecolor = :black,
)

lines!(
    [0, 1], [1, 1],
    color = :black,
    strokewidth = 2,
    strokecolor = :black,
)

lines!(
    [0, 0], [0, 1],
    color = :black,
    strokewidth = 2,
    strokecolor = :black,
)

# Cell vertices
scatter!(ax, cx, cy, color = :black, markersize=25, marker = :rect) # cell vertices
text!(ax, -0.18, -0.1; text = L"C_{xy}")
text!(ax, 1+0.06, -0.1; text = L"C_{xy}")
text!(ax, -0.18, 1.04; text = L"C_{xy}")
text!(ax, 1+0.06, 1.04; text = L"C_{xy}")

# Cell centers
scatter!(ax, [0.5], [0.5], color = :black, markersize=25, marker = :rect) # cell vertices
text!(ax, 0.575, 0.46; text = L"P, \eta")

# Velocity - Vx
lines!(
    ax,
    [-5e-2, 5e-2], [0.5, 0.5],
    color = :black,
    linewidth = 5,
)
text!(ax, -0.18, 0.46; text = L"V_{x}")

lines!(
    ax,
    [1-5e-2, 1+5e-2], [0.5, 0.5],
    color = :black,
    linewidth = 5,
)
text!(ax, 1.06, 0.46; text = L"V_{x}")

text!(ax, -0.25, -0.54; text = L"(V_{x})")
text!(ax,  1.1, -0.54; text = L"(V_{x})")

text!(ax, -0.25, 1.46; text = L"(V_{x})")
text!(ax,  1.1, 1.46; text = L"(V_{x})")

# Velocity - Vy
lines!(
    ax,
    [0.5, 0.5], [1-5e-2, 1+5e-2], 
    color = :black,
    linewidth = 5,
)
text!(ax, 0.5 - 5e-2, 0-0.18; text = L"V_{y}")

lines!(
    ax,
    [0.5, 0.5], [-5e-2, 5e-2], 
    color = :black,
    linewidth = 5,
)
text!(ax, 0.5 - 5e-2, 1+0.1; text = L"V_{y}")
# ghost nodes
text!(ax, -0.5 - 7e-2, 1+0.1; text = L"\left( V_{y} \right)")
text!(ax, -0.5 - 7e-2, -0.18; text = L"\left( V_{y} \right)")
text!(ax, 1.5 - 7e-2, 1+0.1; text = L"\left( V_{y} \right)")
text!(ax, 1.5 - 7e-2, -0.18; text = L"\left( V_{y} \right)")

# Velocity - Vx ghost nodes
lines!(
    ax,
    [-5e-2, 5e-2], [-0.5, -0.5],
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [-5e-2, 5e-2], [1+0.5, 1+0.5],
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [-5e-2, 5e-2], [-0.5, -0.5],
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [1-5e-2, 1+5e-2], [1+0.5, 1+0.5],
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [1-5e-2, 1+5e-2], [-0.5, -0.5],
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)

# Velocity - Vy ghost nodes
lines!(
    ax,
    [-0.5, -0.5], [1-5e-2, 1+5e-2], 
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [-0.5, -0.5], [-5e-2, 5e-2], 
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [1+0.5, 1+0.5], [1-5e-2, 1+5e-2], 
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)
lines!(
    ax,
    [1+0.5, 1+0.5], [-5e-2, 5e-2], 
    color = :black,
    linestyle = :dash,
    linewidth = 2,
)

hidedecorations!(ax)
hidespines!(ax)

f

save("staggered_grid.pdf", f)