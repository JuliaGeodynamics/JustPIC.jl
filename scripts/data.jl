using MAT

function load_benchmark_data(filename)
    params = matread(filename)
    return params["Vxp"], params["Vyp"]
end

function save_timestep!(fname, p, t)
    matwrite(
        fname, Dict("pX" => Array(p[1]), "pY" => Array(p[2]), "time" => t); compress=true
    )
    return nothing
end

function foo(i)
    if i < 10
        return "0000$i"

    elseif 10 ≤ i ≤ 99
        return "000$i"

    elseif 100 ≤ i ≤ 999
        return "00$i"

    elseif 1000 ≤ i ≤ 9999
        return "0$i"
    end
end

function plot(x, y, T, particles, pT, it)
    pX, pY = Array.(particles.coords)
    pidx = Array(particles.index)
    ii = findall(x -> x == true, pidx)

    T = T[2:(end - 1), 2:(end - 1)]
    cmap = :batlow

    f = Figure(; resolution=(900, 450))
    ax1 = Axis(f[1, 1])
    scatter!(ax1, pX[ii], pY[ii]; color=Array(pT[ii]), colorrange=(0, 1), colormap=cmap)

    ax2 = Axis(f[1, 2])
    hm = heatmap!(ax2, x, y, Array(T); colorrange=(0, 1), colormap=cmap)
    Colorbar(f[1, 3], hm)

    hideydecorations!(ax2)
    linkaxes!(ax1, ax2)
    for ax in (ax1, ax2)
        xlims!(ax, 0, 10)
        ylims!(ax, 0, 10)
    end

    fi = foo(it)
    fname = joinpath("imgs", "fig_$(fi).png")
    save(fname, f)

    return f
end
