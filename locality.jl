using Random
using LinearRegression
using GLMakie
using JLD2

using CUDA
using ParallelStencil
@init_parallel_stencil(CUDA,Float64,2)

function foo!(A, ind)
    for i in ind
        @inbounds A[i] += 1
    end
    return nothing
end

function foo_sort!(A, ind)
    sort!(ind)
    for i in ind
        @inbounds A[i] += 1
    end
    return nothing
end

@parallel_indices (i) function foo_ps!(A, ind)
    @inbounds A[ind[i]] += 1
    return nothing
end

function foo_sort_ps!(A, ind)
    sort!(ind)
    n = length(A)
    @parallel (1:n) foo_ps!(A, ind)

    return nothing
end

function main()
    N      = 256
    t_seq  = Float64[]
    t_rand = Float64[]
    t_sort = Float64[]
    NP     = Int64[]

    reps   = 5

    for np in (1:10)

        # A         = zeros(np * N^3)
        A         = @zeros(np * N^3)
        n         = length(A)
        ind0      = [i for i in eachindex(A)];
        ind_rand0 = deepcopy(ind0);
        shuffle!(ind_rand0);

        ind      = CUDA.CuArray(ind0)
        ind_rand = CUDA.CuArray(ind_rand0);

        # ind      = ind0
        # ind_rand = ind_rand0

        c_seq  = 0
        c_rand = 0
        c_sort = 0
        # for _ in 1:reps
        #     c_seq   += (@b foo!($A, $ind)).time
        #     c_rand  += (@b foo!($A, $ind_rand)).time
        #     c_sort  += (@b foo_sort!($A, $ind_rand)).time
        # end

        for _ in 1:reps
            c_seq   += (@b @parallel ($(1:n)) foo_ps!($A, $ind)).time
            c_rand  += (@b @parallel ($(1:n)) foo_ps!($A, $ind_rand)).time
            # c_sort  += (@b foo_sort_ps!($A, $ind_rand)).time
        end

        push!(t_seq,  c_seq)
        push!(t_rand, c_rand)
        push!(t_sort, c_sort)
        push!(NP, np * N^3)

    end

    return t_seq, t_rand, t_sort, NP
end

# t_seq, t_rand, t_sort, NP0 = main()

# jldsave("locality_GPU.jld2"; t_seq, t_rand, t_sort, NP0)
# jldsave("locality_CPU.jld2"; t_seq, t_rand, t_sort, NP0)

data_cpu = jldopen("locality_CPU.jld2");
t_seq_cpu, t_rand_cpu, t_sort_cpu, NP0 = data_cpu["t_seq"], data_cpu["t_rand"], data_cpu["t_sort"], data_cpu["NP0"];

data_gpu = jldopen("locality_GPU.jld2");
t_seq_gpu, t_rand_gpu, t_sort_gpu, NP0_gpu = data_gpu["t_seq"], data_gpu["t_rand"], data_cpu["t_sort"], data_gpu["NP0"];

NP = NP0 ./ 1e6

fig = Figure(size = (1200, 1000))
ax1  = Axis(
    fig[1, 1], aspect=1, 
    xlabel = L"\text{millions of particles} ", ylabel = L"\text{time / time sequential access}",
    xlabelsize = 35, ylabelsize = 35,
    xticklabelsize=30, yticklabelsize=30,
    # xscale = log10, 
    # yscale = log10,
)

# scatterlines!(ax1, NP, t_seq_cpu,
#     linewidth  = 5,
#     markersize = 25,
#     label      = "Sequential access",
# )

### CPU
lsort_cpu = lines!(NP, t_sort_cpu ./ t_seq_cpu,
    linewidth  = 5,
    markersize = 25,
    label      = "Sort indices and sequential access",
    color      = :blue,
)

scatter!(NP, t_sort_cpu ./ t_seq_cpu,
    linewidth  = 5,
    markersize = 25,
    color      = :blue,
)


lrand_cpu = lines!(ax1, NP, t_rand_cpu ./ t_seq_cpu,
    linewidth  = 5,
    markersize = 25,
    color      = :black,
    label      = "Random access",
)

scatter!(ax1, NP, t_rand_cpu ./ t_seq_cpu,
    linewidth  = 5,
    markersize = 25,
    color      = :black,
)

# axislegend(ax1, position = :rb, labelsize = 30)

### GPU
lrand_gpu = lines!(ax1, NP, t_rand_gpu ./ t_seq_gpu,
    linewidth  = 5,
    markersize = 25,
    color      = :black,
    linestyle  = :dash,
    label      = "Random access",
)

scatter!(ax1, NP, t_rand_gpu ./ t_seq_gpu,
    linewidth  = 5,
    markersize = 25,
    color      = :black,
)

ylims!(ax1, 1.5, 14)

leg = Legend(
    fig[1, 2],
    [
        lsort_cpu,
        lrand_cpu,
        lrand_cpu,
        lrand_gpu,
        # [li, sc] => (; color = :red),
        # [li => (; linewidth = 3), sc => (; markersize = 20)],
    ],
    ["Sort indices and sequential access", "Random access", "CPU", "GPU"],
    patchsize = (40, 20),
    labelsize = 20,
    framevisible = false,
)

leg.tellheight = false

save("locality.png", fig)
fig

# t_rand_gpu ./ t_seq_gpu

# # lr_rand =  linregress(NP, t_rand)
# # lr_seq  =  linregress(NP, t_seq)
# # LinearRegression.slope(lr_seq)
# # LinearRegression.slope(lr_rand)|