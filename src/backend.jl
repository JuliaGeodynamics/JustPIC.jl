
function set_backend(new_backend::String)
    if !(
        new_backend ∈ (
            "Threads_Float64_2D",
            "Threads_Float32_2D",
            "Threads_Float64_3D",
            "Threads_Float32_3D",
            "CUDA_Float32_2D",
            "CUDA_Float64_2D",
            "CUDA_Float64_3D",
            "CUDA_Float32_3D",
            "AMDGPU_Float32_2D",
            "AMDGPU_Float64_2D",
            "AMDGPU_Float64_3D",
            "AMDGPU_Float32_3D",
        )
    )
        throw(ArgumentError("Invalid backend: \"$(new_backend)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("backend" => new_backend)
    @info("New backend set; restart your Julia session for this change to take effect!")
end

const backend = @load_preference("backend", "Threads_Float64_2D")

const TA = @static if occursin("CUDA", backend)
    using CUDA
    CUDA.allowscalar(false)

    macro myatomic(expr)
        return esc(
            quote
                CUDA.@atomic $expr
            end,
        )
    end

    CUDA.CuArray

elseif occursin("AMDGPU", backend)
    using AMDGPU
    AMDGPU.allowscalar(false)

    macro myatomic(expr)
        return esc(
            quote
                AMDGPU.@atomic $expr
            end,
        )
    end

    AMDGPU.ROCArray

else
    using Atomix
    macro myatomic(expr)
        return esc(
            quote
                Atomix.@atomic $expr
            end,
        )
    end
    Array
end

let
    s = split(backend, "_")
    device = s[1]
    precission = s[2]
    dimension = parse(Int, s[3][1])
    @eval begin
        println("Backend: $backend")
        @init_parallel_stencil($(Symbol(device)), $(Symbol(precission)), $dimension)
    end
end
