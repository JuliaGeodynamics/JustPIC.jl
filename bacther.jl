
function main()

    isGPU = false, true

    for GPU in isGPU
        jobname = GPU ? "CUDA_perf2D" : "CPU_perf2D"
        nt = GPU ? (1:1) : (2 .^ (5:10))

        for ntᵢ in nt
            str = 
"#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --nodes=1
#SBATCH --output=out_vep.o
#SBATCH --error=er_vep.e
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account c2

srun /users/ademonts/.julia/juliaup/julia-1.10.5+0.aarch64.linux.gnu/bin/julia --project=. -p 1 -t $(ntᵢ) -O3 --check-bounds=no  perf2D.jl $(GPU)"
        
            open("runme_test.sh", "w") do io
                println(io, str)
            end
        end

        # Submit the job
        run(`sbatch runme_test.sh`)
        # remove the file
        sleep(1)
        rm("runme_test.sh")
    end
end
