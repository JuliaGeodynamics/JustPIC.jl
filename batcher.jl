
function main()

    isGPU = false, true

    for GPU in isGPU
        jobname = GPU ? "CUDA_perf2D" : "CPU_perf2D"
        nt = GPU ? (1:1) : (1, 4, 8, 16, 32, 64)

        for ntᵢ in nt
            str = 
"#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --nodes=1
#SBATCH --output=out_vep.o
#SBATCH --error=er_vep.e
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account c23

srun /users/ademonts/.julia/juliaup/julia-1.10.5+0.aarch64.linux.gnu/bin/julia --project=. -p 1 -t $(ntᵢ) -O3 --check-bounds=no  perf2D.jl $(GPU)"
        
            open("runme_test.sh", "w") do io
                println(io, str)
            end

            # Submit the job
            run(`sbatch runme_test.sh`)
            println("Job submitted")
            # remove the file
            sleep(1)
            rm("runme_test.sh")
            println("File removed")
        end

    end
end

main()