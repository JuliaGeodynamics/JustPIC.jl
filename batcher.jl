
function main2D()

    isGPU = false, true

    for GPU in isGPU
        jobname = GPU ? "CUDA_perf2D" : "CPU_perf2D"
        nt = GPU ? (1:1) : (1, 4, 8, 16, 32, 64, 128)

        for ntᵢ in nt
            str = 
"#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --nodes=1
#SBATCH --output=out_vep.o
#SBATCH --error=er_vep.e
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account c23

export MPICH_GPU_SUPPORT_ENABLED=1
 
export JULIAUP_DEPOT_PATH=$SCRATCH/$CLUSTER_NAME/juliaup/depot
export JULIA_DEPOT_PATH=$SCRATCH/$CLUSTER_NAME/juliaup/depot
export PATH="$SCRATCH/$CLUSTER_NAME/juliaup/bin:$PATH"

srun --gpu-bind=per_task:1 --cpu_bind=sockets julia -t $(ntᵢ) -O3 --project perf2D.jl $(GPU)"
        
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

function main3D()

    isGPU = false, true

    for GPU in isGPU
        jobname = GPU ? "CUDA_perf3D" : "CPU_perf3D"
        nt = GPU ? (1:1) : (1, 4, 8, 16, 32, 64, 128)

        for ntᵢ in nt
            str = 
"#!/bin/bash -l
#SBATCH --job-name=\"$(jobname)\"
#SBATCH --nodes=1
#SBATCH --output=out_vep.o
#SBATCH --error=er_vep.e
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --account c23

export MPICH_GPU_SUPPORT_ENABLED=1
 
export JULIAUP_DEPOT_PATH=$SCRATCH/$CLUSTER_NAME/juliaup/depot
export JULIA_DEPOT_PATH=$SCRATCH/$CLUSTER_NAME/juliaup/depot
export PATH="$SCRATCH/$CLUSTER_NAME/juliaup/bin:$PATH"

srun --gpu-bind=per_task:1 --cpu_bind=sockets julia -t $(ntᵢ) -O3 --project perf3D.jl $(GPU)"
         
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

main2D()