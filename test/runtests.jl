using JustPIC

push!(LOAD_PATH, "..")

istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()
    testfiles = sort(
        filter(
            istest,
            vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...),
        ),
    )
    nfail = 0
    printstyled("Testing package JustPIC.jl\n"; bold=true, color=:white)

  

    # 2D tests --------------------------------------------------
    printstyled("Running 2D tests\n"; bold=true, color=:white)
    set_backend("Threads_Float64_2D") # need to restart session if this changes

    for f in testfiles
        if occursin("2D", f) 
            println("\n Running tests from $f")
            try
                run(`$exename --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                nfail += 1
            end
        end
    end

    # 3D tests --------------------------------------------------
    printstyled("Running 3D tests\n"; bold=true, color=:white)
    set_backend("Threads_Float64_3D") # need to restart session if this changes

    for f in testfiles
        if occursin("3D", f) 
            println("\n Running tests from $f")
            try
                run(`$exename --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                nfail += 1
            end
        end
    end

    return nfail
end

exit(runtests())
