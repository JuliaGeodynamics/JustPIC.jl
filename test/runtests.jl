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

    # text to modofy LocalPreferences.toml
    txt3D = "[JustPIC]
    backend = \"Threads_Float64_3D\""
    
    txt3D = "[JustPIC]
    backend = \"Threads_Float64_3D\""
    

    # 2D tests --------------------------------------------------
    printstyled("Running 2D tests\n"; bold=true, color=:white)
    open(joinpath(pwd(), "LocalPreferences.toml"), "w") do file
        write(file, txt2D)
    end

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
    open(joinpath(pwd(), "LocalPreferences.toml"), "w") do file
        write(file, txt3D)
    end

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
