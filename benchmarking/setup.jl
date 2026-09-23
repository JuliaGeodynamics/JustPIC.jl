using Pkg

const BENCHMARK_PROJECT = realpath(@__DIR__)
const ACTIVE_PROJECT = realpath(dirname(Base.active_project()))

ACTIVE_PROJECT == BENCHMARK_PROJECT ||
    error("activate the benchmark environment with --project=benchmarking")

Pkg.develop(; path = dirname(@__DIR__))
Pkg.instantiate()
