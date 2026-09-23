```@raw html
---
aside: false
---
```

# Performance

Commit-by-commit benchmark history for JustPIC's particle workflows on the CPU and GPU
backends. Main-branch results are recorded by Buildkite on the `benchmark-data` branch; the
harness, cases, and performance model are described in
[`benchmarking/README.md`](https://github.com/JuliaGeodynamics/JustPIC.jl/tree/main/benchmarking).

Runtime is measured. Arithmetic intensity, FLOP/s, and effective bandwidth use the
versioned algorithmic model recorded with each result.

```@raw html
<PerformanceDashboard />
```
