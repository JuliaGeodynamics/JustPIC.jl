---
title: "`JustPIC.jl`: A portable particle-in-cell package for Julia"
tags:
  - Julia
  - Particles-in-Cell
  - GPUs
authors:
  - name: Albert de Montserrat
    orcid: 0000-0003-1694-3735
    affiliation: 1
  - name: Pascal Aellig
    affiliation: 2
  - name: Ludovic Räss
    affiliation: 3
  - name: Thibault Duretz
    affiliation: 4
  - name: Ivan Utkin
    affiliation: 1
  - name: Boris Kaus
    affiliation: 2
affiliations:
  - name: ETH Zürich, Zürich, Switzerland
    index: 1
  - name: University of Mainz, Mainz, Germany
    index: 2
  - name: University of Lausanne, Lausanne, Switzerland
    index: 3
  - name: University of Frankfurt, Frankfurt, Germany
    index: 4
date: 14 March 2024
bibliography: paper.bib
---

# Summary

Particles-in-cell (PIC) and marker-in-cell methods are central to many geodynamics workflows. They are used to advect material properties on top of Eulerian grids, track phase interfaces, reconstruct compositional fields, and maintain sharp information during strongly deforming flows. `JustPIC.jl` is a Julia package that provides these particle operations in a form that is both high level for users and portable across hardware backends.

`JustPIC.jl` targets two- and three-dimensional structured Cartesian grids with staggered velocities, a layout common in finite-difference Stokes solvers. The package supports execution on CPUs and on NVIDIA and AMD GPUs through a single API, and it also supports distributed-memory runs with MPI. Rather than focusing only on point-particle advection, `JustPIC.jl` includes the surrounding operations required in production geodynamics models: interpolation between particles and grids, particle injection and deletion, passive markers, marker chains for material interfaces, phase-ratio reconstruction, and semi-Lagrangian field advection. The implementation builds on Julia’s performance-oriented scientific computing ecosystem and interoperates naturally with the multi-xPU stencil programming approach enabled by `ParallelStencil.jl` and `ImplicitGlobalGrid.jl` [@Omlin2024].

## Statement of need

Many research codes contain custom PIC implementations that are deeply tied to a specific solver, memory layout, or accelerator backend. This coupling makes particle methods difficult to reuse and expensive to maintain. Moving from CPU prototypes to GPU production runs often requires reimplementing core kernels, while distributed-memory execution introduces additional complexity around halo exchange and particle ownership. These issues are especially acute in geodynamics, where particle methods are commonly combined with staggered-grid Stokes solvers, phase tracking, and multiphase rheologies.

`JustPIC.jl` addresses this gap by offering a reusable PIC toolkit designed around three goals: portability, composability, and domain relevance. Portability is provided by a backend-agnostic API that runs on CPUs, CUDA GPUs, and AMD GPUs. Composability comes from its modular design: particle containers, interpolation operators, advection schemes, interface tracking, and auxiliary physics can be used independently or combined inside larger simulation codes. Domain relevance is reflected in support for staggered-grid velocities, phase-ratio reconstruction on multiple grid locations, passive tracers, and marker chains for interfaces such as free surfaces or lithological boundaries.

The package is intended both for stand-alone advection studies and as infrastructure for larger Julia geodynamics codes. It reduces the amount of application-specific particle code that users need to write, while preserving the control needed in research settings.

## Functionality and usage

`JustPIC.jl` provides several advection schemes, including Euler, second-order Runge-Kutta, fourth-order Runge-Kutta, and semi-Lagrangian variants. Grid-to-particle and particle-to-grid interpolation routines are implemented for structured staggered grids in two and three dimensions. To preserve particle density and avoid loss of information over time, the package includes particle shuffling, injection, and cleaning routines. Beyond classical PIC, `JustPIC.jl` supports passive markers and marker chains, enabling users to represent both bulk material transport and explicitly connected interfaces in the same framework.

For multiphase applications, phase ratios can be reconstructed at cell centers, vertices, and staggered locations, which is useful when coupling particle-carried material information back to Eulerian solvers. `JustPIC.jl` also includes I/O helpers for checkpointing particle states and supports MPI-based halo exchange for distributed runs on regular decomposed grids.

A typical workflow consists of defining a staggered velocity grid, initializing particles or passive markers, advecting them with one of the provided time integrators, updating particle ownership, and reconstructing fields or phase information on the mesh. The package documentation includes examples for 2D and 3D field advection, GPU execution, mixed CPU/GPU workflows, marker-chain tracking, and MPI-based periodic advection.

## Impact

`JustPIC.jl` helps bridge the gap between compact research prototypes and hardware-portable production workflows. By separating particle mechanics from application-specific governing equations, it allows developers of geodynamics and fluid-dynamics codes to reuse tested particle infrastructure instead of maintaining custom implementations for each project. The result is a more maintainable and reproducible foundation for simulations that require particle advection on modern heterogeneous hardware.

## Acknowledgements

The development of `JustPIC.jl` has been supported by the GPU4GEO project of the Platform for Advanced Scientific Computing (PASC).

## References
