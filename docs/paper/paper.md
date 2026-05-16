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
date: 14 May 2026
bibliography: paper.bib
---

# Summary

Particle-in-cell (PIC) and marker-in-cell methods combine particles that carry material information with an Eulerian grid on which equations are solved. In geodynamics, this makes it possible to follow temperature, composition, phase identity, and interfaces through strongly deforming flows without excessive numerical diffusion [@GeryaYuen2003; @Moresi2003]. `JustPIC.jl` provides these particle operations as a reusable Julia package for two- and three-dimensional structured Cartesian grids with staggered velocities.

The package exposes one high-level workflow across CPUs, NVIDIA GPUs, and AMD GPUs. It includes particle and passive-marker containers, Euler and Runge-Kutta advection, semi-Lagrangian field advection, grid-to-particle and particle-to-grid interpolation, particle injection and cleaning, marker chains for interfaces, phase-ratio reconstruction at cell centers, vertices, and staggered locations, checkpointing, and MPI halo exchange for domain-decomposed simulations.

## Statement of need

Many geodynamics codes contain PIC implementations that are tightly coupled to a solver, memory layout, or accelerator backend. This makes particle methods difficult to reuse, test, and maintain. Moving a CPU prototype to GPU production runs often requires rewriting kernels, while MPI execution adds bookkeeping for particle ownership and halo exchange. These costs are substantial in research workflows where users frequently change rheologies, phase definitions, or interpolation strategies.

`JustPIC.jl` addresses this gap by separating particle mechanics from the governing-equation solver. Its target users are developers of geodynamics and fluid-dynamics models who need tested particle infrastructure but still want direct control over the surrounding numerical model. The package can be used for stand-alone advection experiments or embedded in larger Julia simulation codes.

This scope is intentionally practical. A typical model defines a staggered velocity grid, initializes particles or passive markers, advects them with a chosen time integrator, updates particle ownership and density, and then reconstructs fields or phase fractions on the grid. `JustPIC.jl` packages those repeated operations so that new models can focus on physics choices rather than rebuilding particle infrastructure.

## State of the field

Established geodynamics frameworks such as Underworld2 [@Mansour2020], ASPECT [@Bangerth2017], LaMEM [@Kaus2016], and application-specific marker-in-cell codes provide full modeling environments that include solvers, material models, and particle or compositional advection. These systems are powerful, but their particle machinery is usually part of the host code. `JustPIC.jl` instead focuses on the particle layer itself. The design choice is to complement, rather than replace, complete modeling frameworks: users can pair `JustPIC.jl` with their own finite-difference or stencil solvers, including multi-xPU Julia codes built with `ParallelStencil.jl` and `ImplicitGlobalGrid.jl` [@Omlin2024].

## Software design

`JustPIC.jl` is organized around backend-parametric particle containers and small composable kernels. Particle coordinates and particle-carried fields are stored in `CellArray` objects, with particles grouped by parent cell to preserve spatial locality during repeated advection. The container also stores derived center, vertex, and staggered velocity grids, allowing common operations such as `advection!`, `move_particles!`, `grid2particle!`, `particle2grid!`, `inject_particles!`, and `update_phase_ratios!` to use concise call forms.

The main trade-off is specialization to structured Cartesian grids. This keeps the API close to staggered-grid geodynamics solvers and enables CPU/GPU portability without maintaining separate user-facing implementations. Interpolation kernels are written generically for two and three dimensions, while package extensions provide CUDA and AMDGPU array support. Distributed runs use `ImplicitGlobalGrid.jl` for regular domain decompositions and explicit halo updates of particle coordinates, indices, and particle fields.

The package also distinguishes three particle use cases. `Particles` represent material points coupled back to the Eulerian mesh; `PassiveMarkers` track user-defined trajectories and field histories without feedback; and `MarkerChain` represents connected interfaces such as topography or material boundaries. This separation keeps common operations lightweight while supporting the interface-tracking and phase-reconstruction tasks needed in multiphase geodynamic models.

## Research impact statement

`JustPIC.jl` is released under the MIT license, archived on Zenodo, documented with worked examples, and tested through CPU, GPU, quality-assurance, formatting, documentation, and downstream continuous-integration workflows. The repository includes reproducible examples for two- and three-dimensional field advection, MPI advection, marker-chain tracking, mixed CPU/GPU usage, and checkpoint/restart. These materials make the package reviewable as a stand-alone research-software contribution and usable as a building block for ongoing JuliaGeodynamics and GPU4GEO modeling workflows.

Its near-term significance is strongest for researchers developing high-performance geodynamic solvers in Julia. By providing a shared particle layer, `JustPIC.jl` reduces duplicated implementations across projects and makes comparisons between advection, interpolation, and hardware backends easier to reproduce. The public tests exercise core 2D and 3D operations, interpolation kernels, integrators, checkpointing, and array-container behavior, providing a foundation for extension by other groups.

## AI usage disclosure

The present paper text was revised with assistance from OpenAI Codex for structure, copy-editing, and alignment with JOSS paper requirements. The authors reviewed and edited the resulting text, remain responsible for all claims, and made the software design and scientific decisions.

## Acknowledgements

The development of `JustPIC.jl` has been supported by the GPU4GEO and ∂GPU4GEO projects of the Platform for Advanced Scientific Computing (PASC).

## References
