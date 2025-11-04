---
title: "`JustPIC.jl`: A multi-XPU Particles-in-Cell package."
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

Particles-in-Cell (PiC) methods are widely used in geodynamics to advect fields defined on Eulerian grids, as well as to track the trajectories of different material phases and their associated information. `JustPIC.jl` is a Julia package that provides a flexible and efficient implementation of PiC. `JustPIC.jl` is designed with performance and portability in mind, running seamlessly on single and multiple CPUs (x86 and ARM) and on GPUs (NVIDIA and AMD).

Aside from handling the physical advection of the particles, `JustPIC.jl` also provides a set of tools to interpolate the fields defined in different locations of the Eulerian grid (vertices or cell centers) onto the particles, and vice versa. The current version of `JustPIC.jl` specializes in bi- and tri-dimensional, regular, rectangular, and Cartesian grids where the velocity field is defined in a staggered manner \autoref{fig:grid}, i.e. the velocity components are defined at the center of the cell faces, typical of Finite Difference Stokes solvers. The package is designed to be easily extensible so that irregular grids, collocated grids, and other coordinate systems can be easily implemented.

## Statement of need
It is common in computational geodynamics and fluid-dynamics to employ Lagrangian particles (markers) to advect material properties and track phase interfaces. Existing PiC tools are often tightly coupled to a particular grid type, programming model, or accelerator stack, which forces researchers to rewrite advection code when moving between CPU and GPU accelerators from different vendors, or when integrating with different solver implementations. This fragmentation slows development, complicates reproducibility, and limits the adoption of advanced hardware for large-scale geophysical simulations.

JustPIC.jl addresses this gap by providing a modular, high-performance, and hardware-portable Particles-in-Cell toolkit written in Julia. It offers common particle operations (advection, interpolations, injection and deletion) implemented for both CPU and multiple GPU backends while keeping a single, expressive API. `JustPIC.jl` also includes some operations commonly used in geodynamics, such as computing the phase-ratio at different grid locations in multi-phase simulations, or marker chains to track material interfaces.

## Features

1. Particle advection — Advect particles in 2D/3D structured Cartesian grids with staggered velocities; supports efficient interpolation between grid and particle locations.
2. Particle injection and deletion — Maintain desired particle densities by seeding new particles or removing excess ones based on user-defined criteria and domains.
3. Phase ratio calculations — Compute phase fractions at cell centers, vertices, or faces from particle properties to couple to Eulerian solvers.
4. Marker chain — Track material interfaces with connected marker chains that move consistently with the flow.
5. Passive markers — Carry user-defined attributes (e.g., provenance, phase, temperature) and advect them without feedback on the flow.
6. Portability — Single API across CPUs and multiple GPU backends (e.g., NVIDIA and AMD) to run from laptops to clusters.

## Examples

We go through the example of 3D particle advection that can be found in the [documentation](https://github.com/JuliaGeodynamics/JustPIC.jl/blob/main/docs/src/field_advection3D.md) of JustPIC.jl.

## Acknowledgements

## References
