---
title: 'JustPIC.jl: Particle-in-cell advection and particle–grid coupling on CPUs and GPUs'
tags:
  - Julia
  - geodynamics
  - particle-in-cell
  - advection
  - GPU computing
authors:
  - name: Albert de Montserrat
    orcid: 0000-0003-1694-3735
    affiliation: 1
  - name: Pascal S. Aellig
    orcid: 0009-0008-9039-5646
    affiliation: 2
  - name: Ludovic Räss
    orcid: 0000-0002-1136-899X
    affiliation: 3
  - name: Thibault Duretz
    orcid: 0000-0001-8472-7490
    affiliation: 4
affiliations:
  - name: ETH Zürich, Switzerland
    index: 1
  - name: Johannes Gutenberg University Mainz, Germany
    index: 2
  - name: University of Lausanne, Switzerland
    index: 3
  - name: Goethe University Frankfurt, Germany
    index: 4
date: 5 November 2026
bibliography: paper.bib
---

<!-- Working draft. See README.md for evidence and remaining author decisions.
Author list/order and affiliations follow .zenodo.json, approved for this draft.
Corresponding author and coauthor consent still need confirmation. -->

# Summary

Geodynamic simulations follow the deformation and transport of rocks over time.
A computational grid can represent velocity and temperature, while moving
particles carry information about the material, such as its composition or
thermal history. A particle-in-cell method connects these representations by
moving particles through the velocity field and transferring information between
particles and the grid.

JustPIC.jl is an open-source Julia library [@bezanson2017julia] providing these
operations for two- and three-dimensional simulations. It combines particle
storage, advection, particle–grid interpolation, and particle population
management. Shared computational kernels support CPUs and NVIDIA, AMD, and Apple
GPUs through KernelAbstractions.jl [@churavyKernelAbstractions]. Additional
components track material phase fractions, passive tracers, and free-surface
markers. JustPIC supplies the particle operations to a host application, which
provides the velocity field and the physical equations being solved.

# Statement of need

In simulations involving large deformation, particles provide a way to carry
material information without requiring a computational mesh to deform with the
rock. Using this representation requires more than integrating trajectories:
particle properties must be sampled from and returned to a grid, particles must
remain accessible after crossing cell boundaries, and regions depleted of
particles may need replenishment. These tasks recur across models with different
physical assumptions.

JustPIC targets researchers developing geodynamic solvers and application models
in Julia who need this particle functionality as a reusable component. Its
separation from the host solver lets an application combine particle transport
with its own rheology, thermal model, and boundary conditions. The same separation
allows particle operations to be tested independently of a complete simulation.
CPU execution provides a starting point for model development, while optional GPU
backends make the shared algorithms available on accelerator hardware. Backend
and precision choices remain explicit, including the restriction to single
precision on Apple Metal.

# State of the field

Particle methods are already available within established geodynamic frameworks.
Underworld2 provides a Python-accessible modelling environment based on
finite-element methods and particle-in-cell techniques [@mansour2020underworld].
ASPECT provides a finite-element framework for mantle convection
[@kronbichler2012aspect], with particle transport described in its
[particle documentation](https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/cookbooks-particles.html).
These systems combine transport with a broader modelling environment.

JustPIC addresses the narrower need for particle operations that can be embedded
in a Julia application using rectangular grids. Extending the particle machinery
inside a complete finite-element framework would retain that framework's solver
and mesh interfaces. A separate library instead exposes particle storage and
transfers directly to Julia host codes, including staggered-grid solvers. It
reuses existing infrastructure for array storage and parallel execution rather
than implementing a new device runtime. This scope supports integration into
JustRelax.jl, a Julia geodynamic solver that delegates particle-in-cell advection
to JustPIC [@demontserrat2026justrelax].

<!-- Authors: confirm that the integration rationale above reflects the actual
build-versus-contribute decision and add historical context if needed. -->

# Software design

JustPIC stores particles by their parent grid cell using CellArrays.jl. Each cell
has a fixed number of particle slots, and an occupancy mask distinguishes live
particles from unused storage. Coordinates and companion fields share this
layout. Cell-local organisation limits the search for particles contributing to
a grid location to nearby cells. The reserved capacity trades additional storage
for a bounded local representation; it must be chosen for the expected particle
distribution.

Advection and storage reorganisation are separate operations. Euler,
second-order Runge–Kutta, and fourth-order Runge–Kutta integrators update particle
coordinates using the velocity field. A subsequent movement operation restores
cell ownership and relocates companion fields with their particles. Injection
replenishes cells below a specified occupancy threshold. These stages allow a
host application to control when fields are exchanged and when its particle
population is maintained. Non-periodic boundary escape and unresolved destination
capacity can remove particles, so applications must select capacities and
boundary handling appropriate to their problem.

Transfers connect particle-carried fields to vertex and cell-centred grid fields.
Grid-to-particle interpolation samples the surrounding grid values. The nodal
particle-to-grid operation gathers weighted contributions from particles in
neighbouring cells, avoiding competing particle writes to a shared grid node.
Cell-local storage therefore supports both particle management and the transfer
algorithm. Phase-ratio routines use particle phase labels to construct material
fractions for host calculations. Free-surface tracking and checkpoint/restart
support extend this workflow without introducing a physical solver.

Computational kernels are shared across backends through KernelAbstractions.
Backend extensions supply allocation and conversion methods, while the numerical
implementation remains in the main package. This structure reduces the number
of implementations that must be kept consistent when algorithms change. It
provides execution portability; hardware-specific performance still depends on
the workload and implementation choices. Metal requires Float32, and the
documented support matrix distinguishes tested, experimental, and unsupported
combinations, including limitations on distributed particle workflows.

The repository includes automated tests for particle transport, interpolation,
integrators, phase handling, surface markers, and checkpointing. CPU and GPU test
workflows, worked examples, and API documentation accompany the implementation.
An isolated benchmark environment records timings and execution metadata for
particle and interpolation workloads. These facilities provide ways to evaluate
a selected release.

Figures \ref{fig:vortex} and \ref{fig:torus} illustrate grid-field transport in
two dimensions and particle trajectories in three dimensions. Both use
prescribed velocity fields, isolating particle operations from a host physical
solver. The generating script includes checks against flow reversal and an
analytical rotation, respectively.

![Reversible vortex transport on a 160 × 160 unit-square grid. A smooth annular scalar is sampled onto a regular lattice of sixteen particles per cell, advected through a divergence-free vortex, and reconstructed at grid vertices at three times. The velocity amplitude is multiplied by cos(πt/3), sampled at step midpoints, so the flow reverses after t = 1.5. Particle integration uses RK4 over 960 steps, without injection. All panels share the scalar colour scale; the initial panel is also reconstructed from particles. \label{fig:vortex}](figures/vortex_transport.png)

![Three-dimensional particle transport under solid-body rotation. Particles selected from a regular lattice inside a torus are advanced using JustPIC on a 40³ grid through angles of 0, 0.7, and 1.4 radians about the unit vector parallel to (1, 0.4, 0.7). Colour denotes initial azimuth and follows each particle; the shapes are particle scatter plots, with no fitted surface. RK4 uses 120 steps and no injection. Axes are normalised by the unit-cube side length and show its central region. \label{fig:torus}](figures/torus_rotation.png)

# Research impact statement

JustPIC is used as the particle component of JustRelax, whose published software
paper explicitly identifies JustPIC as the provider of particle-in-cell
advection [@demontserrat2026justrelax]. That paper describes geodynamic
applications including shear localisation, subduction, and three-dimensional
plume dynamics. This integration demonstrates the intended division of work:
the host package supplies the geodynamic equations and constitutive behaviour,
while JustPIC supplies reusable particle operations.

The relationship also has a concrete software maintenance path. The JustPIC
repository contains a downstream test workflow for JustRelax, and the JustRelax
documentation provides examples using JustPIC particle containers and phase
ratios. These are inspectable integration materials, rather than an anticipated
future application. JustPIC's own documentation also supplies two- and
three-dimensional field-advection examples for examining particle transport
independently of the full host model.

Public release records document iterative development: version 0.2.2 was released
in February 2024, while version 0.7.16 was released in November 2026. Their release
notes record changes to particle movement, interpolation-related interfaces, and
backend support. The repository distributes the software under the MIT licence,
with contribution instructions and public issue tracking. The combination of a
published downstream integration and reusable examples establishes a specific
research role without relying on unmeasured performance claims or an assumed
number of external users.

<!-- Add the exact reviewed software archive DOI before submission. The existing
concept DOI is recorded in README.md; version-specific metadata remains to check. -->

# AI usage disclosure

OpenAI Codex, using GPT-6, assisted with planning and drafting this manuscript,
locating public references, and checking claims against repository files.
Assistance also included preparing the bibliography, supporting writing notes,
and the reproducible figure-generation script.

<!-- Authors: complete the history of AI use in software and documentation,
confirm tool/model versions, and add the required statement of human review,
editing, validation, and responsibility for core design decisions after those
checks have actually been performed. This draft does not assert completed review. -->

# Acknowledgements

Development of JustPIC.jl is supported by the GPU4GEO project within the Swiss
Platform for Advanced Scientific Computing (PASC).

<!-- Authors: confirm all funding, grant identifiers, sponsor involvement, and
conflicts of interest. README and .zenodo.json contain different funding detail. -->

# References
