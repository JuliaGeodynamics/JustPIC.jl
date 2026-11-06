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

# Numerical illustrations

Figures \ref{fig:vortex} and \ref{fig:deformation} isolate particle transport and
particle–grid coupling using prescribed, nondimensional velocity fields on a
unit square and unit cube, respectively. Both examples use fourth-order
Runge–Kutta integration on the CPU, without particle injection; the generating
script checks that particle counts remain unchanged.

For the two-dimensional reversible vortex, the velocity is

$$
\mathbf{u}(x,y,t)=
\begin{pmatrix}
\sin^2(\pi x)\sin(2\pi y)\\
-\sin^2(\pi y)\sin(2\pi x)
\end{pmatrix}
\cos\left(\frac{\pi t}{3}\right),
\qquad (x,y)\in[0,1]^2,\quad 0\leq t\leq3.
$$

This field is divergence-free and has zero normal velocity at the boundaries.
Its time-dependent factor reverses sign at $t=1.5$, and its integral over the
full interval is zero, so exact trajectories retrace their paths and return to
their initial positions at $t=3$. The transported scalar starts as a smooth ring,

$$
q_0(x,y)=\exp\left[-\left(\frac{r-0.14}{0.025}\right)^2\right],
\qquad r=\sqrt{(x-0.5)^2+(y-0.72)^2}.
$$

Grid-to-particle interpolation assigns this field to sixteen particles per cell
on a $160\times160$ grid. Each particle retains its scalar value as it moves,
representing material transport governed by
$\partial_t q+\mathbf{u}\cdot\nabla q=0$. Particle-to-grid interpolation
reconstructs the three displayed snapshots. The middle panel shows stretching
into a thin filament; the final panel shows recovery after reversal. All three
panels, including the initial state, use the same reconstruction and colour
scale. The 960-step integration samples the velocity amplitude at each step
midpoint. The script checks finite reconstructions, particle return distances,
and the relative grid-field return error.

![Reversible vortex at $t=0$, $1.5$, and $3$ (left to right). Colour shows the reconstructed scalar; white contours mark $q=0.25$ and $0.65$. Faint streamlines show the spatial flow pattern; arrows indicate the initial and reversed directions. At the midpoint the instantaneous velocity is zero, so the curves provide a reference pattern without arrowheads. \label{fig:vortex}](figures/vortex_transport.png)

The three-dimensional example subjects a toroidal cloud to a reversible
stretching and folding flow,

$$
\mathbf{u}(x,y,z,t)=
\begin{pmatrix}
2\sin^2(\pi x)\sin(2\pi y)\sin(2\pi z)\\
-\sin^2(\pi y)\sin(2\pi x)\sin(2\pi z)\\
-\sin^2(\pi z)\sin(2\pi x)\sin(2\pi y)
\end{pmatrix}
\cos\left(\frac{\pi t}{2}\right),
\qquad (x,y,z)\in[0,1]^3,\quad 0\leq t\leq2.
$$

The three spatial derivatives cancel, making the prescribed flow
divergence-free, and its normal velocity vanishes on the cube boundaries.
Particles are selected from a regular lattice with eight candidates per cell
on a $48^3$ grid, retaining those initially inside the torus

$$
\left[\sqrt{(x_0-0.5)^2+(y_0-0.5)^2}-0.25\right]^2
+(z_0-0.5)^2<0.065^2.
$$

The initially compact ring develops opposing curled lobes and narrow strands as
particles move through regions of different velocity. At $t=1$, the accumulated
forward deformation is greatest: the cloud has folded out of its original plane
while its coloured bands remain attached to the material. The time-dependent
factor then changes sign, causing exact trajectories to retrace their paths and
restore the torus at $t=2$. Colour encodes each particle's initial azimuth,
$\phi_0=\operatorname{atan2}(y_0-0.5,x_0-0.5)$, allowing the rearrangement and
recovery of material to be followed across all three panels.

The run uses 480 integration steps with the amplitude sampled at step midpoints.
All 18,304 selected particles are retained, and the maximum final displacement
from their initial positions is $2.8\times10^{-9}$ unit lengths. The generating
script also checks finite, in-domain snapshots and repeats the experiment at
lower resolution in single precision. This example exercises three-dimensional
advection and the relocation of particle-associated information across cell
boundaries during substantial deformation. The rendered spheres represent
individual particles; their lighting and illustrative size make the folded
structure visible without fitting a surface to the cloud.

![Twist, fold, recover: three-dimensional particle transport at $t=0$, $1$, and $2$ (left to right). A toroidal cloud stretches into opposing curled lobes and narrow strands, then recovers as the divergence-free flow reverses. Colour tracks initial particle azimuth, showing the rearrangement and restoration of material. Faint streamlines show the spatial flow pattern, with arrows before and after reversal; at the stationary midpoint, curves are shown only as a reference pattern. The camera, coordinate limits, and colour scale are identical across panels; coordinates are normalised by the cube side length. \label{fig:deformation}](figures/deformation_3d.png)

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
