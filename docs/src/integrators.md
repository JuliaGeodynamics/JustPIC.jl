# Time integration

Advection equation

$\frac{\partial x_i}{\partial t} = v_i$

## Euler

$x^{t+1}_i = x^t_i + v_i \Delta t $

## Runge-Kutta 2
The general two-stage Runge-Kutta method is given by

$x^{t+1}_i = x^t_i + \Delta t \left( \left(1 - \frac{1}{2\alpha}\right) v_i^t(x_i^t)+ \frac{1}{2\alpha} v_i^t(x_i^t + \alpha\Delta  v_i^t(x_i^t)) \right) $

By default, JustPIC.jl uses $\alpha = 0.5$, which corresponds to the midpoint method.