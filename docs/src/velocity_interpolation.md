# Modified Quadratic Spline MQS

The scheme guarantee bi-linear interpolation of $\partial u_i/\partial x_i$ from pressure nodes where they are defined by solving (in)compressible continuity equation.

Example for the $u_x$ component in 2D:

*Step 1* Compute the normalized distances between particle and left-bottom corner of the cell:

$t_{x} = 1 - \frac{x_m - xc_j}{\Delta x}$

$t_{y} =  1 - \frac{y_m - yc_j}{\Delta y} $

*Step 2* Compute $u_x$ velocity with bi-linear scheme for the bottom and top

$u_{m}^{(13)} = u_{i,j} t_x + u_{i,j+1} (t_x-1)$

$u_{m}^{(23)} = u_{i+1,j} t_x + u_{i+1,j+1} (t_x-1)$

*Step 3* Compute $u_x$ of the marker with bi-linear scheme in vertical direction

$u_{m}^{(13)} = u_{m}^{(13)} + \frac{1}{2} (t_x-\frac{3}{2})^2 (u_{i,j-1}-2u_{i,j}+u_{i,j-1})$

$u_{m}^{(24)} = u_{m}^{(24)} + \frac{1}{2} (t_x-\frac{3}{2})^2 (u_{i+1,j-1}-2u_{i+1,j}+u_{i+1,j-1})$

*Step 4* Compute $u_x$  vx of the marker with bi-linear scheme in vertical direction

$u_{m} = t_y u_{m}^{(13)}+(t_y-1) u_{m}^{(24)}$