# Solving Stokes and continuity equations
# in primitive variable formulation
# with variable viscosity
# using Finite Differences (FD) on a staggered grid

using LinearAlgebra
using SparseArrays
# Advection scheme options:
# 1 = linear
# 2 = (Jenny, 2001)
# 3 = (Gerya, 2019)
# 4 = (Gerya, 2020)

# Define Numerical model
xsize = 100.0  # Horizontal model size, m
ysize = 100.0  # Vertical model size, m
Nx = 41        # Horizontal grid resolution
Ny = 41        # Vertical grid resolution
Nx1 = Nx + 1
Ny1 = Ny + 1
dx = xsize / (Nx - 1)  # Horizontal grid step, m
dy = ysize / (Ny - 1)  # Vertical grid step, m

x = range(0, stop=xsize, length=Nx)   # Horizontal coordinates of basic grid points, m
y = range(0, stop=ysize, length=Ny)   # Vertical coordinates of basic grid points, m
xvx = range(0, stop=xsize, length=Nx) # vx grid points
yvx = range(-dy/2, stop=ysize + dy/2, length=Ny1) # vx grid points
xvy = range(-dx/2, stop=xsize + dx/2, length=Nx1) # vy grid points
yvy = range(0, stop=ysize, length=Ny) # vy grid points
xp = range(-dx/2, stop=xsize + dx/2, length=Nx1) # Pressure grid points
yp = range(-dy/2, stop=ysize + dy/2, length=Ny1) # Pressure grid points

# Define velocity, pressure, and material property arrays
vx = zeros(Ny1, Nx1) # Vx, m/s
vy = zeros(Ny1, Nx1) # Vy, m/s
vxp = zeros(Ny1, Nx1) # Vx in pressure nodes, m/s
vyp = zeros(Ny1, Nx1) # Vy in pressure nodes, m/s
pr = zeros(Ny1, Nx1) # Pressure, Pa
gy = 0.0 # Gravity acceleration, m/s^2
RHO = zeros(Ny, Nx) # Density, kg/m^3
ETA = zeros(Ny, Nx) # Viscosity, Pa*s

# Define markers
Nxmc = 2  # Number of markers per cell in x-direction
Nymc = 2  # Number of markers per cell in y-direction
Nxm = (Nx - 1) * Nxmc  # Marker grid resolution in horizontal direction
Nym = (Ny - 1) * Nymc  # Marker grid resolution in vertical direction
dxm = xsize / Nxm  # Marker grid step in horizontal direction, m
dym = ysize / Nym  # Marker grid step in vertical direction, m
marknum = Nxm * Nym  # Number of markers
xm = zeros(marknum)  # Horizontal coordinates, m
ym = zeros(marknum)  # Vertical coordinates, m
rhom = zeros(marknum)  # Density, kg/m^3
etam = zeros(marknum)  # Viscosity, Pa*s

# Compose density array on markers
rp = 20000.0  # Plume radius, m
m = 1  # Marker counter

tm = zeros(marknum)  # Marker type
for jm in 1:Nxm
    for im in 1:Nym
        # Define marker coordinates
        xm[m] = dxm/2 + (jm-1)*dxm + (rand() - 0.5) * dxm
        ym[m] = dym/2 + (im-1)*dym + (rand() - 0.5) * dym
        
        # Marker properties
        if xm[m] < ym[m]
            rhom[m] = 3300  # Mantle density
            etam[m] = 1e21  # Mantle viscosity
            tm[m] = 1
        else
            rhom[m] = 3200  # Plume density
            etam[m] = 1e21  # Plume viscosity
            tm[m] = 2
        end
        
        # Update marker counter
        m += 1
    end
end

# Introducing scaled pressure
pscale = 1e21 / dx

# Define global matrices L, R
N = Nx1 * Ny1 * 3  # Global number of unknowns
L = spzeros(N, N)  # Matrix of coefficients (left part)
R = zeros(N)  # Vector of right parts

# Boundary conditions: free slip=-1; No Slip=1
bcleft = -1
bcright = -1
bctop = -1
bcbottom = -1

# Timestepping
dt = 0e12  # Initial timestep
dxymax = 0.01  # Max marker movement per timestep
visstep = 1000  # Number of steps between visualization
ntimesteps = 1000000  # Number of advection time steps

# Interpolate RHO, ETA from markers
RHOSUM = zeros(Ny, Nx)
ETASUM = zeros(Ny, Nx)
WTSUM = zeros(Ny, Nx)

# Loop over the markers
for m in 1:marknum
    # Define i,j indexes for the upper left node
    j = floor(Int, (xm[m] - x[1]) / dx) + 1
    i = floor(Int, (ym[m] - y[1]) / dy) + 1
    
    # Boundary checks for j
    if j < 1
        j = 1
    elseif j > Nx - 1
        j = Nx - 1
    end
    
    # Boundary checks for i
    if i < 1
        i = 1
    elseif i > Ny - 1
        i = Ny - 1
    end
    
    # Compute distances
    dxmj = xm[m] - x[j]
    dymi = ym[m] - y[i]
    
    # Compute weights
    wtmij = (1 - dxmj / dx) * (1 - dymi / dy)
    wtmi1j = (1 - dxmj / dx) * (dymi / dy)    
    wtmij1 = (dxmj / dx) * (1 - dymi / dy)
    wtmi1j1 = (dxmj / dx) * (dymi / dy)
    
    # Update properties for i,j Node
    RHOSUM[i, j] += rhom[m] * wtmij
    ETASUM[i, j] += etam[m] * wtmij
    WTSUM[i, j] += wtmij
    
    # Update properties for i+1,j Node
    RHOSUM[i + 1, j] += rhom[m] * wtmi1j
    ETASUM[i + 1, j] += etam[m] * wtmi1j
    WTSUM[i + 1, j] += wtmi1j
    
    # Update properties for i,j+1 Node
    RHOSUM[i, j + 1] += rhom[m] * wtmij1
    ETASUM[i, j + 1] += etam[m] * wtmij1
    WTSUM[i, j + 1] += wtmij1
    
    # Update properties for i+1,j+1 Node
    RHOSUM[i + 1, j + 1] += rhom[m] * wtmi1j1
    ETASUM[i + 1, j + 1] += etam[m] * wtmi1j1
    WTSUM[i + 1, j + 1] += wtmi1j1
end

# Compute ETA, RHO
for j in 1:Nx
    for i in 1:Ny
        if WTSUM[i, j] > 0
            RHO[i, j] = RHOSUM[i, j] / WTSUM[i, j]
            ETA[i, j] = ETASUM[i, j] / WTSUM[i, j]
        end
    end
end

# Compute viscosity in pressure nodes
ETAP = zeros(Ny1, Nx1)  # Viscosity in pressure nodes, Pa*s
for j in 2:Nx
    for i in 2:Ny
        # Harmonic average
        ETAP[i, j] = 1 / ((1 / ETA[i, j] + 1 / ETA[i, j - 1] +
                            1 / ETA[i - 1, j] + 1 / ETA[i - 1, j - 1]) / 4)
    end
end

# 3) Composing global matrices L() and R() for OMEGA
# Going through all points of the 2D grid and composing respective equations
for j in 1:Nx1
    for i in 1:Ny1
        # Define global indexes in algebraic space
        kvx = ((j - 1) * Ny1 + i - 1) * 3 + 1  # Vx
        kvy = kvx + 1  # Vy
        kpm = kvx + 2  # P
        
        # Vx equation for external points
        if i == 1 || i == Ny1 || j == 1 || j == Nx || j == Nx1
            # Boundary Condition
            # 1*Vx = 0
            L[kvx, kvx] = 1  # Left part
            R[kvx] = 0  # Right part
            # Top boundary
            if i == 1 && j > 1 && j < Nx
                L[kvx, kvx + 3] = bctop  # Left part
            end
            # Bottom boundary
            if i == Ny1 && j > 1 && j < Nx
                L[kvx, kvx - 3] = bcbottom  # Left part
            end
        # Internal BC on Diagonal Lines
        elseif i > 3 && i < Ny1 - 2 && (j == i - 0 || j == i + 2)
            # 1*Vx = 0
            L[kvx, kvx] = 1  # Left part
            if j == i - 0
                R[kvx] = 1e-9  # Right part
            else
                R[kvx] = -0e-9  # Right part
            end
        else
            # Internal points: x-Stokes equation
            # ETA * (d2Vx/dx^2 + d2Vx/dy^2) - dP/dx = 0
            ETA1 = ETA[i - 1, j]
            ETA2 = ETA[i, j]
            ETAP1 = ETAP[i, j]
            ETAP2 = ETAP[i, j + 1]
            # Left part
            L[kvx, kvx - Ny1 * 3] = 2 * ETAP1 / dx^2  # Vx1
            L[kvx, kvx - 3] = ETA1 / dy^2  # Vx2
            L[kvx, kvx] = -2 * (ETAP1 + ETAP2) / dx^2 - (ETA1 + ETA2) / dy^2  # Vx3
            L[kvx, kvx + 3] = ETA2 / dy^2  # Vx4
            L[kvx, kvx + Ny1 * 3] = 2 * ETAP2 / dx^2  # Vx5
            L[kvx, kvy] = -ETA2 / (dx * dy)  # Vy2
            L[kvx, kvy + Ny1 * 3] = ETA2 / (dx * dy)  # Vy4
            L[kvx, kvy - 3] = ETA1 / (dx * dy)  # Vy1
            L[kvx, kvy + Ny1 * 3 - 3] = -ETA1 / (dx * dy)  # Vy3
            L[kvx, kpm] = pscale / dx  # P1
            L[kvx, kpm + Ny1 * 3] = -pscale / dx  # P2
            # Right part
            R[kvx] = 0
        end
        
        # Vy equation for external points
        if j == 1 || j == Nx1 || i == 1 || i == Ny || i == Ny1
            # Boundary Condition
            # 1*Vy = 0
            L[kvy, kvy] = 1  # Left part
            R[kvy] = 0  # Right part
            # Left boundary
            if j == 1 && i > 1 && i < Ny
                L[kvy, kvy + 3 * Ny1] = bcleft  # Left part
            end
            # Right boundary
            if j == Nx1 && i > 1 && i < Ny
                L[kvy, kvy - 3 * Ny1] = bcright  # Left part
            end
        # Internal BC on Diagonal Lines
        elseif i > 3 && i < Ny - 2 && (j == i + 1 || j == i + 3)
            # 1*Vy = 0
            L[kvy, kvy] = 1  # Left part
            if j == i + 1
                R[kvy] = 1e-9  # Right part
            else
                R[kvy] = -0e-9  # Right part
            end
        else
            # Internal points: y-Stokes equation
            # ETA * (d2Vy/dx^2 + d2Vy/dy^2) - dP/dy = -RHO * gy
            ETA1 = ETA[i, j - 1]
            ETA2 = ETA[i, j]
            ETAP1 = ETAP[i, j]
            ETAP2 = ETAP[i + 1, j]
            # Density gradients
            dRHOdx = (RHO[i, j] - RHO[i, j - 1]) / dx
            dRHOdy = (RHO[i + 1, j - 1] - RHO[i - 1, j - 1] + RHO[i + 1, j] - RHO[i - 1, j]) / (dy * 4)
            # Left part
            L[kvy, kvy - Ny1 * 3] = ETA1 / dx^2  # Vy1
            L[kvy, kvy - 3] = 2 * ETAP1 / dy^2  # Vy2
            L[kvy, kvy] = -2 * (ETAP1 + ETAP2) / dy^2 - (ETA1 + ETA2) / dx^2 - dRHOdy * gy * dt  # Vy3
            L[kvy, kvy + 3] = 2 * ETAP2 / dy^2  # Vy4
            L[kvy, kvy + Ny1 * 3] = ETA2 / dx^2  # Vy5
            L[kvy, kvx] = -ETA2 / (dx * dy) - dRHOdx * gy * dt / 4  # Vx3
            L[kvy, kvx + 3] = ETA2 / (dx * dy) - dRHOdx * gy * dt / 4  # Vx4
            L[kvy, kvx - Ny1 * 3] = ETA1 / (dx * dy) - dRHOdx * gy * dt / 4  # Vx1
            L[kvy, kvx + 3 - Ny1 * 3] = -ETA1 / (dx * dy) - dRHOdx * gy * dt / 4  # Vx2
            L[kvy, kpm] = pscale / dy  # P1
            L[kvy, kpm + 3] = -pscale / dy  # P2
            # Right part
            R[kvy] = -(RHO[i, j - 1] + RHO[i, j]) / 2 * gy
        end
        
        # P equation for external points
        if i == 1 || j == 1 || i == Ny1 || j == Nx1 || (i == 2 && j == 2)
            # Boundary Condition
            # 1*P = 0
            L[kpm, kpm] = 1  # Left part
            R[kpm] = 0  # Right part
            # Real BC
            if i == 2 && j == 2
                L[kpm, kpm] = 1 * pscale  # Left part
                R[kpm] = 1e+9  # Right part
            end
        else
            # Internal points: continuity equation
            # dVx/dx + dVy/dy = 0
            # Left part
            L[kpm, kvx - Ny1 * 3] = -1 / dx  # Vx1
            L[kpm, kvx] = 1 / dx  # Vx2
            L[kpm, kvy - 3] = -1 / dy  # Vy1
            L[kpm, kvy] = 1 / dy  # Vy2
            # Right part
            R[kpm] = 0
        end
    end
end

# 4) Solving matrices and reloading the solution
S = L \ R  # Obtaining algebraic vector of solutions S()

# Reload solutions S() to vx(), vy(), p()
# Going through all grid points
for j in 1:Nx1
    for i in 1:Ny1
        # Define global indexes in algebraic space
        kvx = ((j - 1) * Ny1 + i - 1) * 3 + 1  # Vx
        kvy = kvx + 1  # Vy
        kpm = kvx + 2  # P
        # Reload solution
        vx[i, j] = S[kvx]
        vy[i, j] = S[kvy]
        pr[i, j] = S[kpm] * pscale
    end
end

# Define timestep
dt = 1e+30
maxvx = maximum(abs.(vx))
maxvy = maximum(abs.(vy))
if dt * maxvx > dxymax * dx
    dt = dxymax * dx / maxvx
end
if dt * maxvy > dxymax * dy
    dt = dxymax * dy / maxvy
end

# Compute velocity in basic nodes
# vx, vy
vxb = zeros(Ny, Nx)  # Vx, m/s
vyb = zeros(Ny, Nx)  # Vx, m/s
for j in 1:Nx
    for i in 1:Ny
        vxb[i, j] = (vx[i, j] + vx[i + 1, j]) / 2
        vyb[i, j] = (vy[i, j] + vy[i, j + 1]) / 2
    end
end

# Compute velocity in internal pressure nodes
# vx
vxp = zeros(Ny+1, Nx+1)  # Vx, m/s
for j in 2:Nx
    for i in 2:Ny
        vxp[i, j] = (vx[i, j] + vx[i, j - 1]) / 2
    end
end

# Apply BC
# Top
vxp[1, 2:Nx-1] .= -bctop * vxp[2, 2:Nx-1]
# Bottom
vxp[Ny1, 2:Nx-1] .= -bcbottom * vxp[Ny, 2:Nx-1]
# Left
vxp[:, 1] .= -vxp[:, 2]
# Right
vxp[:, Nx1] .= -vxp[:, Nx]

# vy
vyp = zeros(Ny+1, Nx+1)  # Vx, m/s
for j in 2:Nx
    for i in 2:Ny
        vyp[i, j] = (vy[i, j] + vy[i - 1, j]) / 2
    end
end    

# Apply BC
# Left
vyp[2:Ny-1, 1] .= -bcleft * vyp[2:Ny-1, 2]
# Right
vyp[2:Ny-1, Nx1] .= -bcright * vyp[2:Ny-1, Nx]  # Free slip
# Top
vyp[1, :] .= -vyp[2, :]
# Bottom
vyp[Ny1, :] .= -vyp[Ny, :]

# FINAL (Gerya, 2020)
xm = xm4
ym = ym4
# marknum = marknum4[timestep]
marknum4 = marknum

timestep=1;
for timestep=timestep:1:ntimesteps
    # Move markers with 4th order Runge-Kutta
    vxm = zeros(4)
    vym = zeros(4)

    for m in 1:marknum
        # Save initial marker coordinates
        xA = xm[m]
        yA = ym[m]
        
        for rk in 1:4
            # Interpolate vx
            # Define i,j indexes for the upper left node
            j = floor(Int, (xm[m] - xvx[1]) / dx) + 1
            i = floor(Int, (ym[m] - yvx[1]) / dy) + 1
            if j < 1
                j = 1
            elseif j > Nx - 1
                j = Nx - 1
            end
            if i < 1
                i = 1
            elseif i > Ny
                i = Ny
            end
            
            # Compute distances
            dxmj = xm[m] - xvx[j]
            dymi = ym[m] - yvx[i]
            
            # Compute weights
            # Compute vx velocity for the top and bottom of the cell
            vxm13 = vx[i, j] * (1 - dxmj / dx) + vx[i, j + 1] * dxmj / dx
            vxm24 = vx[i + 1, j] * (1 - dxmj / dx) + vx[i + 1, j + 1] * dxmj / dx
            
            # Compute correction
            if dxmj / dx >= 0.5
                if j < Nx - 1
                    vxm13 += 1 / 2 * ((dxmj / dx - 0.5)^2) * (vx[i, j] - 2 * vx[i, j + 1] + vx[i, j + 2])
                    vxm24 += 1 / 2 * ((dxmj / dx - 0.5)^2) * (vx[i + 1, j] - 2 * vx[i + 1, j + 1] + vx[i + 1, j + 2])
                end
            else
                if j > 1
                    vxm13 += 1 / 2 * ((dxmj / dx - 0.5)^2) * (vx[i, j - 1] - 2 * vx[i, j] + vx[i, j + 1])
                    vxm24 += 1 / 2 * ((dxmj / dx - 0.5)^2) * (vx[i + 1, j - 1] - 2 * vx[i + 1, j] + vx[i + 1, j + 1])
                end
            end
            
            # Compute vx
            vxm[rk] = (1 - dymi / dy) * vxm13 + (dymi / dy) * vxm24
            
            # Interpolate vy
            # Define i,j indexes for the upper left node
            j = floor(Int, (xm[m] - xvy[1]) / dx) + 1
            i = floor(Int, (ym[m] - yvy[1]) / dy) + 1
            if j < 1
                j = 1
            elseif j > Nx
                j = Nx
            end
            if i < 1
                i = 1
            elseif i > Ny - 1
                i = Ny - 1
            end
            
            # Compute distances
            dxmj = xm[m] - xvy[j]
            dymi = ym[m] - yvy[i]
            
            # Compute weights
            # Compute vy velocity for the left and right of the cell
            vym12 = vy[i, j] * (1 - dymi / dy) + vy[i + 1, j] * dymi / dy
            vym34 = vy[i, j + 1] * (1 - dymi / dy) + vy[i + 1, j + 1] * dymi / dy
            
            # Compute correction
            if dymi / dy >= 0.5
                if i < Ny - 1
                    vym12 += 1 / 2 * ((dymi / dy - 0.5)^2) * (vy[i, j] - 2 * vy[i + 1, j] + vy[i + 2, j])
                    vym34 += 1 / 2 * ((dymi / dy - 0.5)^2) * (vy[i, j + 1] - 2 * vy[i + 1, j + 1] + vy[i + 2, j + 1])
                end
            else
                if i > 1
                    vym12 += 1 / 2 * ((dymi / dy - 0.5)^2) * (vy[i - 1, j] - 2 * vy[i, j] + vy[i + 1, j])
                    vym34 += 1 / 2 * ((dymi / dy - 0.5)^2) * (vy[i - 1, j + 1] - 2 * vy[i, j + 1] + vy[i + 1, j + 1])
                end
            end
            
            # Compute vy
            vym[rk] = (1 - dxmj / dx) * vym12 + (dxmj / dx) * vym34
            
            # Change coordinates to obtain B,C,D points
            if rk == 1 || rk == 2
                xm[m] = xA + dt / 2 * vxm[rk]
                ym[m] = yA + dt / 2 * vym[rk]
            elseif rk == 3
                xm[m] = xA + dt * vxm[rk]
                ym[m] = yA + dt * vym[rk]
            end
        end
        
        # Restore initial coordinates
        xm[m] = xA
        ym[m] = yA
        
        # Compute effective velocity
        vxmeff = 1 / 6 * (vxm[1] + 2 * vxm[2] + 2 * vxm[3] + vxm[4])
        vymeff = 1 / 6 * (vym[1] + 2 * vym[2] + 2 * vym[3] + vym[4])
        
        # Move markers
        xm[m] = xm[m] + dt * vxmeff
        ym[m] = ym[m] + dt * vymeff
    end  

    xm4 = xm
    ym4 = ym
end