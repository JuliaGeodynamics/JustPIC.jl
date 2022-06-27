using PIC, CUDA, MAT
CUDA.allowscalar(false)

# TODO
# 1. Optimize storage. Could utilize Float32 for marker coordinates
# 2. Add periodic boundaries
# 3. Introduce minimal and maximal particle count, add and remove particles

function initialize_particles!(pX,pY,pC,pT,pA,rad2,lx,ly,dx,dy,np)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    for ipy = 1:np
        for ipx = 1:np
            ip           = (ipx-1)*np + ipy
            xv,yv        = (ix-1)*dx, (iy-1)*dy
            pX[ip,ix,iy] = (ipx-0.5)/np
            pY[ip,ix,iy] = (ipy-0.5)/np
            pA[ip,ix,iy] = true
            r2           = (xv + pX[ip,ix,iy]*dx - lx/2)^2 + (yv + pY[ip,ix,iy]*dy - ly/2)^2
            pC[ip,ix,iy] = r2 < rad2 ? 1.0 : 0.0
            pT[ip,ix,iy] = exp(-((xv + pX[ip,ix,iy]*dx - lx/2)^2 + (yv + pY[ip,ix,iy]*dy - ly/2)^2)/rad2)
        end
    end
    return
end

function blerp(fx,fy,v11,v12,v21,v22)
    return (v11*(1.0-fx) + v21*fx)*(1.0-fy) + (v12*(1.0-fx) + v22*fx)*fy
end

wrap(i,imin,imax) = clamp(i,imin,imax)

function advect!(pX,pY,pX_old,pY_old,pC,pC_old,pT,pT_old,pA,pA_old,Vx,Vy,dt,dx,dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    np,nx,ny = size(pX)
    for ip = 1:np
        if pA_old[ip,ix,iy] == false continue end
        cx,cy  = pX_old[ip,ix,iy], pY_old[ip,ix,iy]
        fx     = cx < 0.5 ? cx + 0.5 : cx - 0.5
        fy     = cy < 0.5 ? cy + 0.5 : cy - 0.5
        ix1    = cx < 0.5 ? ix : ix+1
        iy1    = cy < 0.5 ? iy : iy+1
        ix2    = cx < 0.5 ? ix+1 : ix+2
        iy2    = cy < 0.5 ? iy+1 : iy+2
        pvx    = blerp(fx,fy,Vx[ix1,iy1],Vx[ix1,iy2],Vx[ix2,iy1],Vx[ix2,iy2])
        pvy    = blerp(fx,fy,Vy[ix1,iy1],Vy[ix1,iy2],Vy[ix2,iy1],Vy[ix2,iy2])
        px_new = pX_old[ip,ix,iy] + pvx*dt/dx
        py_new = pY_old[ip,ix,iy] + pvy*dt/dy
        if px_new < 0.0 || px_new > 1.0 || py_new < 0.0 || py_new > 1.0
            pA[ip,ix,iy] = false
        else
            pX[ip,ix,iy] = px_new
            pY[ip,ix,iy] = py_new
        end
    end
    # redundatly compute neighbors
    for ioy = -1:1
        for iox = -1:1
            inx,iny = wrap(ix+iox,1,nx), wrap(iy+ioy,1,ny)
            if inx == ix && iny == iy continue end
            for ip = 1:np
                if pA_old[ip,inx,iny] == false continue end
                cx,cy  = pX_old[ip,inx,iny], pY_old[ip,inx,iny]
                fx     = cx < 0.5 ? cx + 0.5 : cx - 0.5
                fy     = cy < 0.5 ? cy + 0.5 : cy - 0.5
                inx1   = cx < 0.5 ? inx : inx+1
                iny1   = cy < 0.5 ? iny : iny+1
                inx2   = cx < 0.5 ? inx+1 : inx+2
                iny2   = cy < 0.5 ? iny+1 : iny+2
                pvx    = blerp(fx,fy,Vx[inx1,iny1],Vx[inx1,iny2],Vx[inx2,iny1],Vx[inx2,iny2])
                pvy    = blerp(fx,fy,Vy[inx1,iny1],Vy[inx1,iny2],Vy[inx2,iny1],Vy[inx2,iny2])
                px_new = pX_old[ip,inx,iny] + pvx*dt/dx + iox
                py_new = pY_old[ip,inx,iny] + pvy*dt/dy + ioy
                if px_new >= 0.0 && px_new <= 1.0 && py_new >= 0.0 && py_new <= 1.0
                    # find inactive particle and insert
                    for ip2 = 1:size(pX,1)
                        if pA[ip2,ix,iy] == false
                            pX[ip2,ix,iy] = px_new
                            pY[ip2,ix,iy] = py_new
                            pC[ip2,ix,iy] = pC_old[ip,inx,iny]
                            pT[ip2,ix,iy] = pT_old[ip,inx,iny]
                            pA[ip2,ix,iy] = true
                            break
                        end
                    end
                end
            end
        end
    end
    return
end

function p2g!(C,T,pC,pT,pX,pY,pA)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    np,nx,ny = size(pX)
    wt = 0.0
    pc = 0.0
    pt = 0.0
    for ioy = -1:1
        for iox = -1:1
            inx,iny = wrap(ix+iox,1,nx),wrap(iy+ioy,1,ny)
            if iox != 0 && ioy != 0 && inx == ix && iny == iy continue end
            for ip = 1:np
                if pA[ip,inx,iny] == false continue end
                cx  = pX[ip,inx,iny] + iox - 0.5
                cy  = pY[ip,inx,iny] + ioy - 0.5
                k   = max(1.0-abs(cx),0.0)*max(1.0-abs(cy),0.0)
                wt += k
                pc += k*pC[ip,inx,iny]
                pt += k*pT[ip,inx,iny]
            end
        end
    end
    if wt > 0.0
        pc /= wt
        pt /= wt
    end
    C[ix,iy] = pc
    T[ix,iy] = pt
    return
end

function g2p!(pT,T,T_old,pX,pY,pA,pic_amount)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    np,nx,ny = size(pX)
    for ip = 1:np
        if pA[ip,ix,iy] == false continue end
        cx,cy        = pX[ip,ix,iy], pY[ip,ix,iy]
        fx           = cx < 0.5 ? cx + 0.5 : cx - 0.5
        fy           = cy < 0.5 ? cy + 0.5 : cy - 0.5
        ix1          = cx < 0.5 ? wrap(ix-1,1,nx) : ix
        iy1          = cy < 0.5 ? wrap(iy-1,1,ny) : iy
        ix2          = cx < 0.5 ? ix : wrap(ix+1,1,nx)
        iy2          = cy < 0.5 ? iy : wrap(iy+1,1,ny)
        pT_pic       = blerp(fx,fy,T[ix1,iy1],T[ix1,iy2],T[ix2,iy1],T[ix2,iy2])
        pT_flip      = pT[ip,ix,iy] + pT_pic - blerp(fx,fy,T_old[ix1,iy1],T_old[ix1,iy2],T_old[ix2,iy1],T_old[ix2,iy2])
        pT[ip,ix,iy] = pT_pic*pic_amount + pT_flip*(1.0-pic_amount)
    end
    return
end

function reseed!(pX,pY,pA,pC,pT,min_pcount,max_pcount,np)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    pcount   = 0
    last_idx = 1
    for ip = 1:size(pX,1)
        if pA[ip,ix,iy] == true
            pcount += 1; last_idx = ip
        end
    end
    while pcount < min_pcount
        for ip = 1:size(pX,1)
            if pA[ip,ix,iy] == false
                pX[ip,ix,iy] = ((min_pcount - pcount) ÷ np + 1 - 0.5)/np
                pY[ip,ix,iy] = ((min_pcount - pcount) % np + 1 - 0.5)/np
                pA[ip,ix,iy] = true
                pC[ip,ix,iy] = 0.0
                pT[ip,ix,iy] = 0.0
                pcount += 1
                break
            end
        end
    end
    while pcount > max_pcount
        if pA[last_idx,ix,iy] == true
            pA[last_idx,ix,iy] = false; pcount -= 1
        end
        last_idx -= 1
    end
    return
end

function convert_viz!(pX_viz,pY_viz,pX,pY,pA,dx,dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,2) || iy > size(pX,3) return end
    for ip = 1:size(pX,1)
        if pA[ip,ix,iy] == true
            xv,yv = (ix-1)*dx, (iy-1)*dy
            pX_viz[ip,ix,iy] = xv + pX[ip,ix,iy]*dx
            pY_viz[ip,ix,iy] = yv + pY[ip,ix,iy]*dy
        else
            pX_viz[ip,ix,iy] = NaN
            pY_viz[ip,ix,iy] = NaN
        end
    end
    return
end

function save_static_data!(fname,lx,ly,dx,dy,nx,ny,np,nt,dt)
    matwrite(fname, Dict(
        "lx" => lx,
        "ly" => ly,
        "dx" => dx,
        "dy" => dy,
        "nx" => nx,
        "ny" => ny,
        "np" => np,
        "nt" => nt,
        "dt" => dt,
    ); compress = true)
    return
end

function save_timestep!(fname,C,T,pX,pY,pC,pT,pA,time)
    matwrite(fname, Dict(
        "C"  => Array(C),
        "T"  => Array(T),
        "pX" => Array(pX),
        "pY" => Array(pY),
        "pC" => Array(pC),
        "pT" => Array(pT),
        "pA" => Array(pA),
        "time" => time,
    ); compress=true)
    return
end

function load_benchmark_data(filename)
    params = matread(filename)
    return params["Vxp"],params["Vyp"]
end

function main()
    # load data
    Vx,Vy   = load_benchmark_data("data/data41_benchmark.mat")
    # physics
    lx,ly   = 10.0,10.0
    rad2    = 2.0
    vx0,vy0 = maximum(abs.(Vx)),maximum(abs.(Vy))
    # numerics
    nx,ny   = 40,40
    np      = 2
    nt      = 1000
    nsave   = 10
    pic_amount = 0.0
    # preprocessing
    dx,dy   = lx/nx,ly/ny
    dt      = min(dx,dy)/max(abs(vx0),abs(vy0))/4
    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)'
    # allocations
    pX      = CuArray{Float64}(undef,np*np+10,nx,ny)
    pY      = CuArray{Float64}(undef,np*np+10,nx,ny)
    pC      = CuArray{Float64}(undef,np*np+10,nx,ny)
    pT      = CuArray{Float64}(undef,np*np+10,nx,ny)
    pA      = CuArray{Bool}(undef,np*np+10,nx,ny)
    pX_old  = copy(pX)
    pY_old  = copy(pY)
    pC_old  = copy(pC)
    pT_old  = copy(pT)
    pA_old  = copy(pA)
    C       = CuArray{Float64}(undef,nx,ny)
    T       = CuArray{Float64}(undef,nx,ny)
    T_old   = copy(T)
    Vx      = CuArray{Float64}(Vx)
    Vy      = CuArray{Float64}(Vy)
    # visualization arrays
    pX_viz  = copy(pX)
    pY_viz  = copy(pY)
    # initialization
    pA      .= false
    nthreads = (16,16)
    nblocks  = ceil.(Int, (nx,ny)./nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks initialize_particles!(pX,pY,pC,pT,pA,rad2,lx,ly,dx,dy,np)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks p2g!(C,T,pC,pT,pX,pY,pA)
    save_static_data!("out/griddata.mat",lx,ly,dx,dy,nx,ny,np,nt,dt)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks convert_viz!(pX_viz,pY_viz,pX,pY,pA,dx,dy)
    save_timestep!("out/step_0.mat",C,T,pX_viz,pY_viz,pC,pT,pA,0.0)
    # action
    tt = 0.0; it = 1
    while it <= nt
        if it == nt ÷ 2 dt = -dt end
        println(" # it = $it")
        pX_old .= pX
        pY_old .= pY
        pC_old .= pC
        pT_old .= pT
        pA_old .= pA
        CUDA.@sync @cuda threads=nthreads blocks=nblocks p2g!(C,T,pC,pT,pX,pY,pA)
        T_old  .= T
        CUDA.@sync @cuda threads=nthreads blocks=nblocks advect!(pX,pY,pX_old,pY_old,pC,pC_old,pT,pT_old,pA,pA_old,Vx,Vy,dt,dx,dy)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks g2p!(pT,T,T_old,pX,pY,pA,pic_amount)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks reseed!(pX,pY,pA,pC,pT,np*np,np*np+5,np)
        tt += dt
        # save resuts
        if it % nsave == 0
            CUDA.@sync @cuda threads=nthreads blocks=nblocks convert_viz!(pX_viz,pY_viz,pX,pY,pA,dx,dy)
            save_timestep!("out/step_$it.mat",C,T,pX_viz,pY_viz,pC,pT,pA,tt)
        end
        it += 1
    end
    return
end

main()
