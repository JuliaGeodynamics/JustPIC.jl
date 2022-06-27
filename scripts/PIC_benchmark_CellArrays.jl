using JustPIC, CUDA, MAT, CellArrays, StaticArrays
CUDA.allowscalar(false)

# TODO
# 1. Optimize storage. Could utilize Float32 for marker coordinates
# 2. Add periodic boundaries
# 3. Introduce minimal and maximal particle count, add and remove particles

function initialize_particles!(pX,pY,pC,pT,pA,rad2,lx,ly,dx,dy,np)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,1) || iy > size(pX,2) return end
    for ipy = 1:np
        for ipx = 1:np
            ip                  = (ipx-1)*np + ipy
            xv,yv               = (ix-1)*dx, (iy-1)*dy
            field(pX,ip)[ix,iy] = (ipx-0.5)/np
            field(pY,ip)[ix,iy] = (ipy-0.5)/np
            field(pA,ip)[ix,iy] = true
            r2                  = (xv + pX[ix,iy][ip]*dx - lx/2)^2 + (yv + pY[ix,iy][ip]*dy - ly/2)^2
            field(pC,ip)[ix,iy] = r2 < rad2 ? 1.0 : 0.0
            field(pT,ip)[ix,iy] = exp(-((xv + pX[ix,iy][ip]*dx - lx/2)^2 + (yv + pY[ix,iy][ip]*dy - ly/2)^2)/rad2)
        end
    end
    return
end

function copy_particles!(pX_old,pY_old,pC_old,pT_old,pA_old,pX,pY,pC,pT,pA)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,1) || iy > size(pX,2) return end
    pX_old[ix,iy] = pX[ix,iy]
    pY_old[ix,iy] = pY[ix,iy]
    pC_old[ix,iy] = pC[ix,iy]
    pT_old[ix,iy] = pT[ix,iy]
    pA_old[ix,iy] = pA[ix,iy]
    return
end

@inline function blerp(fx,fy,v11,v12,v21,v22)
    return (v11*(1.0-fx) + v21*fx)*(1.0-fy) + (v12*(1.0-fx) + v22*fx)*fy
end

@inline wrap(i,imin,imax) = clamp(i,imin,imax)

@inline function cell_fractions(cx,cy)
    fx = cx < 0.5 ? cx + 0.5 : cx - 0.5
    fy = cy < 0.5 ? cy + 0.5 : cy - 0.5
    return fx,fy
end

@inline function adjacent_cells(cx,cy,ix,iy)
    ix1 = cx < 0.5 ? ix   : ix+1
    iy1 = cy < 0.5 ? iy   : iy+1
    ix2 = cx < 0.5 ? ix+1 : ix+2
    iy2 = cy < 0.5 ? iy+1 : iy+2
    return ix1,ix2,iy1,iy2
end

function advect!(pX,pY,pX_old,pY_old,pC,pC_old,pT,pT_old,pA,pA_old,Vx,Vy,dt,dx,dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,1) || iy > size(pX,2) return end
    nx,ny = size(pX)
    for ip = 1:prod(cellsize(pX))
        if pA_old[ix,iy][ip] == false continue end
        cx,cy  = pX_old[ix,iy][ip], pY_old[ix,iy][ip]
        fx,fy  = cell_fractions(cx,cy)
        ix1,ix2,iy1,iy2 = adjacent_cells(cx,cy,ix,iy)
        pvx    = blerp(fx,fy,Vx[ix1,iy1],Vx[ix1,iy2],Vx[ix2,iy1],Vx[ix2,iy2])
        pvy    = blerp(fx,fy,Vy[ix1,iy1],Vy[ix1,iy2],Vy[ix2,iy1],Vy[ix2,iy2])
        px_new = pX_old[ix,iy][ip] + pvx*dt/dx
        py_new = pY_old[ix,iy][ip] + pvy*dt/dy
        if px_new < 0.0 || px_new > 1.0 || py_new < 0.0 || py_new > 1.0
            field(pA,ip)[ix,iy] = false
        else
            field(pX,ip)[ix,iy] = px_new
            field(pY,ip)[ix,iy] = py_new
        end
    end
    # redundatly compute neighbors
    for ioy = -1:1
        for iox = -1:1
            inx,iny = wrap(ix+iox,1,nx), wrap(iy+ioy,1,ny)
            if inx == ix && iny == iy continue end
            for ip = 1:prod(cellsize(pX))
                if pA_old[inx,iny][ip] == false continue end
                cx,cy  = pX_old[inx,iny][ip], pY_old[inx,iny][ip]
                fx,fy  = cell_fractions(cx,cy)
                inx1,inx2,iny1,iny2 = adjacent_cells(cx,cy,inx,iny)
                pvx    = blerp(fx,fy,Vx[inx1,iny1],Vx[inx1,iny2],Vx[inx2,iny1],Vx[inx2,iny2])
                pvy    = blerp(fx,fy,Vy[inx1,iny1],Vy[inx1,iny2],Vy[inx2,iny1],Vy[inx2,iny2])
                px_new = pX_old[inx,iny][ip] + pvx*dt/dx + iox
                py_new = pY_old[inx,iny][ip] + pvy*dt/dy + ioy
                if px_new >= 0.0 && px_new <= 1.0 && py_new >= 0.0 && py_new <= 1.0
                    # find inactive particle and insert
                    for ip2 = 1:prod(cellsize(pX))
                        if pA[ix,iy][ip2] == false
                            field(pX,ip2)[ix,iy] = px_new
                            field(pY,ip2)[ix,iy] = py_new
                            field(pC,ip2)[ix,iy] = pC_old[inx,iny][ip]
                            field(pT,ip2)[ix,iy] = pT_old[inx,iny][ip]
                            field(pA,ip2)[ix,iy] = true
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
    if ix > size(pX,1) || iy > size(pX,2) return end
    nx,ny = size(pX)
    wt = 0.0
    pc = 0.0
    pt = 0.0
    for ioy = -1:1
        for iox = -1:1
            inx,iny = wrap(ix+iox,1,nx),wrap(iy+ioy,1,ny)
            if iox != 0 && ioy != 0 && inx == ix && iny == iy continue end
            for ip = 1:prod(cellsize(pC))
                if pA[inx,iny][ip] == false continue end
                cx  = pX[inx,iny][ip] + iox - 0.5
                cy  = pY[inx,iny][ip] + ioy - 0.5
                k   = max(1.0-abs(cx),0.0)*max(1.0-abs(cy),0.0)
                wt += k
                pc += k*pC[inx,iny][ip]
                pt += k*pT[inx,iny][ip]
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
    if ix > size(pX,1) || iy > size(pX,2) return end
    nx,ny = size(pX)
    for ip = 1:prod(cellsize(pT))
        if pA[ix,iy][ip] == false continue end
        cx,cy        = pX[ix,iy][ip], pY[ix,iy][ip]
        fx,fy        = cell_fractions(cx,cy)
        ix1          = cx < 0.5 ? wrap(ix-1,1,nx) : ix
        iy1          = cy < 0.5 ? wrap(iy-1,1,ny) : iy
        ix2          = cx < 0.5 ? ix : wrap(ix+1,1,nx)
        iy2          = cy < 0.5 ? iy : wrap(iy+1,1,ny)
        pT_pic       = blerp(fx,fy,T[ix1,iy1],T[ix1,iy2],T[ix2,iy1],T[ix2,iy2])
        pT_flip      = pT[ix,iy][ip] + pT_pic - blerp(fx,fy,T_old[ix1,iy1],T_old[ix1,iy2],T_old[ix2,iy1],T_old[ix2,iy2])
        field(pT,ip)[ix,iy] = pT_pic*pic_amount + pT_flip*(1.0-pic_amount)
    end
    return
end

function reseed!(pX,pY,pA,pC,pT,min_pcount,max_pcount,np)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,1) || iy > size(pX,2) return end
    pcount   = 0
    last_idx = 1
    for ip = 1:prod(cellsize(pX))
        if pA[ix,iy][ip] == true
            pcount += 1; last_idx = ip
        end
    end
    while pcount < min_pcount
        for ip = 1:prod(cellsize(pX))
            if pA[ix,iy][ip] == false
                field(pX,ip)[ix,iy] = ((min_pcount - pcount) ÷ np + 1 - 0.5)/np
                field(pY,ip)[ix,iy] = ((min_pcount - pcount) % np + 1 - 0.5)/np
                field(pA,ip)[ix,iy] = true
                field(pC,ip)[ix,iy] = 0.0
                field(pT,ip)[ix,iy] = 0.0
                pcount += 1
                break
            end
        end
    end
    while pcount > max_pcount
        if field(pA,last_idx)[ix,iy] == true
            field(pA,last_idx)[ix,iy] = false; pcount -= 1
        end
        last_idx -= 1
    end
    return
end

function convert_viz!(pX_viz,pY_viz,pX,pY,pA,dx,dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix > size(pX,1) || iy > size(pX,2) return end
    for ip = 1:prod(cellsize(pX))
        if pA[ix,iy][ip] == true
            xv,yv = (ix-1)*dx, (iy-1)*dy
            field(pX_viz,ip)[ix,iy] = xv + pX[ix,iy][ip]*dx
            field(pY_viz,ip)[ix,iy] = yv + pY[ix,iy][ip]*dy
        else
            field(pX_viz,ip)[ix,iy] = NaN
            field(pY_viz,ip)[ix,iy] = NaN
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
        "pX" => Array(pX.data),
        "pY" => Array(pY.data),
        "pC" => Array(pC.data),
        "pT" => Array(pT.data),
        "pA" => Array(pA.data),
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
    np_d    = 2
    np_e    = 2
    nt      = 1000
    nsave   = 10
    pic_amount = 0.0
    # preprocessing
    dx,dy   = lx/nx,ly/ny
    dt      = min(dx,dy)/max(abs(vx0),abs(vy0))/4
    xc,yc   = LinRange(dx/2,lx-dx/2,nx),LinRange(dy/2,ly-dy/2,ny)'
    np0     = np_d^2
    npmax   = np0 + np_e
    F64C    = SMatrix{npmax,1,Float64,npmax}
    BoolC   = SMatrix{npmax,1,Bool,npmax}
    # allocations
    pX      = CuCellArray{F64C}(undef,nx,ny)
    pY      = CuCellArray{F64C}(undef,nx,ny)
    pC      = CuCellArray{F64C}(undef,nx,ny)
    pT      = CuCellArray{F64C}(undef,nx,ny)
    pA      = CuCellArray{BoolC}(undef,nx,ny)
    pX_old  = deepcopy(pX)
    pY_old  = deepcopy(pY)
    pC_old  = deepcopy(pC)
    pT_old  = deepcopy(pT)
    pA_old  = deepcopy(pA)
    C       = CuArray{Float64}(undef,nx,ny)
    T       = CuArray{Float64}(undef,nx,ny)
    T_old   = copy(T)
    Vx      = CuArray{Float64}(Vx)
    Vy      = CuArray{Float64}(Vy)
    # visualization arrays
    pX_viz  = deepcopy(pX)
    pY_viz  = deepcopy(pY)
    # initialization
    fill!(pA,BoolC(fill(false,npmax)))
    nthreads = (16,16)
    nblocks  = ceil.(Int, (nx,ny)./nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks initialize_particles!(pX,pY,pC,pT,pA,rad2,lx,ly,dx,dy,np_d)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks p2g!(C,T,pC,pT,pX,pY,pA)
    save_static_data!("out/griddata.mat",lx,ly,dx,dy,nx,ny,np0,nt,dt)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks convert_viz!(pX_viz,pY_viz,pX,pY,pA,dx,dy)
    save_timestep!("out/step_0.mat",C,T,pX_viz,pY_viz,pC,pT,pA,0.0)
    # action
    tt = 0.0; it = 1
    @time while it <= nt
        if it == nt ÷ 2 dt = -dt end
        println(" # it = $it")
        CUDA.@sync @cuda threads=nthreads blocks=nblocks copy_particles!(pX_old,pY_old,pC_old,pT_old,pA_old,pX,pY,pC,pT,pA)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks p2g!(C,T,pC,pT,pX,pY,pA)
        T_old  .= T
        CUDA.@sync @cuda threads=nthreads blocks=nblocks advect!(pX,pY,pX_old,pY_old,pC,pC_old,pT,pT_old,pA,pA_old,Vx,Vy,dt,dx,dy)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks g2p!(pT,T,T_old,pX,pY,pA,pic_amount)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks reseed!(pX,pY,pA,pC,pT,np0,np0+5,np_d)
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
