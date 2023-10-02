struct Particles{N,M,I,T1,T2,T3}
    coords::NTuple{N,T1}
    index::T2
    inject::T3
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
        coords::NTuple{N,T1},
        index,
        inject,
        nxcell::I,
        max_xcell::I,
        min_xcell::I,
        np::I,
        nxi,
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        T3 = typeof(inject)

        return new{N,max_xcell,I,T1,T2,T3}(
            coords, index, inject, nxcell, max_xcell, min_xcell, np
        )
    end
end
