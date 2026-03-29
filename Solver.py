import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def solve_u(px, py, w,
            nleft, nright, nbottom, ntop,
            nboundaries, npoints, neighbors,
            tol, max_iter, omega):

    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.int64)

    u = np.zeros(npoints, dtype=float)

    # boundary values
    u[:nleft] = py[:nleft]**2
    u[nleft:nright] = 1.0 + py[nleft:nright]**2
    u[nright:ntop] = 1.0 + px[nright:ntop]**2
    u[ntop:nbottom] = np.sin(np.pi * px[ntop:nbottom] / 2.0) + py[ntop:nbottom]**2

    RHS = np.zeros(npoints, dtype=float)
    RHS[:nboundaries] = u[:nboundaries]
    RHS[nboundaries:] = 4.0

    interior = np.arange(nboundaries, npoints, dtype=np.int64)
    nloc = neighbors.shape[1]

    nbr_int = neighbors[interior]

    rows_bc = np.arange(nboundaries, dtype=np.int64)
    cols_bc = np.arange(nboundaries, dtype=np.int64)
    data_bc = np.ones(nboundaries, dtype=float)

    rows_int = np.repeat(interior, nloc)
    cols_int = nbr_int.ravel()

    residual = []
    error = []

    diffusive = w[interior, 2, :] + w[interior, 3, :]

    for k in range(max_iter):
        u_old = u.copy()

        advective = u_old[interior, None] * (w[interior, 0, :] + w[interior, 1, :])
        val = diffusive + advective 

        rows = np.concatenate([rows_bc, rows_int, interior])
        cols = np.concatenate([cols_bc, cols_int, interior])
        data = np.concatenate([
            data_bc,
            val.ravel(),
            -2.0 * (px[interior] + py[interior])
        ])

        LHS = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(npoints, npoints)).tocsr()

        u_new = scipy.sparse.linalg.spsolve(LHS, RHS)
        #u_new, code = scipy.sparse.linalg.bicgstab(LHS, RHS, rtol=1e-15, maxiter=100)
        u = (1.0 - omega) * u_old + omega * u_new

        delta_u = np.max(u) - np.min(u)
        if abs(delta_u) < 1e-14:
            delta_u = 1.0

        u_nbr = u[nbr_int]
        ux  = np.sum(w[interior, 0, :] * u_nbr, axis=1)
        uy  = np.sum(w[interior, 1, :] * u_nbr, axis=1)
        uxx = np.sum(w[interior, 2, :] * u_nbr, axis=1)
        uyy = np.sum(w[interior, 3, :] * u_nbr, axis=1)

        R = uxx + uyy + u[interior] * (ux + uy) - 2.0 * (px[interior] + py[interior]) * u[interior] - 4.0

        diag = LHS.diagonal().copy()
        diag[np.abs(diag) < 1e-14] = 1.0
        r = R / (diag[interior] * delta_u)
        rs = np.sqrt(np.mean(r**2))

        err = u - u_old
        err = np.sqrt(np.mean(err[interior]**2))

        print(k, rs, err)
        residual.append(rs)
        error.append(err)

        if rs < tol and err < tol:
            return u, residual, error, k

    return u, residual, error, max_iter