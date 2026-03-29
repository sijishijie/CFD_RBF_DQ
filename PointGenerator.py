# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:53:02 2026

@author: Adhi
"""
import numpy as np

import numpy as np

def yb_func(x):
    return 0.5 * (np.tanh(2 - 10*x) - np.tanh(2))

import numpy as np

def yb_func(x):
    return 0.5 * (np.tanh(2 - 10*x) - np.tanh(2))

def generate_points_assignment(h):
    nl = int(np.round(1 / (h/2)))
    pxl = np.zeros(nl)
    pyl = np.linspace(h/2, 1-h/2, nl)
    nleft = nl

    yb1 = yb_func(1.0)
    nr = int(np.round((1 - yb1) / (h/2)))
    pxr = np.ones(nr)
    pyr = np.linspace(yb1 + h/2, 1 - h/2, nr)
    nright = nleft + nr

    nb1 = int(np.round(0.5 / (h/8)))
    pxb1 = np.linspace(0, 0.5 - h/8, nb1)
    pyb1 = yb_func(pxb1)

    nb2 = int(np.round(0.5 / (h/2)))
    pxb2 = np.linspace(0.5, 1, nb2)
    pyb2 = yb_func(pxb2)

    nbottom = nright + nb1 + nb2

    nt = int(np.round(1 / (h/2)))
    pxt = np.linspace(0, 1, nt)
    pyt = np.ones(nt)
    ntop = nbottom + nt

    pxbd = np.hstack((pxl, pxr, pxb1, pxb2, pxt))
    pybd = np.hstack((pyl, pyr, pyb1, pyb2, pyt))
    nboundaries = pxbd.size

    nx = int(np.round(1 / h))
    ny = int(np.round((1 - yb1) / h))

    pxi = np.linspace(h, 1-h, nx)
    pyi = np.linspace(yb1 + h, 1-h, ny)
    pxi, pyi = np.meshgrid(pxi, pyi)
    pxi = pxi.ravel()
    pyi = pyi.ravel()

    inside = pyi > yb_func(pxi) + h
    pxi = pxi[inside]
    pyi = pyi[inside]

    px = np.hstack((pxbd, pxi))
    py = np.hstack((pybd, pyi))
    npoints = px.size

    return px, py, nleft, nright, nbottom, ntop, nboundaries, npoints

def generate_points_cavity(lengthx, lengthy, h):
    nx = np.floor(lengthx / h + 1).astype(np.int64)
    ny = np.floor(lengthy / h + 1).astype(np.int64)
    # boundary condition f = 1 + x
    # left boundary
    pxleft = np.zeros(ny-2)
    pyleft = np.linspace(0+h,lengthy-h,ny-2)
    nleft = ny-2
    # right boundary
    pxright = np.ones(ny-2) * lengthx
    pyright = np.linspace(0+h,lengthy-h,ny-2)
    nright = nleft + ny - 2
    # top boundary
    ntl = nright
    pxtop = np.linspace(0,lengthx,nx)
    pytop = np.ones(nx) * lengthy
    ntop = nright + nx
    ntr = ntop-1
    # bottom boundary
    pxbottom = np.linspace(0,lengthx,nx)
    pybottom = np.zeros(nx)
    nbottom = ntop + nx
    nbl = ntop
    nbr = nbottom - 1
    # combine into boundary points
    pxboundary = np.hstack((pxleft, pxright, pxtop, pxbottom))
    pyboundary = np.hstack((pyleft, pyright, pytop, pybottom))
    # interior points
    pxinterior = np.linspace(h, lengthx-h, nx-2)
    pyinterior = np.linspace(h, lengthy-h, ny-2)
    pxinterior, pyinterior = np.meshgrid(pxinterior, pyinterior)
    pxinterior = pxinterior.flatten()
    pyinterior = pyinterior.flatten()
    # combine boundary and interior points
    px = np.hstack((pxboundary, pxinterior))
    py = np.hstack((pyboundary, pyinterior))
    nboundaries = np.shape(pxboundary)[0]
    npoints = np.shape(px)[0]
    return px, py, nleft, nright, ntop, nbottom, \
        ntl, ntr, nbl, nbr, nboundaries, npoints

def generate_points_random(lengthx, lengthy, h):
    idx = []
    first_orthogonal_point = []
    second_orthogonal_point = []
    nboundaries = 0

    nx = np.floor(lengthx / h + 1).astype(np.int64)
    ny = np.floor(lengthy / h + 1).astype(np.int64)
    
    px = np.array([])
    py = np.array([])

    # left boundary
    pxleft = np.zeros(ny - 6)
    pyleft = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nleft = np.shape(pxleft)[0]
    idx.append(np.arange(0, nleft))
    nboundaries += nleft
    
    # right boundary  (FIX: use lengthx, not lengthy)
    pxright = np.ones(ny - 6) * lengthx
    pyright = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nright = np.shape(pxright)[0]
    idx.append(np.arange(nboundaries, nboundaries + nright))
    nboundaries += nright

    # top boundary
    pxtop = np.linspace(0, lengthx, nx)
    pytop = np.ones(nx) * lengthy
    ntop = np.shape(pxtop)[0]
    idx.append(np.arange(nboundaries, nboundaries + ntop))
    nboundaries += ntop

    # bottom boundary
    pxbottom = np.linspace(0, lengthx, nx)
    pybottom = np.zeros(nx)
    nbottom = np.shape(pxbottom)[0]
    idx.append(np.arange(nboundaries, nboundaries + nbottom))
    nboundaries += nbottom

    # combine into boundary points
    pxboundary = np.hstack((pxleft, pxright, pxtop, pxbottom))
    pyboundary = np.hstack((pyleft, pyright, pytop, pybottom))
    nboundaries = np.shape(pxboundary)[0]
    idx.append(np.arange(0, nboundaries))
    
    px = np.concatenate((px, pxboundary))
    py = np.concatenate((py, pyboundary))
    
    nfirst = nboundaries
    
    # Generate first orthogonal boundary points
    # left boundary
    pxleft = np.ones(ny - 6) * h
    pyleft = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nleft = np.shape(pxleft)[0]
    first_orthogonal_point.append(np.arange(nfirst, nfirst + nleft))
    nfirst += nleft
    
    # right boundary  (FIX: use lengthx, not lengthy)
    pxright = np.ones(ny - 6) * (lengthx-h)
    pyright = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nright = np.shape(pxright)[0]
    first_orthogonal_point.append(np.arange(nfirst, nfirst + nright))
    nfirst += nright

    # top boundary
    pxtop = np.linspace(0, lengthx, nx)
    pytop = np.ones(nx) * (lengthy-h)
    ntop = np.shape(pxtop)[0]
    first_orthogonal_point.append(np.arange(nfirst, nfirst + ntop))
    nfirst += ntop

    # bottom boundary
    pxbottom = np.linspace(0, lengthx, nx)
    pybottom = np.ones(nx) * h
    nbottom = np.shape(pxbottom)[0]
    first_orthogonal_point.append(np.arange(nfirst, nfirst + nbottom))
    nfirst += nbottom
    
    pxfirst = np.hstack((pxleft, pxright, pxtop, pxbottom))
    pyfirst = np.hstack((pyleft, pyright, pytop, pybottom))
    
    px = np.concatenate((px, pxfirst))
    py = np.concatenate((py, pyfirst))
    
    nsecond = nfirst
    
    # Generate second orthogonal boundary points
    # left boundary
    pxleft = np.ones(ny - 6) * (2*h)
    pyleft = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nleft = np.shape(pxleft)[0]
    second_orthogonal_point.append(np.arange(nsecond, nsecond + nleft))
    nsecond += nleft
    
    # right boundary  (FIX: use lengthx, not lengthy)
    pxright = np.ones(ny - 6) * (lengthx-2*h)
    pyright = np.linspace(3 * h, lengthy - 3 * h, ny - 6)
    nright = np.shape(pxright)[0]
    second_orthogonal_point.append(np.arange(nsecond, nsecond + nright))
    nsecond += nright

    # top boundary
    pxtop = np.linspace(0, lengthx, nx)
    pytop = np.ones(nx) * (lengthy-2*h)
    ntop = np.shape(pxtop)[0]
    second_orthogonal_point.append(np.arange(nsecond, nsecond + ntop))
    nsecond += ntop

    # bottom boundary
    pxbottom = np.linspace(0, lengthx, nx)
    pybottom = np.ones(nx) * (2*h)
    nbottom = np.shape(pxbottom)[0]
    second_orthogonal_point.append(np.arange(nsecond, nsecond + nbottom))
    nsecond += nbottom
    
    pxsecond = np.hstack((pxleft, pxright, pxtop, pxbottom))
    pysecond = np.hstack((pyleft, pyright, pytop, pybottom))
    
    px = np.concatenate((px, pxsecond))
    py = np.concatenate((py, pysecond))
    
    npoints = np.floor((nx * ny - nsecond) * 0.85).astype(np.int64)
    
    pxrandom = np.random.uniform(3*h, lengthx-3*h, npoints)
    pyrandom = np.random.uniform(3*h, lengthy-3*h, npoints)
    
    # --- REPLACE your current random generation block with this ---

    # target random count (keep your original formula)
    npoints_target = np.floor((nx * ny - nsecond) * 0.85).astype(np.int64)
    
    # fill right up to the 2nd orthogonal layer (no artificial gap)
    x0, x1 = 2*h, lengthx - 2*h
    y0, y1 = 2*h, lengthy - 2*h
    
    # minimum distance control
    r = 0.9 * h
    cell = r / np.sqrt(2.0)
    
    # build grid over interior box
    gw = int(np.ceil((x1 - x0) / cell))
    gh = int(np.ceil((y1 - y0) / cell))
    grid = -np.ones((gh, gw), dtype=np.int64)
    
    # existing points we must avoid (boundary + orth layers already in px,py)
    ex = px.copy()
    ey = py.copy()
    
    def grid_xy(x, y):
        return int((x - x0) / cell), int((y - y0) / cell)
    
    def in_box(x, y):
        return (x0 <= x <= x1) and (y0 <= y <= y1)
    
    # pre-fill grid with existing points that are near/inside the random box
    for i in range(ex.size):
        if (x0 - r <= ex[i] <= x1 + r) and (y0 - r <= ey[i] <= y1 + r):
            gx, gy = grid_xy(ex[i], ey[i])
            if 0 <= gx < gw and 0 <= gy < gh:
                grid[gy, gx] = i  # safe: your structured points already satisfy spacing ~h
    
    px_new = []
    py_new = []
    
    # dart throwing
    max_trials = int(200 * max(1, npoints_target))  # cap runtime
    trials = 0
    while (len(px_new) < npoints_target) and (trials < max_trials):
        trials += 1
        cx = np.random.uniform(x0, x1)
        cy = np.random.uniform(y0, y1)
    
        gx, gy = grid_xy(cx, cy)
        if not (0 <= gx < gw and 0 <= gy < gh):
            continue
    
        ok = True
        # check nearby cells only
        for yy in range(max(0, gy - 2), min(gh, gy + 3)):
            for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                j = grid[yy, xx]
                if j != -1:
                    # j may refer to existing points (0..ex.size-1) OR new points (stored later)
                    if j < ex.size:
                        dxj = cx - ex[j]
                        dyj = cy - ey[j]
                    else:
                        jj = j - ex.size
                        dxj = cx - px_new[jj]
                        dyj = cy - py_new[jj]
                    if dxj*dxj + dyj*dyj < r*r:
                        ok = False
                        break
            if not ok:
                break
    
        if ok:
            # accept
            j_store = ex.size + len(px_new)
            grid[gy, gx] = j_store
            px_new.append(cx)
            py_new.append(cy)
    
    pxrandom = np.array(px_new, dtype=float)
    pyrandom = np.array(py_new, dtype=float)
    npoints = pxrandom.size  # (keeps your return meaning: number of random points)
    
    px = np.concatenate((px, pxrandom))
    py = np.concatenate((py, pyrandom))
    
    return px, py, idx, first_orthogonal_point, second_orthogonal_point, \
            nboundaries, npoints

