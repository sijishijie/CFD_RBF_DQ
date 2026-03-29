#%%
from PointGenerator import generate_points
from NeighborSearch import knn
from rbf import RBFDQ
import numpy as np
import matplotlib.pyplot as plt
from Poisson import Poisson
import json
#%%
def compute_L2_norm(u_num, u_exact):
    num = np.sqrt(np.sum(np.abs(u_num-u_exact)**2))
    den = np.sqrt(np.sum(np.abs(u_exact)**2))
    if den == 0:
        return 0
    L2_norm = num / den
    return L2_norm

def variation_test(h, n, c):
    px, py, nboundaries, npoints = generate_points(lx, ly, h)
    neighbors = np.asarray(knn(px, py, n, npoints), dtype=np.int64)
    dx = px[neighbors] - px[:, None]
    dy = py[neighbors] - py[:, None]
    d = np.sqrt(dx**2 + dy**2)
    D = np.max(d, axis=1)
    rbfdq = RBFDQ(px, py, D, neighbors, nboundaries, npoints, n=n, c=c)
    w = rbfdq.compute_weight()

    solver = Poisson(px, py, w, nboundaries, npoints, n, neighbors)
    u_num = solver.solve_linear_equations()
    u_exact = 1 + px + np.sin(np.pi * px) * np.sin(np.pi * py)
    L2_norm = compute_L2_norm(u_num, u_exact)
    print(f"L2 norm error: \n e = {L2_norm} \n h = {h} \n n = {n} \n c = {c}")
    
    return L2_norm

lx = 1
ly = 1

# 1. choose one h
# 2. for a fixed h choose n
# 3. for fixed (h,n) choose c
# 4. compute error for (h,n,c)
# 5. pick combination of (n,c) for h
# 6. repeat for every h
cs = np.linspace(0,5,21)
ns = np.array([10, 16, 22, 28])
hs = np.array([0.05, 0.02, 0.01, 0.0067, 0.005])

n=16
optimum = []
least_error = []
for i, h in enumerate(hs):
    error = 999
    for j, n in enumerate(ns):
        for k, c in enumerate(cs):
            L2_norm = variation_test(h, n, c)
            if L2_norm < error:
                error = L2_norm
                n_c = (n, c)
    optimum.append(n_c)
    least_error.append(error)
#%%
from NumpyArrayEncoder import NumpyArrayEncoder            
data = {"h": hs, "n_c": optimum, "e": least_error}
with open("PoissonTest/results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, cls=NumpyArrayEncoder,indent=2)

# %%
