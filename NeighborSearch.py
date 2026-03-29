import numpy as np
from scipy.spatial import cKDTree

import numpy as np
from scipy.spatial import cKDTree

def kdtree(px, py, r0, nboundaries, npoints):
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    pts = np.column_stack((px[:npoints], py[:npoints]))
    tree = cKDTree(pts)
    lists = tree.query_ball_point(pts, r=r0)

    max_n = max(len(lst) for lst in lists)
    neighbors = np.empty((npoints, max_n), dtype=np.int64)

    for i, lst in enumerate(lists):
        arr = np.asarray(lst, dtype=np.int64)

        # self di kolom 0
        arr = arr[arr != i]
        arr = np.concatenate(([i], arr))

        k = arr.size
        neighbors[i, :k] = arr
        neighbors[i, k:] = i   # pad pakai self, bukan -1

    return neighbors
def knn(px, py, n, npoints):
    # r0 cuma dipertahankan supaya call lama tidak berubah
    # hasil: 16 support points TOTAL, termasuk dirinya sendiri
    # index 0 = dirinya sendiri

    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    pts = np.column_stack((px[:npoints], py[:npoints]))
    interior = np.arange(0, npoints, dtype=np.int64)

    tree = cKDTree(pts)

    k_query = min(n, npoints)
    _, idx = tree.query(pts[interior], k=k_query)

    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]

    out = []
    for i, row in zip(interior, idx):
        row = row.tolist()

        # paksa dirinya sendiri ada di index 0
        if i in row:
            row.remove(i)
        row = [i] + row

        # ambil 16 total termasuk diri sendiri
        out.append(row[:k_query])

    return out