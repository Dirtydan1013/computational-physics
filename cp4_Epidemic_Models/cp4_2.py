import numpy as np

def label_clusters(grid, periodic=False):
    """
    grid: 2D boolean array, True=occupied
    periodic: 是否週期邊界
    回傳 labels: same shape, 0=空，1,2,...=叢集編號
    """
    h, w = grid.shape
    labels = np.zeros_like(grid, dtype=int)
    current = 0

    for i in range(h):
        for j in range(w):
            if grid[i,j] and labels[i,j] == 0:
                current += 1
                # flood-fill / DFS
                stack = [(i,j)]
                labels[i,j] = current
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x + dx, y + dy
                        if periodic:
                            nx %= h
                            ny %= w
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx,ny] and labels[nx,ny] == 0:
                                labels[nx,ny] = current
                                stack.append((nx,ny))
    return labels

if __name__ == "__main__":
    # 
    grid = np.array([
        [ True, True, False, False, True, False],
        [ False, False, False, False, False, True],
        [ True, False, False, False, True, False],
        [ False, False, True, True, False, False],
        [ True, False, True, False, False, False],
    ])  # shape=(5,6)

    # (a) open boundary
    labels_open = label_clusters(grid, periodic=False)
    print("Open boundary labels:\n", labels_open)

    # (b) periodic boundary
    labels_per = label_clusters(grid, periodic=True)
    print("Periodic boundary labels:\n", labels_per)
