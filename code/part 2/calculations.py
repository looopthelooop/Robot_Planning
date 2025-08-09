def heuristic(a, b):
    # a, b are (x, y, θ), ignore θ for heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def check_neighbors(node, grid):
    x, y, theta = node
    neighbors = []
    T, Y, X = grid.shape
    for dx, dy, dt in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        nx, ny, nt = x+dx, y+dy, (theta+dt)%T
        if 0 <= nx < X and 0 <= ny < Y:
            if grid[nt, ny, nx] == 0:
                neighbors.append((nx, ny, nt))
    return neighbors
