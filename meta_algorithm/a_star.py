import numpy as np
import heapq

def is_collision_free(pt, obstacles_arr, margin=0.5):
    for obs in obstacles_arr:
        dist = np.sqrt((pt[0] - obs[0])**2 + (pt[1] - obs[1])**2)
        if dist <= obs[2] + margin:
            return False
    return True

def get_a_star_path(start, end, space_limit, obstacles_arr, resolution=1.0):
    grid_size = int(space_limit / resolution) + 1
    
    def to_grid(x, y):
        return int(round(x / resolution)), int(round(y / resolution))
        
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * resolution
    
    start_idx = to_grid(start[0], start[1])
    end_idx = to_grid(end[0], end[1])
    
    # (f_cost, g_cost, x_idx, y_idx)
    queue = [(heuristic(start_idx, end_idx), 0, start_idx[0], start_idx[1])]
    
    # Record the actual cost g_cost and the parent node.
    costs = {start_idx: 0}
    parents = {start_idx: None}
    
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    while queue:
        _, current_cost, cx, cy = heapq.heappop(queue)
        
        if (cx, cy) == end_idx:
            break
            
        for dx, dy in directions:
            nx = cx + dx
            ny = cy + dy
            n_idx = (nx, ny)
            
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
                continue
                
            n_pt = np.array([nx * resolution, ny * resolution])
            if not is_collision_free(n_pt, obstacles_arr):
                continue
                
            move_cost = np.sqrt(dx**2 + dy**2) * resolution
            new_cost = current_cost + move_cost
            
            if n_idx not in costs or new_cost < costs[n_idx]:
                costs[n_idx] = new_cost
                parents[n_idx] = (cx, cy)
                f_cost = new_cost + heuristic(n_idx, end_idx)
                heapq.heappush(queue, (f_cost, new_cost, nx, ny))
                
    if end_idx not in parents:
        print("No valid path found！")
        return None
        
    path = []
    curr = end_idx
    while curr is not None:
        path.append([curr[0] * resolution, curr[1] * resolution])
        curr = parents[curr]
        
    path.reverse()
    return np.array(path)
