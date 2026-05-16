import numpy as np
import heapq

def is_collision_free(pt, obstacles_arr, margin=0.5):
    for obs in obstacles_arr:
        dist = np.sqrt((pt[0] - obs[0])**2 + (pt[1] - obs[1])**2)
        if dist <= obs[2] + margin:
            return False
    return True

def get_dijkstra_path(start, end, space_limit, obstacles_arr, resolution=1.0):
    grid_size = int(space_limit / resolution) + 1
    
    # 将坐标转换为网格索引
    def to_grid(x, y):
        return int(round(x / resolution)), int(round(y / resolution))
    
    start_idx = to_grid(start[0], start[1])
    end_idx = to_grid(end[0], end[1])
    
    # 堆栈：(cost, x_idx, y_idx)
    queue = [(0, start_idx[0], start_idx[1])]
    
    # 记录代价和父节点
    costs = {start_idx: 0}
    parents = {start_idx: None}
    
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    while queue:
        current_cost, cx, cy = heapq.heappop(queue)
        
        if (cx, cy) == end_idx:
            break
            
        for dx, dy in directions:
            nx = cx + dx
            ny = cy + dy
            n_idx = (nx, ny)
            
            # 边界检查
            if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
                continue
                
            # 碰撞检查
            n_pt = np.array([nx * resolution, ny * resolution])
            if not is_collision_free(n_pt, obstacles_arr):
                continue
                
            move_cost = np.sqrt(dx**2 + dy**2) * resolution
            new_cost = current_cost + move_cost
            
            if n_idx not in costs or new_cost < costs[n_idx]:
                costs[n_idx] = new_cost
                parents[n_idx] = (cx, cy)
                heapq.heappush(queue, (new_cost, nx, ny))
                
    # 回溯路径
    if end_idx not in parents:
        print("未找到有效路径！")
        return None
        
    path = []
    curr = end_idx
    while curr is not None:
        path.append([curr[0] * resolution, curr[1] * resolution])
        curr = parents[curr]
        
    path.reverse()
    return np.array(path)

def resample_path(path_pts, n_waypoints):
    # 计算累积距离作为插值基准
    diffs = np.diff(path_pts, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    
    total_dist = cum_dists[-1]
    target_dists = np.linspace(0, total_dist, n_waypoints + 2)
    
    new_path = np.zeros((n_waypoints + 2, 2))
    new_path[:, 0] = np.interp(target_dists, cum_dists, path_pts[:, 0])
    new_path[:, 1] = np.interp(target_dists, cum_dists, path_pts[:, 1])
    return new_path
