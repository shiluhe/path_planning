import numpy as np
import random

def is_collision_free(pt, obstacles_arr, margin=0.5):
    for obs in obstacles_arr:
        dist = np.sqrt((pt[0] - obs[0])**2 + (pt[1] - obs[1])**2)
        if dist <= obs[2] + margin:
            return False
    return True

def get_aco_path(start, end, space_limit, obstacles_arr, resolution=1.0, 
                 n_ants=30, max_iter=50, alpha=1.0, beta=4.0, evaporation_rate=0.2):
    grid_size = int(space_limit / resolution) + 1
    
    def to_grid(x, y):
        return int(round(x / resolution)), int(round(y / resolution))
        
    start_idx = to_grid(start[0], start[1])
    end_idx = to_grid(end[0], end[1])
    
    # 节点上的信息素浓度矩阵
    pheromones = np.ones((grid_size, grid_size)) * 0.1
    
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    best_path = None
    best_cost = float('inf')
    
    for _ in range(max_iter):
        all_paths = []
        all_costs = []
        
        for ant in range(n_ants):
            curr = start_idx
            path = [curr]
            visited = set([curr])
            
            reached_end = False
            # 限制最大步数，防死胡同无限循环
            for step in range(grid_size * 4):
                if curr == end_idx:
                    reached_end = True
                    break
                    
                probs = []
                valid_neighbors = []
                
                for dx, dy in directions:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) not in visited:
                            n_pt = np.array([nx * resolution, ny * resolution])
                            if is_collision_free(n_pt, obstacles_arr):
                                valid_neighbors.append((nx, ny))
                                tau = pheromones[nx, ny]
                                # 启发式因子：到终点距离的倒数
                                dist_to_end = np.sqrt((nx - end_idx[0])**2 + (ny - end_idx[1])**2)
                                eta = 1.0 / (dist_to_end + 1e-6)
                                probs.append((tau ** alpha) * (eta ** beta))
                                
                if not valid_neighbors:
                    break # 走到死角
                    
                probs = np.array(probs)
                if np.sum(probs) == 0:
                    probs = np.ones(len(probs)) / len(probs)
                else:
                    probs /= np.sum(probs)
                    
                # 轮盘赌选择下一个节点
                idx = np.random.choice(len(valid_neighbors), p=probs)
                curr = valid_neighbors[idx]
                path.append(curr)
                visited.add(curr)
                
            if reached_end:
                # 计算路径代价
                cost = sum(np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2) for i in range(1, len(path)))
                all_paths.append(path)
                all_costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
        
        # 信息素挥发与更新
        pheromones *= (1 - evaporation_rate)
        for path, cost in zip(all_paths, all_costs):
            delta_tau = 10.0 / cost
            for node in path:
                pheromones[node[0], node[1]] += delta_tau
                
    if best_path is None:
        print("未找到有效路径！")
        return None
        
    real_path = [[pt[0] * resolution, pt[1] * resolution] for pt in best_path]
    return np.array(real_path)
