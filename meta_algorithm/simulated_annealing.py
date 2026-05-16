import numpy as np
import random

def calculate_path_cost(path, obstacles_arr, margin=0.5):
    cost = 0.0
    penalty = 0.0
    
    # 碰撞惩罚
    for pt in path:
        for obs in obstacles_arr:
            dist = np.sqrt((pt[0] - obs[0])**2 + (pt[1] - obs[1])**2)
            if dist <= obs[2] + margin:
                penalty += 10000.0  # 极大的惩罚值，驱使远离障碍物
                
    # 距离代价
    diffs = np.diff(path, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cost = np.sum(dists) + penalty
    
    return cost

def get_sa_path(start, end, space_limit, obstacles_arr, n_nodes=20, initial_temp=1000.0, cooling_rate=0.98, max_iter=2000):
    start_pt = np.array(start)
    end_pt = np.array(end)
    
    # 1. 初始解 (直线)
    t = np.linspace(0, 1, n_nodes + 2)
    current_pts = ((1 - t.reshape(-1, 1)) * start_pt + t.reshape(-1, 1) * end_pt)[1:-1]
    
    current_path = np.vstack([start_pt, current_pts, end_pt])
    current_cost = calculate_path_cost(current_path, obstacles_arr)
    
    best_pts = current_pts.copy()
    best_cost = current_cost
    
    temp = initial_temp
    
    # 2. 退火迭代
    for _ in range(max_iter):
        # 产生新解：随机扰动1~3个节点
        neighbor_pts = current_pts.copy()
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, n_nodes - 1)
            # 添加符合当前温度尺度的高斯噪声
            noise_scale = space_limit * 0.05 * (temp / initial_temp) + 0.1
            neighbor_pts[idx] += np.random.normal(0, noise_scale, size=2)
            
        # 边界约束
        neighbor_pts = np.clip(neighbor_pts, 0, space_limit)
        
        neighbor_path = np.vstack([start_pt, neighbor_pts, end_pt])
        neighbor_cost = calculate_path_cost(neighbor_path, obstacles_arr)
        
        # 能量差
        delta_cost = neighbor_cost - current_cost
        
        # 3. Metropolis准则接受新解
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / temp):
            current_pts = neighbor_pts
            current_cost = neighbor_cost
            
            # 更新全局最优解
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts.copy()
                
        # 4. 降温
        temp *= cooling_rate
        if temp < 1e-3:
            break
            
    best_full_path = np.vstack([start_pt, best_pts, end_pt])
    return best_full_path
