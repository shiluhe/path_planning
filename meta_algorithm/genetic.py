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

def get_genetic_path(start, end, space_limit, obstacles_arr, n_nodes=15, pop_size=100, generations=200):
    start_pt = np.array(start)
    end_pt = np.array(end)
    
    # 1. 初始化种群 (在直线基础上添加高斯噪声)
    population = []
    t = np.linspace(0, 1, n_nodes + 2)
    straight_line = (1 - t.reshape(-1, 1)) * start_pt + t.reshape(-1, 1) * end_pt
    
    for _ in range(pop_size):
        noise = np.random.normal(0, space_limit * 0.1, size=(n_nodes, 2))
        pts = straight_line[1:-1] + noise
        pts = np.clip(pts, 0, space_limit)
        population.append(pts)
        
    best_path = None
    best_cost = float('inf')
    
    for gen in range(generations):
        # 2. 适应度评估
        costs = []
        for pts in population:
            full_path = np.vstack([start_pt, pts, end_pt])
            cost = calculate_path_cost(full_path, obstacles_arr)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_path = full_path
                
        costs = np.array(costs)
        # 转换为适应度，代价越小适应度越大
        fitness = 1.0 / (costs + 1e-6)
        prob = fitness / np.sum(fitness)
        
        new_population = []
        
        # 3. 精英保留策略 (保留代价最小的一个)
        best_idx = np.argmin(costs)
        new_population.append(population[best_idx].copy())
        
        # 4. 选择、交叉、变异
        while len(new_population) < pop_size:
            # 轮盘赌选择
            p1_idx = np.random.choice(pop_size, p=prob)
            p2_idx = np.random.choice(pop_size, p=prob)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # 单点交叉
            child = np.zeros_like(p1)
            crossover_pt = random.randint(1, n_nodes - 2)
            child[:crossover_pt] = p1[:crossover_pt]
            child[crossover_pt:] = p2[crossover_pt:]
            
            # 变异
            if random.random() < 0.3:  # 30% 变异率
                mut_pt = random.randint(0, n_nodes - 1)
                child[mut_pt] += np.random.normal(0, space_limit * 0.05, size=2)
                child = np.clip(child, 0, space_limit)
                
            new_population.append(child)
            
        population = new_population
        
    return best_path
