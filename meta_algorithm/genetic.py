import numpy as np
import random

def _segment_min_dist_sq_to_centers(p1, p2, centers):
    """Minimum squared distance from a segment p1->p2 to each obstacle center."""
    seg = p2 - p1
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return np.sum((centers - p1) ** 2, axis=1)

    t = np.dot(centers - p1, seg) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = p1 + t[:, np.newaxis] * seg
    return np.sum((closest - centers) ** 2, axis=1)


def _count_segment_collisions(path, obstacles_arr, margin=0.5):
    path = np.asarray(path)
    centers = obstacles_arr[:, :2]
    radii_sq = (obstacles_arr[:, 2] + margin) ** 2

    collisions = 0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        min_dist_sq = _segment_min_dist_sq_to_centers(p1, p2, centers)
        if np.any(min_dist_sq <= radii_sq):
            collisions += 1
    return collisions


def _project_points_outside_obstacles(points, obstacles_arr, margin=0.5):
    """Project waypoints outside obstacle (radius + margin)."""
    projected = points.copy()
    for i in range(projected.shape[0]):
        for obs in obstacles_arr:
            center = obs[:2]
            safe_r = obs[2] + margin
            vec = projected[i] - center
            dist = np.linalg.norm(vec)
            if dist < safe_r:
                if dist < 1e-9:
                    # Deterministic fallback direction to avoid NaN
                    vec = np.array([1.0, 0.0])
                    dist = 1.0
                projected[i] = center + vec / dist * (safe_r + 1e-3)
    return projected


def calculate_path_cost(path, obstacles_arr, margin=0.5):
    cost = 0.0
    penalty = 0.0
    # vectorized distance and collision checks
    path = np.array(path)
    obstacles_centers = obstacles_arr[:, :2]
    obstacles_radii = obstacles_arr[:, 2]
    
    # distance cost
    diffs = np.diff(path, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    # collision penalty
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        threshold_sq = (obstacles_radii + margin) ** 2
        min_dist_sq = _segment_min_dist_sq_to_centers(p1, p2, obstacles_centers)

        violations = threshold_sq - min_dist_sq
        violations = violations[violations > 0]
        if len(violations) > 0:
            penalty += 1e8 * len(violations) + np.sum(violations) * 5e5
                
    # smooth cost
    smoothness = 0.0
    if len(path) >= 3:
        diffs2 = np.diff(diffs, axis=0)
        smoothness = np.sum(np.linalg.norm(diffs2, axis=1)) * 30.0
        
    cost = np.sum(dists) * 1.5 + penalty + smoothness
    
    return cost

def get_genetic_path(start, end, space_limit, obstacles_arr, n_nodes=15, pop_size=100, generations=200):
    start_pt = np.array(start)
    end_pt = np.array(end)
    
    # 1. initialize population with random paths
    population = []
    t = np.linspace(0, 1, n_nodes + 2)
    straight_line = (1 - t.reshape(-1, 1)) * start_pt + t.reshape(-1, 1) * end_pt

    # dijkstra path as a better-informed seed for the initial population, improving convergence speed and solution quality
    seed_base = straight_line[1:-1]
    try:
        from meta_algorithm.dijkstra import get_dijkstra_path, resample_path

        dijkstra_path = get_dijkstra_path(start_pt, end_pt, space_limit, obstacles_arr)
        if dijkstra_path is not None:
            seed_base = resample_path(dijkstra_path, n_nodes)[1:-1]
    except Exception:
        pass

    for i in range(pop_size):
        noise_scale = space_limit * (0.03 if i < max(1, pop_size // 4) else 0.08)
        noise = np.random.normal(0, noise_scale, size=(n_nodes, 2))
        pts = seed_base + noise
        pts = _project_points_outside_obstacles(pts, obstacles_arr, margin=0.5)
        pts = np.clip(pts, 0, space_limit)
        population.append(pts)
        
    best_path = None
    best_cost = float('inf')
    
    for gen in range(generations):
        # 2. fitness assess
        costs = []
        for pts in population:
            full_path = np.vstack([start_pt, pts, end_pt])
            cost = calculate_path_cost(full_path, obstacles_arr)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_path = full_path
                
        costs = np.array(costs)

        fitness = 1.0 / (costs + 1e-6)
        prob = fitness / np.sum(fitness)
        
        new_population = []
        
        # 3. retaining the one with the lowest cost
        best_idx = np.argmin(costs)
        new_population.append(population[best_idx].copy())
        
        # 4. selection, crossover, mutation
        while len(new_population) < pop_size:
            # “Roulette selection”
            p1_idx = np.random.choice(pop_size, p=prob)
            p2_idx = np.random.choice(pop_size, p=prob)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Arithmetic Crossover：use random alpha for each gene 
            # to create a child that is a blend of two parents, 
            # promoting diversity and avoiding premature convergence.
            alpha = np.random.rand(n_nodes, 1)
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation: Avoid single-point mutations and use range-smoothing mutations.
            if random.random() < 0.3:  # 30% mutation rate
                mut_start = random.randint(0, n_nodes - 3)
                mut_end = min(n_nodes, mut_start + random.randint(1, 4))
                child[mut_start:mut_end] += np.random.normal(0, space_limit * 0.05, size=(mut_end - mut_start, 2))
            child = _project_points_outside_obstacles(child, obstacles_arr, margin=0.5)
            child = np.clip(child, 0, space_limit)

            new_population.append(child)
            
        population = new_population
        
    # if collision detected, fallback to straight line but warning
    collisions = _count_segment_collisions(best_path, obstacles_arr, margin=0.5)
    if collisions > 0:
        print(f"[GA warning] best path still has {collisions} colliding segments.")

    return best_path
