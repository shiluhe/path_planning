"""
UAV Path Planning - Python Implementation
Two numerical methods: Penalty + Gradient Descent / SQP (scipy.optimize)
Six meta_init_algorithms for initialization: Dijkstra, A*, RRT, GA, ACO, SA, (Default: Straight Line)
"""

 # 设置非交互式后端，解决 GUI xcb 报错
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import heapq
import argparse
from scipy.interpolate import interp1d
from meta_algorithm.dijkstra import get_dijkstra_path, resample_path
from meta_algorithm.a_star import get_a_star_path
from meta_algorithm.rrt import get_rrt_path
from meta_algorithm.genetic import get_genetic_path
from meta_algorithm.ant_colony import get_aco_path
from meta_algorithm.simulated_annealing import get_sa_path
from init_numerical_solution import obj_length, inequality_constraint, equality_constraint, plot_result

# ============ Global Parameters ============
space_limit = 30
start_point = np.array([0, 0])
end_point = np.array([30, 30])

obstacles = np.array([
    [5,  6,  3],
    [12, 7,  2],
    [18, 20, 4],
    [24, 25, 3],
    [10, 22, 3],
    [20, 5,  4],
    [16, 15, 3],
    [28, 10, 2],
    [21, 14, 3],
    [10, 2, 2]
])

# ============ Method 1: Penalty + Gradient Descent ============
def run_penalty_gradient(n_waypoints=20, init_method='straight'):
    print("\n" + "="*60)
    print(f"Method 1: Penalty Function + Gradient Descent (n_waypoints={n_waypoints}, init_method={init_method})")
    print("="*60)

    import time
    t0 = time.time()

    # Entrance mata_algorithm
    if init_method == 'dijkstra':
        raw_path = get_dijkstra_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'a_star':
        raw_path = get_a_star_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'rrt':
        raw_path = get_rrt_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'genetic':
        # Genetic Algorithm return value is already a resampled set of points based on its configuration,
        # but to ensure consistency with n_waypoints, we can still resample it.
        raw_path = get_genetic_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'aco':
        raw_path = get_aco_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'sa':
        raw_path = get_sa_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    else:
        # Fallback to straight line
        t = np.linspace(0, 1, n_waypoints + 2)
        initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point

    X_initial = initial_path[1:-1]

    t1 = time.time()
    init_time = t1 - t0
    
    diffs_init = np.diff(initial_path, axis=0)
    distances_init = np.sqrt(np.sum(diffs_init**2, axis=1))
    initial_L = np.sum(distances_init)
    
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Initial path length: {initial_L:.2f} m")

    opt_start = time.time()

    # Increase penalty weights (sigma) and reduce learning rate to fix bug
    sigma = 100000
    max_iter = 4000
    learning_rate = 0.002
    tolerance = 1e-7

    X = X_initial.copy()

    print(f"Starting iteration... (max {max_iter} iterations)")

    for iter_num in range(1, max_iter + 1):
        grad = np.zeros((n_waypoints, 2))
        current_path = np.vstack([start_point, X, end_point])

        for i in range(n_waypoints):
            idx = i + 1
            Pi = current_path[idx]
            Prev = current_path[idx - 1]
            Next = current_path[idx + 1]

            # Length gradient (with eps to avoid division by zero)
            eps = 1e-10
            dist_prev = np.linalg.norm(Pi - Prev) + eps
            dist_next = np.linalg.norm(Pi - Next) + eps
            grad_length = (Pi - Prev) / dist_prev + (Pi - Next) / dist_next

            # Penalty gradient (pointing outward from obstacle)
            grad_penalty = np.zeros(2)
            for j in range(obstacles.shape[0]):
                obs_center = obstacles[j, :2]
                R = obstacles[j, 2]
                dist_obs = np.linalg.norm(Pi - obs_center) + eps

                # refuse drama big change of gradient to fix bugs
                # influence_radius = R * 1.05
                # if dist_obs < influence_radius:
                #     grad_penalty += 2 * sigma * (influence_radius - dist_obs) * (Pi - obs_center) / dist_obs

            grad[i] = grad_length + grad_penalty

        # Gradient descent update
        X_new = X - learning_rate * grad
        X_new = np.clip(X_new, 0, space_limit)

        # Project points inside obstacles back to boundary
        for i in range(n_waypoints):
            for j in range(obstacles.shape[0]):
                obs_center = obstacles[j, :2]
                R = obstacles[j, 2]
                dist = np.linalg.norm(X_new[i] - obs_center)
                if dist < R * 0.99:  # 0.99 buffer to ensure outside
                    X_new[i] = obs_center + (X_new[i] - obs_center) / dist * R * 1.01

        if np.linalg.norm(X_new - X) < tolerance:
            print(f"Converged at iteration {iter_num}.")
            X = X_new
            break

        X = X_new

        if iter_num % 1000 == 0:
            print(f"  Iteration {iter_num}/{max_iter}...")

    final_path = np.vstack([start_point, X, end_point])
    
    opt_end = time.time()
    opt_time = opt_end - opt_start

    # Calculate path length
    diffs = np.diff(final_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    total_L = np.sum(distances)

    print(f"Optimization time: {opt_time:.4f} seconds")
    print(f"Optimized path length: {total_L:.2f} m")

    plot_result(initial_path, final_path,
                f"Penalty + Gradient Descent (n={n_waypoints})",
                f"{init_method}_penalty_n{n_waypoints}.png",
                obstacles_arr=obstacles, start_p=start_point, end_p=end_point, space_lim=space_limit)

    return final_path, total_L


# ============ Method 2: SQP ============
def run_sqp(n_waypoints=200, init_method='straight'):
    print("\n" + "="*60)
    print(f"Method 2: SQP (Sequential Quadratic Programming) (n_waypoints={n_waypoints}, init_method={init_method})")
    print("="*60)

    import time
    t0 = time.time()

    # Entrance mata_algorithm
    if init_method == 'dijkstra':
        raw_path = get_dijkstra_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'a_star':
        raw_path = get_a_star_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'rrt':
        raw_path = get_rrt_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'genetic':
        raw_path = get_genetic_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'aco':
        raw_path = get_aco_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    elif init_method == 'sa':
        raw_path = get_sa_path(start_point, end_point, space_limit, obstacles)
        if raw_path is not None:
            initial_path = resample_path(raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    else:
        t = np.linspace(0, 1, n_waypoints + 2)
        initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point

    X_initial = initial_path[1:-1]

    t1 = time.time()
    init_time = t1 - t0
    
    diffs_init = np.diff(initial_path, axis=0)
    distances_init = np.sqrt(np.sum(diffs_init**2, axis=1))
    initial_L = np.sum(distances_init)
    
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Initial path length: {initial_L:.2f} m")

    opt_start = time.time()

    x0 = X_initial.flatten()
    bounds = [(0, space_limit) for _ in range(len(x0))]

    print(f"Starting SQP optimization... (n_waypoints={n_waypoints})")

    result = minimize(
        lambda x: obj_length(x, start_point, end_point),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: inequality_constraint(x, obstacles)},
            {'type': 'eq', 'fun': lambda x: equality_constraint(x, start_point, end_point)}
        ],
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )

    X_opt = result.x.reshape(-1, 2)
    final_path = np.vstack([start_point, X_opt, end_point])
    
    opt_end = time.time()
    opt_time = opt_end - opt_start

    diffs = np.diff(final_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    total_L = np.sum(distances)

    print(f"Optimization time: {opt_time:.4f} seconds")
    print(f"Optimized path length: {total_L:.2f} m")

    plot_result(initial_path, final_path,
                f"SQP Optimization (n={n_waypoints})",
                f"{init_method}_sqp_n{n_waypoints}.png",
                obstacles_arr=obstacles, start_p=start_point, end_p=end_point, space_lim=space_limit)

    return final_path, total_L


# ============ Main Program ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Path Planning Solver")
    parser.add_argument("--init_method", type=str, choices=["dijkstra", "a_star", "rrt", "genetic", "aco", "sa", "straight"], default="straight", help="Initialization method")
    parser.add_argument("--opt_method", type=str, choices=["penalty", "sqp"], default="sqp", help="Optimization method")
    parser.add_argument("--waypoints", type=int, default=200, help="Number of waypoints")
    args = parser.parse_args()

    # print("="*60)
    # print("UAV Path Planning Solver")
    # print("="*60)
    # print(f"Space range: {space_limit}x{space_limit}")
    # print(f"Start point: {start_point}")
    # print(f"End point: {end_point}")
    # print(f"Number of obstacles: {obstacles.shape[0]}")

    if args.opt_method in ["penalty"]:
        run_penalty_gradient(n_waypoints=args.waypoints, init_method=args.init_method)

    if args.opt_method in ["sqp"]:
        run_sqp(n_waypoints=args.waypoints, init_method=args.init_method)

    print("\n" + "="*60)
    print("All optimizations complete! Results saved to results/ folder")
    print("="*60)