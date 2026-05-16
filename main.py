"""
UAV Path Planning - Python Implementation
Two numerical methods: Penalty + Gradient Descent / SQP (scipy.optimize)
Six meta_init_algorithms for initialization: Dijkstra, A*, RRT, GA, ACO, SA, (Default: Straight Line)
"""

import matplotlib
matplotlib.use('Agg') # 设置非交互式后端，解决 GUI xcb 报错

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import heapq
import argparse
from scipy.interpolate import interp1d
from meta_algorithm.dijkstra import get_dijkstra_path, resample_path
from init_numerical_solution import obj_length, inequality_constraint, plot_result

# ============ Global Parameters ============
space_limit = 30
start_point = np.array([0, 0])
end_point = np.array([30, 30])

obstacles = np.array([
    [5,  6,  3],
    [12, 8,  2],
    [18, 20, 4],
    [24, 25, 3],
    [10, 22, 3],
    [20, 5,  4],
    [16, 15, 3],
    [28, 10, 2]
])

# ============ Method 1: Penalty + Gradient Descent ============
def run_penalty_gradient(n_waypoints=20, init_method='straight'):
    print("\n" + "="*60)
    print(f"Method 1: Penalty Function + Gradient Descent (n_waypoints={n_waypoints}, init_method={init_method})")
    print("="*60)

    # Generate initial path
    if init_method == 'dijkstra':
        dijkstra_raw_path = get_dijkstra_path(start_point, end_point, space_limit, obstacles)
        if dijkstra_raw_path is not None:
            initial_path = resample_path(dijkstra_raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    else:
        # Fallback to straight line
        t = np.linspace(0, 1, n_waypoints + 2)
        initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point

    X_initial = initial_path[1:-1]

    sigma = 50000
    max_iter = 4000
    learning_rate = 0.005
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

                if dist_obs < R:
                    grad_penalty += 2 * sigma * (R - dist_obs) * (Pi - obs_center) / dist_obs

            grad[i] = grad_length + grad_penalty

        # Gradient descent update
        X_new = X - learning_rate * grad

        # Clip to space boundaries [0, space_limit]
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

    # Calculate path length
    diffs = np.diff(final_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    total_L = np.sum(distances)

    print(f"Optimized path length: {total_L:.2f} m")
    print(f"Total waypoints: {len(final_path)}")

    plot_result(initial_path, final_path,
                f"Penalty + Gradient Descent (n={n_waypoints})",
                f"{init_method}_penalty_n{n_waypoints}.png")

    return final_path, total_L

# ============ Method 2: SQP ============
def run_sqp(n_waypoints=200, init_method='straight'):
    """Method 2: SQP using scipy.optimize.minimize"""
    print("\n" + "="*60)
    print(f"Method 2: SQP (Sequential Quadratic Programming) (n_waypoints={n_waypoints}, init_method={init_method})")
    print("="*60)

    # Generate initial path
    if init_method == 'dijkstra':
        dijkstra_raw_path = get_dijkstra_path(start_point, end_point, space_limit, obstacles)
        if dijkstra_raw_path is not None:
            initial_path = resample_path(dijkstra_raw_path, n_waypoints)
        else:
            t = np.linspace(0, 1, n_waypoints + 2)
            initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point
    else:
        # Fallback to straight line
        t = np.linspace(0, 1, n_waypoints + 2)
        initial_path = (1 - t.reshape(-1, 1)) * start_point + t.reshape(-1, 1) * end_point

    X_initial = initial_path[1:-1]

    x0 = X_initial.flatten()
    bounds = [(0, space_limit) for _ in range(len(x0))]

    print(f"Starting SQP optimization... (n_waypoints={n_waypoints})")

    result = minimize(
        lambda x: obj_length(x, start_point, end_point),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda x: inequality_constraint(x, obstacles)},
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )

    X_opt = result.x.reshape(-1, 2)
    final_path = np.vstack([start_point, X_opt, end_point])

    diffs = np.diff(final_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    total_L = np.sum(distances)

    print(f"Optimized path length: {total_L:.2f} m")
    print(f"Total waypoints: {len(final_path)}")
    print(f"Optimization status: {result.message}")

    plot_result(initial_path, final_path,
                f"SQP Optimization (n={n_waypoints})",
                f"{init_method}_sqp_n{n_waypoints}.png")

    return final_path, total_L

# ============ Main Program ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Path Planning Solver")
    parser.add_argument("--init_method", type=str, choices=["dijkstra", "straight"], default="straight", help="Initialization method")
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