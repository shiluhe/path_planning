"""
UAV Path Planning - Python Implementation
Two optimization methods: Penalty + Gradient Descent / SQP (scipy.optimize)
"""

import matplotlib
matplotlib.use('Agg') # 设置非交互式后端，解决 GUI xcb 报错

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

os.makedirs("results", exist_ok=True)

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

# ============ Objective Function ============
def obj_length(x, start_p, end_p):
    """Objective: minimize total path length"""
    pts = x.reshape(-1, 2)
    full_path = np.vstack([start_p, pts, end_p])
    diffs = np.diff(full_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

# ============ Constraint Function ============
def inequality_constraint(x, obstacles_arr):
    """Inequality constraint: point must be outside all obstacles (c <= 0)"""
    pts = x.reshape(-1, 2)
    num_pts = pts.shape[0]
    num_obs = obstacles_arr.shape[0]
    c_list = []
    for i in range(num_pts):
        for j in range(num_obs):
            dist_sq = (pts[i, 0] - obstacles_arr[j, 0])**2 + (pts[i, 1] - obstacles_arr[j, 1])**2
            R_sq = obstacles_arr[j, 2]**2
            # c_list.append(R_sq - dist_sq)
            c_list.append(dist_sq - R_sq)
    return np.array(c_list)

# ============ Plotting Function ============
def plot_result(initial_path, final_path, title, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, space_limit)
    ax.set_ylim(0, space_limit)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)

    # Draw obstacles
    for obs in obstacles:
        xc, yc, r = obs
        circle = plt.Circle((xc, yc), r, color='red', alpha=0.5, label='_nolegend_')
        ax.add_patch(circle)
        ax.plot(xc, yc, 'r+', markersize=4)

    # Draw start and end points
    ax.plot(start_point[0], start_point[1], 'gs', markersize=10, markerfacecolor='g')
    ax.annotate('Start A', (start_point[0] + 0.5, start_point[1]), fontweight='bold')
    ax.plot(end_point[0], end_point[1], 'bs', markersize=10, markerfacecolor='b')
    ax.annotate('End B', (end_point[0] - 3, end_point[1] + 1), fontweight='bold')

    # Draw initial path
    ax.plot(initial_path[:, 0], initial_path[:, 1], 'k--o',
            linewidth=1, markersize=4, label='Initial Path')

    # Draw optimized path
    ax.plot(final_path[:, 0], final_path[:, 1], 'b-p',
            linewidth=2, markersize=8, markerfacecolor='y', label='Optimized Path')

    ax.legend()
    fig.savefig(f"results/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: results/{filename}")

# ============ Method 1: Penalty + Gradient Descent ============
def run_penalty_gradient(n_waypoints=10):
    """Method 1: Penalty function + Gradient descent with reduced waypoints"""
    print("\n" + "="*60)
    print(f"Method 1: Penalty Function + Gradient Descent (n_waypoints={n_waypoints})")
    print("="*60)

    # Generate initial path
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
                f"penalty_gradient_n{n_waypoints}.png")

    np.save(f"results/penalty_gradient_path_n{n_waypoints}.npy", final_path)
    print(f"Path data saved: results/penalty_gradient_path_n{n_waypoints}.npy")

    # Save detailed results
    with open(f"results/penalty_gradient_results_n{n_waypoints}.txt", "w") as f:
        f.write(f"Penalty + Gradient Descent Method Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of waypoints: {n_waypoints}\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Penalty factor sigma: {sigma}\n")
        f.write(f"Tolerance: {tolerance}\n\n")
        f.write(f"Final path length: {total_L:.2f} m\n")
        f.write(f"Converged at iteration: {iter_num}\n\n")
        f.write("Final path coordinates:\n")
        for i, pt in enumerate(final_path):
            f.write(f"  Point {i}: ({pt[0]:.4f}, {pt[1]:.4f})\n")

    print(f"Results saved: results/penalty_gradient_results_n{n_waypoints}.txt")

    return final_path, total_L

# ============ Method 2: SQP ============
def run_sqp(n_waypoints=200):
    """Method 2: SQP using scipy.optimize.minimize"""
    print("\n" + "="*60)
    print(f"Method 2: SQP (Sequential Quadratic Programming) (n_waypoints={n_waypoints})")
    print("="*60)

    # Generate initial path
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
                f"sqp_path_n{n_waypoints}.png")

    np.save(f"results/sqp_path_n{n_waypoints}.npy", final_path)
    print(f"Path data saved: results/sqp_path_n{n_waypoints}.npy")

    # Save detailed results
    with open(f"results/sqp_results_n{n_waypoints}.txt", "w") as f:
        f.write(f"SQP Method Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of waypoints: {n_waypoints}\n")
        f.write(f"Max iterations: 1000\n")
        f.write(f"Tolerance: 1e-6\n\n")
        f.write(f"Final path length: {total_L:.2f} m\n")
        f.write(f"Optimization exit flag: {result.status}\n")
        f.write(f"Optimization message: {result.message}\n")
        f.write(f"Function evaluations: {result.nfev}\n")
        f.write(f"Gradient evaluations: {result.njev}\n\n")
        f.write("Final path coordinates:\n")
        for i, pt in enumerate(final_path):
            f.write(f"  Point {i}: ({pt[0]:.4f}, {pt[1]:.4f})\n")

    print(f"Results saved: results/sqp_results_n{n_waypoints}.txt")

    return final_path, total_L