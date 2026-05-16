"""
UAV Path Planning - Python Implementation
Two optimization methods: Penalty + Gradient Descent / SQP (scipy.optimize)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os


# ============ Objective Function ============
def obj_length(x, start_p, end_p):
    """Objective: minimize total path length"""
    pts = x.reshape(-1, 2)
    full_path = np.vstack([start_p, pts, end_p])
    diffs = np.diff(full_path, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

# ============ Constrain include waypoints distances equal and outside obstacles ============
def equality_constraint(x, start_p, end_p):
    # waypoints have equal distances
    pts = x.reshape(-1, 2)
    full_path = np.vstack([start_p, pts, end_p])
    diffs = np.diff(full_path, axis=0)
    dists_sq = np.sum(diffs**2, axis=1)
    
    return np.diff(dists_sq)

def inequality_constraint(x, obstacles_arr):
    pts = x.reshape(-1, 2)
    num_pts = pts.shape[0]
    num_obs = obstacles_arr.shape[0]
    c_list = []

    # 1. outside all obstacles
    for i in range(num_pts):
        for j in range(num_obs):
            dist_sq = (pts[i, 0] - obstacles_arr[j, 0])**2 + (pts[i, 1] - obstacles_arr[j, 1])**2
            R_sq = obstacles_arr[j, 2]**2
            c_list.append(dist_sq - R_sq)
            
    # 2. waypoints have approximately equal distances
    # diffs = np.diff(pts, axis=0)
    # dists_sq = np.sum(diffs**2, axis=1)
    # if len(dists_sq) > 1:
    #     mean_dist_sq = np.mean(dists_sq)
    #     for d in dists_sq:
    #         c_list.append(0.1 - (d - mean_dist_sq)**2)

    return np.array(c_list)

# ============ Plotting Function ============
def plot_result(initial_path, final_path, title, filename, obstacles_arr=None, start_p=None, end_p=None, space_lim=None):
    if obstacles_arr is None:
        obstacles_arr = obstacles
    if start_p is None:
        start_p = start_point
    if end_p is None:
        end_p = end_point
    if space_lim is None:
        space_lim = space_limit

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, space_lim)
    ax.set_ylim(0, space_lim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True)

    # Draw obstacles
    for obs in obstacles_arr:
        xc, yc, r = obs
        circle = plt.Circle((xc, yc), r, color='red', alpha=0.5, label='_nolegend_')
        ax.add_patch(circle)
        ax.plot(xc, yc, 'r+', markersize=4)

    # Draw start and end points
    ax.plot(start_p[0], start_p[1], 'gs', markersize=10, markerfacecolor='g')
    ax.annotate('Start A', (start_p[0] + 0.5, start_p[1]), fontweight='bold')
    ax.plot(end_p[0], end_p[1], 'bs', markersize=10, markerfacecolor='b')
    ax.annotate('End B', (end_p[0] - 3, end_p[1] + 1), fontweight='bold')

    # Draw initial path
    ax.plot(initial_path[:, 0], initial_path[:, 1], 'k--o',
            linewidth=1, markersize=4, label='Initial Path')

    # Draw optimized path
    ax.plot(final_path[:, 0], final_path[:, 1], 'b-o',
            linewidth=1, markersize=4, markerfacecolor='b', label='Optimized Path')

    ax.legend()
    fig.savefig(f"results/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: results/{filename}")