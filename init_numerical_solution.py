"""
UAV Path Planning - Python Implementation
about constraints and photo/video saving functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation, FFMpegWriter
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

def _segment_min_dist_sq_to_center(p1, p2, center):
    """Minimum squared distance from segment p1->p2 to a point center."""
    seg = p2 - p1
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        return np.sum((p1 - center) ** 2)
    t = np.dot(center - p1, seg) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = p1 + t * seg
    return np.sum((closest - center) ** 2)


def inequality_constraint(x, obstacles_arr, start_p=None, end_p=None, margin=0.5, safety_eps=1e-3):
    pts = x.reshape(-1, 2)
    if start_p is not None and end_p is not None:
        full_path = np.vstack([start_p, pts, end_p])
    else:
        full_path = pts

    num_obs = obstacles_arr.shape[0]
    c_list = []

    # 1) waypoints outside all obstacles
    for i in range(full_path.shape[0]):
        for j in range(num_obs):
            dist_sq = (full_path[i, 0] - obstacles_arr[j, 0])**2 + (full_path[i, 1] - obstacles_arr[j, 1])**2
            R_sq = (obstacles_arr[j, 2] + margin + safety_eps) ** 2
            c_list.append(dist_sq - R_sq)

    # 2) segments outside all obstacles (critical to avoid chord crossing)
    for i in range(full_path.shape[0] - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        for j in range(num_obs):
            center = obstacles_arr[j, :2]
            R_sq = (obstacles_arr[j, 2] + margin + safety_eps) ** 2
            min_dist_sq = _segment_min_dist_sq_to_center(p1, p2, center)
            c_list.append(min_dist_sq - R_sq)

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

def animate_result(initial_path, final_path, title, filename, obstacles_arr=None, start_p=None, end_p=None, space_lim=None):
    
    if obstacles_arr is None:
        from main import obstacles
        obstacles_arr = obstacles
    if start_p is None:
        from main import start_point
        start_p = start_point
    if end_p is None:
        from main import end_point
        end_p = end_point
    if space_lim is None:
        from main import space_limit
        space_lim = space_limit

    os.makedirs('video', exist_ok=True)

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
        circle = plt.Circle((xc, yc), r, color='red', alpha=0.5)
        ax.add_patch(circle)
        ax.plot(xc, yc, 'r+', markersize=4)

    # Draw start and end points
    ax.plot(start_p[0], start_p[1], 'gs', markersize=10, markerfacecolor='g')
    ax.annotate('Start A', (start_p[0] + 0.5, start_p[1]), fontweight='bold')
    ax.plot(end_p[0], end_p[1], 'bs', markersize=10, markerfacecolor='b')
    ax.annotate('End B', (end_p[0] - 3, end_p[1] + 1), fontweight='bold')

    # Draw initial path faintly
    ax.plot(initial_path[:, 0], initial_path[:, 1], 'k--^',
            linewidth=1, markersize=3, label='Initial Path', alpha=0.5)

    line, = ax.plot([], [], 'b-', linewidth=2, label='Optimized Path')
    drone, = ax.plot([], [], 'ro', markersize=8, label='UAV')

    ax.legend(loc='upper left')

    def init():
        line.set_data([], [])
        drone.set_data([], [])
        return line, drone

    def update(frame):
        line.set_data(final_path[:frame+1, 0], final_path[:frame+1, 1])
        if frame < len(final_path):
            drone.set_data([final_path[frame, 0]], [final_path[frame, 1]])
        return line, drone

    frames = len(final_path)
    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True, interval=100)

    save_path = os.path.join('video', filename)
    ani.save(save_path, dpi=100, writer=FFMpegWriter(fps=10))
    plt.close()
    print(f"Animation saved: {save_path}")
