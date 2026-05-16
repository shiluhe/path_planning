import numpy as np
import random

def is_collision_free(pt, obstacles_arr, margin=0.5):
    for obs in obstacles_arr:
        dist = np.sqrt((pt[0] - obs[0])**2 + (pt[1] - obs[1])**2)
        if dist <= obs[2] + margin:
            return False
    return True

def is_edge_collision_free(pt1, pt2, obstacles_arr, margin=0.5, resolution=0.5):
    dist = np.linalg.norm(pt2 - pt1)
    n_pts = int(dist / resolution)
    for i in range(n_pts + 1):
        pt = pt1 + (pt2 - pt1) * (i / n_pts if n_pts > 0 else 0)
        if not is_collision_free(pt, obstacles_arr, margin):
            return False
    return True

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def get_rrt_path(start, end, space_limit, obstacles_arr, step_size=1.0, max_iter=5000):
    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])
    
    node_list = [start_node]
    
    for _ in range(max_iter):
        # 10% 概率向终点采样，90% 全局随机
        if random.random() > 0.1:
            rnd_x = random.uniform(0, space_limit)
            rnd_y = random.uniform(0, space_limit)
            rnd_node = Node(rnd_x, rnd_y)
        else:
            rnd_node = Node(end_node.x, end_node.y)
            
        # 寻找最近树节点
        nearest_node = min(node_list, key=lambda n: (n.x - rnd_node.x)**2 + (n.y - rnd_node.y)**2)
        
        # 步进
        theta = np.arctan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
        new_x = nearest_node.x + step_size * np.cos(theta)
        new_y = nearest_node.y + step_size * np.sin(theta)
        
        if not (0 <= new_x <= space_limit and 0 <= new_y <= space_limit):
            continue
            
        new_node = Node(new_x, new_y)
        new_pt = np.array([new_x, new_y])
        nearest_pt = np.array([nearest_node.x, nearest_node.y])
        
        # 碰撞检测
        if is_edge_collision_free(nearest_pt, new_pt, obstacles_arr):
            new_node.parent = nearest_node
            node_list.append(new_node)
            
            # 检查是否到达终点附近
            if np.sqrt((new_node.x - end_node.x)**2 + (new_node.y - end_node.y)**2) <= step_size:
                if is_edge_collision_free(np.array([new_node.x, new_node.y]), np.array([end_node.x, end_node.y]), obstacles_arr):
                    end_node.parent = new_node
                    node_list.append(end_node)
                    break
                    
    if end_node.parent is None:
        print("No valid path found！")
        return None
        
    path = []
    curr = end_node
    while curr is not None:
        path.append([curr.x, curr.y])
        curr = curr.parent
        
    path.reverse()
    return np.array(path)
