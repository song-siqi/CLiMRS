def preprocess_corner_path(path, extra_dist=0.3):
    """
    对轴对齐路径进行拐角预处理：
    如果最后一段是拐弯（即倒数第二段和最后一段方向不同），
    则在拐弯前多加一个预备点（在拐弯前的方向上多走 extra_dist）。
    """
    if len(path) < 3:
        return path
    p1 = np.array(path[-3])
    p2 = np.array(path[-2])
    p3 = np.array(path[-1])
    dir1 = p2 - p1
    dir2 = p3 - p2
    # 判断是否拐弯（方向变化）
    if not np.allclose(dir1/np.linalg.norm(dir1), dir2/np.linalg.norm(dir2), atol=1e-2):
        # 在 p2 点前的方向上多走 extra_dist
        prep = p2 + (dir1/np.linalg.norm(dir1)) * extra_dist
        # 如果 prep 和 p3 距离比 p2 近，说明多走会超过目标，跳过
        if np.linalg.norm(prep - p3) < np.linalg.norm(p2 - p3):
            return path
        new_path = path[:-2] + [prep, p2, p3]
        return new_path
    return path
import torch
import numpy as np
from rrt_algorithms.rrt.rrt_star_bid import RRTStarBidirectional
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymtorch

def plot_obstacles(X_dimensions, obstacles, start, goal, path=None):
    fig, ax = plt.subplots()
    ax.set_xlim(X_dimensions[0])
    ax.set_ylim(X_dimensions[1])
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1], color='gray')
        ax.add_patch(rect)
    ax.plot(start[0], start[1], 'go')  # 起点
    ax.plot(goal[0], goal[1], 'ro')    # 终点
    if path is not None:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], 'b.-')
    plt.show()

def assign_boxes_to_cars_by_distance(gym, env_ptr, root_state, car_handles, box_handles):
    car_positions = []
    for car_handle in car_handles:
        idx = gym.get_actor_index(env_ptr, car_handle, gymapi.DOMAIN_SIM)
        car_positions.append(root_state[idx, 0:2].cpu().numpy())
    box_positions = []
    for box_handle in box_handles:
        idx = gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
        box_positions.append(root_state[idx, 0:2].cpu().numpy())

    car_pos_list = car_positions.copy()
    box_pos_list = box_positions.copy()
    car_handles_list = car_handles.copy()
    box_handles_list = box_handles.copy()
    assigned_cars = []
    assigned_boxes = []
    while box_pos_list and car_pos_list:
        min_dist = float('inf')
        min_i, min_j = 0, 0
        for j, box_pos in enumerate(box_pos_list):
            for i, car_pos in enumerate(car_pos_list):
                dist = np.linalg.norm(car_pos - box_pos)
                if dist < min_dist:
                    min_dist = dist
                    min_i, min_j = i, j
        assigned_cars.append(car_handles_list[min_i])
        assigned_boxes.append(box_handles_list[min_j])
        car_pos_list.pop(min_i)
        box_pos_list.pop(min_j)
        car_handles_list.pop(min_i)
        box_handles_list.pop(min_j)
    
    return assigned_cars, assigned_boxes

def assign_box_targets_for_franka_area(gym, env_ptr, root_state, box_handles, franka_handle):
    franka_idx = gym.get_actor_index(env_ptr, franka_handle, gymapi.DOMAIN_SIM)
    franka_center = root_state[franka_idx, 0:2].cpu().numpy()
    offset = 1.0  

    targets = []
    center = franka_center + np.array([offset, 0])
    targets.append((box_handles[1], center))
    left_pos = franka_center + np.array([0, -offset])
    right_pos = franka_center + np.array([0, offset])
    other_boxes = [box_handles[0], box_handles[2]]
    box_positions = []
    for box_handle in other_boxes:
        idx = gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
        box_positions.append(root_state[idx, 0:2].cpu().numpy())
    dists_left = [np.linalg.norm(pos - left_pos) for pos in box_positions]
    if dists_left[0] < dists_left[1]:
        targets.append((other_boxes[0], left_pos))
        targets.append((other_boxes[1], right_pos))
    else:
        targets.append((other_boxes[1], left_pos))
        targets.append((other_boxes[0], right_pos))
    return targets

def is_segment_blocked(p1, p2, obstacles):
    for obs in obstacles:
        xmin, ymin, xmax, ymax = obs
        if (min(p1[0], p2[0]) <= xmax and max(p1[0], p2[0]) >= xmin and
            min(p1[1], p2[1]) <= ymax and max(p1[1], p2[1]) >= ymin):
            return True
    return False

def find_block_edge(p1, p2, obstacles, axis):
            # axis: 0 for x, 1 for y
            step = 0.05
            direction = np.sign(p2[axis] - p1[axis])
            current = np.array(p1)
            while abs(current[axis] - p2[axis]) > step:
                current[axis] += direction * step
                if is_segment_blocked(p1, current, obstacles):
                    current[axis] -= direction * step
                    break
            return np.array(current)

def move_along_axis(current, target, obstacles, axis):
    """
    沿指定轴方向移动到目标位置
    axis: 0 for x, 1 for y
    返回: (new_position, success)
    """
    if np.isclose(current[axis], target[axis], atol=1e-2):
        return current.copy(), True
    
    next_pos = current.copy()
    next_pos[axis] = target[axis]
    
    if not is_segment_blocked(current, next_pos, obstacles):
        return next_pos.copy(), True
    else:
        block_pos = find_block_edge(current, next_pos, obstacles, axis)
        if not np.allclose(current, block_pos, atol=1e-2):
            return block_pos.copy(), False
        else:
            return current.copy(), False


def plan_paths_for_cars_and_boxes(gym, sim, device, env_ptr, root_state, car_handles, box_handles, all_obstacle_handles, max_samples=1024):
    assigned_cars, assigned_boxes = assign_boxes_to_cars_by_distance(gym, env_ptr, root_state, car_handles, box_handles)
    paths = []
    for car_handle, box_handle in zip(assigned_cars, assigned_boxes):
        path = plan_path_to_box(
            gym, sim, device, env_ptr, root_state, car_handle, box_handle, all_obstacle_handles, max_samples
        )
        paths.append(path)
    return paths

def plan_path_to_box(gym, sim, device, env_ptr, root_state, robot_handle, box_handle, all_obstacle_handles, max_samples=1024):
    robot_idx = gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
    box_idx = gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)

    robot_pos = root_state[robot_idx, 0:2].cpu().numpy()
    box_pos = root_state[box_idx, 0:2].cpu().numpy()
    robot_size = 1.0
    default_box_length_size = 0.5
    default_box_width_size = 0.5

    box_pos_rounded = tuple(np.round(box_pos, 1))
    if box_pos_rounded == (4.0, 8.0):
        push_target = np.array([5.0, 8.0])
    elif box_pos_rounded == (-4.0, 8.0):
        push_target = np.array([-4.0, 9.0])
    elif box_pos_rounded == (-4.0, -8.0):
        push_target = np.array([-5.0, -8.0])
    elif box_pos_rounded == (4.0, -8.0):
        push_target = np.array([4.0, -9.0])
    else:
        franka_pos = np.array([-0.05, -3.0])
        direction = box_pos - franka_pos
        direction = direction / np.linalg.norm(direction)
        push_offset = default_box_length_size / 2 + robot_size / 2 + 0.1
        push_target = box_pos + direction * push_offset

    obstacles = []
    idx = gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
    pos = root_state[idx, 0:2].cpu().numpy()
    box_length = default_box_length_size
    box_width = default_box_width_size
    obstacles.append((
        pos[0] - box_length/2 - robot_size/2, pos[1] - box_width/2 - robot_size/2,
        pos[0] + box_length/2 + robot_size/2, pos[1] + box_width/2 + robot_size/2
    ))
    for i in range(root_state.shape[0]):
        if i == robot_idx or i == box_idx:
            continue
        obj_pos = root_state[i, 0:2].cpu().numpy()
        obstacles.append((
            obj_pos[0] - 0.5, obj_pos[1] - 0.5,
            obj_pos[0] + 0.5, obj_pos[1] + 0.5
        ))

    for i, obstacle_handle in enumerate(all_obstacle_handles[5:9]):
        obstacle_idx = gym.get_actor_index(env_ptr, obstacle_handle, gymapi.DOMAIN_SIM)
        obstacle_pos = root_state[obstacle_idx, 0:2].cpu().numpy()
        if i < 2:
            box_length = default_box_length_size * 2 * 5.0 
            box_width = default_box_width_size * 2 * 0.25 
        else:
            box_length = default_box_length_size * 2 * 0.25 
            box_width = default_box_width_size * 2 * 5.0 
        obstacle_size = [box_length, box_width]
        obstacles.append((
            obstacle_pos[0] - obstacle_size[0] / 2 - robot_size/2, obstacle_pos[1] - obstacle_size[1] / 2 - robot_size/2,
            obstacle_pos[0] + obstacle_size[0] / 2 + robot_size/2, obstacle_pos[1] + obstacle_size[1] / 2 + robot_size/2
        ))
    # franka障碍物
    franka_pos = [-0.05, -3.0]
    franka_size = [1.5, 1.5] 
    obstacles.append((
        franka_pos[0] - franka_size[0]/2, franka_pos[1] - franka_size[1]/2,
        franka_pos[0] + franka_size[0]/2, franka_pos[1] + franka_size[1]/2
    ))
    X_dimensions = np.array([(-6, 6), (-10, 10)]) 
    q = 1
    r = 0.1
    rewire_count = 32
    prc = 0.1

    X = SearchSpace(X_dimensions, obstacles)
    rrt = RRTStarBidirectional(X, q, robot_pos, push_target, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star_bidirectional()
      
    if path is None:
        print("fail to find path")
        return
    if not isinstance(path, list):
        path = list(path)
    if np.linalg.norm(np.array(path[0]) - np.array(robot_pos)) > np.linalg.norm(np.array(path[-1]) - np.array(robot_pos)):
        path.reverse()
    return path


def plan_path_to_franka(gym, sim, device, env_ptr, root_state, robot_handle, box_handle, all_obstacle_handles, max_samples=1024, custom_push_target=None,start_pos=None, force_axis_aligned=True):
    if start_pos is not None:
        robot_pos = start_pos
    else:
        robot_idx = gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
        robot_pos = root_state[robot_idx, 0:2].cpu().numpy()
    box_idx = gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
    box_pos = root_state[box_idx, 0:2].cpu().numpy()
    
    robot_size = 0.8
    default_box_length_size = 0.5
    default_box_width_size = 0.5
    
    if custom_push_target is not None:
        push_target = custom_push_target
    else:
        push_target = box_pos

    obstacles = []
    for i, obstacle_handle in enumerate(all_obstacle_handles[5:9]):
        obstacle_idx = gym.get_actor_index(env_ptr, obstacle_handle, gymapi.DOMAIN_SIM)
        obstacle_pos = root_state[obstacle_idx, 0:2].cpu().numpy()
        if i < 2:
            box_length = default_box_length_size * 2 * 5.0 
            box_width = default_box_width_size * 2 * 0.25 
        else:
            box_length = default_box_length_size * 2 * 0.25 
            box_width = default_box_width_size * 2 * 5.0 
        obstacle_size = [box_length, box_width]
        obstacles.append((
            obstacle_pos[0] - obstacle_size[0] / 2 - robot_size/2, obstacle_pos[1] - obstacle_size[1] / 2 - robot_size/2,
            obstacle_pos[0] + obstacle_size[0] / 2 + robot_size/2, obstacle_pos[1] + obstacle_size[1] / 2 + robot_size/2
        ))
    franka_pos = [-0.05, -3.0]
    franka_size = [1.5, 1.5] 
    obstacles.append((
        franka_pos[0] - franka_size[0]/2, franka_pos[1] - franka_size[1]/2,
        franka_pos[0] + franka_size[0]/2, franka_pos[1] + franka_size[1]/2
    ))

    if force_axis_aligned:
        path = [np.array(robot_pos)]
        current = np.array(robot_pos)
        target = np.array(push_target)
        use_axis = True

        target_rounded = tuple(np.round(target, 1))
        if target_rounded in [(4.0, 8.0), (-4.0, -8.0)]:
            axes = [0, 1]
        elif target_rounded in [(4.0, -8.0), (-4.0, 8.0)]:
            axes = [1, 0]
        else:
            axes = [0, 1]

        for axis in axes:
            new_pos, success = move_along_axis(current, target, obstacles, axis)
            if not success:
                use_axis = False
                break
            if not np.allclose(current, new_pos, atol=1e-2):
                path.append(new_pos)
                current = new_pos

        if use_axis:
            box_center = np.array(push_target)
            if len(path) >= 2:
                prev = path[-1]
                dir_vec = box_center - prev
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-3:
                    unit_dir = dir_vec / norm
                    prep = (box_center - unit_dir * 1.0).tolist()
                    path.extend([prep, box_center.tolist()])
            else:
                path.append(box_center.tolist())

            X_dimensions = np.array([(-6, 6), (-10, 10)])
            # plot_obstacles(X_dimensions, obstacles, robot_pos, push_target, path)
            return path

    X_dimensions = np.array([(-6, 6), (-10, 10)]) 
    q = 1
    r = 0.1
    rewire_count = 32
    prc = 0.1

    X = SearchSpace(X_dimensions, obstacles)
    rrt = RRTStarBidirectional(X, q, robot_pos, push_target, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star_bidirectional()
      
    if path is None:
        print("fail to find path")
        return
    if not isinstance(path, list):
        path = list(path)
    if np.linalg.norm(np.array(path[0]) - np.array(robot_pos)) > np.linalg.norm(np.array(path[-1]) - np.array(robot_pos)):
        path.reverse()
    return path

def plan_paths_for_boxes_to_franka_area(gym, sim, device, env_ptr, root_state, car_handles, box_handles, franka_handle, all_obstacle_handles, max_samples=1024, start_positions=None):
    targets = assign_box_targets_for_franka_area(gym, env_ptr, root_state, box_handles, franka_handle)
    paths = []
    for i, (box_handle, target_pos) in enumerate(targets):
        car_handle = car_handles[i]
        if start_positions is not None:
            start_pos = start_positions[i]
        else:
            start_pos = None
        path = plan_path_to_franka(
            gym, sim, device, env_ptr, root_state, car_handle, box_handle, all_obstacle_handles, max_samples,
            custom_push_target=target_pos, start_pos=start_pos
        )
        paths.append(path)
    return paths