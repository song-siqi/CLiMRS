import numpy as np

waypoints = [
    (1.0, 1.0),
    (2.5, 0.5),
    (4.0, 3.0),
    (5.5, 1.5),
]

# 参数设置
pos_threshold = 0.1       # 距离小于 0.1m 切换 waypoint
max_linear_vel = 2.0      # 最大线速度
max_angular_vel = 1.0     # 最大角速度

def compute_control(current_pos, current_yaw, target_pos):
    """
    简单的纯跟踪控制（Pure Pursuit / Proportional steering）：
      - 线速度：与距离成正比（或固定）
      - 角速度：与航向误差成正比
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = np.hypot(dx, dy)
    angle_to_goal = np.arctan2(dy, dx)
    yaw_error = angle_to_goal - current_yaw
    # 归一化 yaw_error 到 [-pi, pi]
    yaw_error = (yaw_error + np.pi) % (2*np.pi) - np.pi

    # 控制律（可根据需求调参）
    v = min(max_linear_vel, 1.5 * distance)
    w = np.clip(2.0 * yaw_error, -max_angular_vel, max_angular_vel)
    return v, w

def run_waypoint_navigation(env):
    current_wp_idx = 0
    num_wps = len(waypoints)

    # 主仿真循环
    while current_wp_idx < num_wps:
        # 从环境中获得小车观测：位置 (x,y) 和航向 yaw
        obs = env.get_observations()
        pos = obs["position"]    # e.g. np.array([x, y])
        yaw = obs["yaw"]         # scalar，弧度

        target = waypoints[current_wp_idx]
        dist_to_wp = np.linalg.norm(pos - np.array(target))

        # 如果到达当前 waypoint，就切换到下一个
        if dist_to_wp < pos_threshold:
            print(f"Reached waypoint {current_wp_idx} at {target}")
            current_wp_idx += 1
            continue

        # 否则计算控制命令并发送
        v, w = compute_control(pos, yaw, target)
        action = {
            "linear_velocity": v,
            "angular_velocity": w
        }
        env.step(action)

    print("All waypoints reached!")

# 在实际使用时，把 env 换成你自己的环境接口
if __name__ == "__main__":
    # env = make_your_isaacgym_env()
    run_waypoint_navigation(env)