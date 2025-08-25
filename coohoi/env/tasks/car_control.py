import numpy as np
import torch

class DifferentialDriveController:
    def __init__(self, wheel_base, max_speed):
        """
        初始化差速驱动控制器
        :param wheel_base: 两轮之间的距离
        :param max_speed: 轮子的最大速度
        """
        self.wheel_base = wheel_base
        self.max_speed = max_speed

    def quat_to_yaw(self, quat):
        if len(quat.shape) == 0:
            quat = quat.unsqueeze(0)
        w, x, y, z = quat[3], quat[0], quat[1], quat[2]  # [x,y,z,w]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def set_wheel_velocities(self, left_vel, right_vel):
        """
        设置左右轮子的速度
        :param left_vel: 左轮速度
        :param right_vel: 右轮速度
        """
        print(f"设置左轮速度: {left_vel:.2f}, 右轮速度: {right_vel:.2f}")

    def control_to_waypoint(self, current_pos, current_quat, target_waypoint):
        """
        控制机器人朝向目标路径点
        :param current_pos: 当前位姿 [x, y, z]
        :param current_quat: 当前朝向四元数 [x, y, z, w]
        :param target_waypoint: 目标路径点 [x, y, z]
        """
        direction = np.array(target_waypoint[:2]) - np.array(current_pos[:2])
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            self.set_wheel_velocities(0, 0)
            return

        direction = direction / distance
        target_angle = np.arctan2(direction[1], direction[0])

        # 计算当前朝向角
        x, y, z, w = current_quat
        current_angle = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        angle_diff = target_angle - current_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        linear_speed = min(self.max_speed, distance * 2.0)
        angular_speed = angle_diff * 0.5

        left_vel = linear_speed - angular_speed * self.wheel_base / 2.0
        right_vel = linear_speed + angular_speed * self.wheel_base / 2.0

        left_vel = np.clip(left_vel, -self.max_speed, self.max_speed)
        right_vel = np.clip(right_vel, -self.max_speed, self.max_speed)

        self.set_wheel_velocities(left_vel, right_vel)

# 示例用法
if __name__ == "__main__":
    controller = DifferentialDriveController(wheel_base=0.5, max_speed=5.0)
    current_pos = [0.0, 0.0, 0.0]
    current_quat = [0.0, 0.0, 0.0, 1.0]
    target_waypoint = [2.0, 2.0, 0.0]
    controller.control_to_waypoint(current_pos, current_quat, target_waypoint)