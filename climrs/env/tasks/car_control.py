import numpy as np
import torch

class DifferentialDriveController:
    def __init__(self, wheel_base, max_speed):
        """
        Initialize differential drive controller
        :param wheel_base: distance between wheels
        :param max_speed: maximum wheel speed
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
        Set left and right wheel velocities
        :param left_vel: left wheel velocity
        :param right_vel: right wheel velocity
        """
        print(f"Set left wheel velocity: {left_vel:.2f}, right wheel velocity: {right_vel:.2f}")

    def control_to_waypoint(self, current_pos, current_quat, target_waypoint):
        """
        Control robot to target waypoint
        :param current_pos: current position [x, y, z]
        :param current_quat: current orientation quaternion [x, y, z, w]
        :param target_waypoint: target waypoint [x, y, z]
        """
        direction = np.array(target_waypoint[:2]) - np.array(current_pos[:2])
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            self.set_wheel_velocities(0, 0)
            return

        direction = direction / distance
        target_angle = np.arctan2(direction[1], direction[0])

        # Calculate current orientation angle
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