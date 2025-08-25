import numpy as np
from dataclasses import dataclass
import pybullet as p

@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray

class FrankaIKGym:
    def __init__(self, urdf_path: str, base_pos=(0, -2.0, 0), base_orn=(0, 0, 0, 1)):
        self.p = p
        self.physics_client = p.connect(p.DIRECT)
        self.robot = p.loadURDF(
            urdf_path,
            basePosition=base_pos,
            baseOrientation=base_orn,
            useFixedBase=True
        )
        self.ee_link = 7  # panda_link7
        self.dof = 7
        self.last_q = np.zeros(self.dof)
        self.lower_limits = []
        self.upper_limits = []
        for i in range(self.dof):
            info = self.p.getJointInfo(self.robot, i)
            self.lower_limits.append(info[8])
            self.upper_limits.append(info[9])
        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)

    def solve(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        输入: pos (3,), quat (4,) (wxyz)
        输出: 7维关节角
        """
        # PyBullet期望四元数为xyzw
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        q = self.p.calculateInverseKinematics(
            self.robot, self.ee_link, pos, quat_xyzw,
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=(self.upper_limits - self.lower_limits).tolist(),
            restPoses=self.last_q.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4
        )
        q = np.array(q[:self.dof])
        q = np.clip(q, self.lower_limits, self.upper_limits)
        self.last_q = q
        # 可选：同步pybullet内部状态（用于FK验证）
        for i in range(self.dof):
            self.p.resetJointState(self.robot, i, q[i])
        # FK验证
        link_state = self.p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        ee_pos = np.array(link_state[4])
        pos_err = np.linalg.norm(ee_pos - pos)
        # print(f"IK pos_err: {pos_err:.4f}")
        return q

    def disconnect(self):
        self.p.disconnect(self.physics_client)