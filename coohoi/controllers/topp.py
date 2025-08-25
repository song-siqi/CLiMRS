from typing import Callable, List
from dataclasses import dataclass
import numpy as np
import toppra as ta
from controllers.kinematics import Pose

class Topp:
    # re-parameterize trajectory using TOPP-RA

    def __init__(self, dof: int, qc_vel: float, qc_acc: float, ik: Callable):
        self.dof = dof
        self.qc_vel = [(-qc_vel, qc_vel)] * self.dof
        self.qc_vel = ta.constraint.JointVelocityConstraint(np.array(self.qc_vel))
        self.qc_acc = [(-qc_acc, qc_acc)] * self.dof
        self.qc_acc = ta.constraint.JointAccelerationConstraint(np.array(self.qc_acc))
        self.ik: Callable = ik
    
    def jnt_traj(self, pose_path: List[Pose], traj_type: str = "linear"):
        print("pose_path len:", len(pose_path))
        for i, pose in enumerate(pose_path):
            print(f"pose {i}: pos={pose.pos}, quat={pose.quat}")
        assert self.ik is not None, "IK solver not set"
        ss = np.linspace(0, 1, len(pose_path))
        jnts = []
        for pose in pose_path:
            q = self.ik(pose.pos, pose.quat)
            print(f"IK target pos: {pose.pos}, quat: {pose.quat}, IK result: {q}")
            jnts.append(q)
        if traj_type == "linear":
            path = ta.SplineInterpolator(ss, jnts)
        else:
            raise ValueError(f"Unsupported traj_type: {traj_type}")
        instance = ta.algorithm.TOPPRA([self.qc_vel, self.qc_acc], path)
        return instance.compute_trajectory(0, 0)

    @staticmethod
    def query(traj: ta.interpolator.AbstractGeometricPath, t: float):
        t = np.clip(t, 0, traj.duration)
        return traj.eval(t)