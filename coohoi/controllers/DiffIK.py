import numpy as np
from scipy.spatial.transform import Rotation as R

class DiffIK:
    def __init__(self, dt=0.002, max_angvel=0.4, pos_vel=0.1, rot_vel=0.3):
        self.dt = dt
        damping = 1e-4
        self.diag = np.eye(6) * damping
        self.err = np.zeros(6)
        self.max_angvel = max_angvel
        self.pos_vel = pos_vel
        self.rot_vel = rot_vel

    def cal_dq(self, site_pos, site_quat, jac, target_pos, target_quat):
        # 位置误差
        err_pos = target_pos - site_pos
        err_pos_norm = np.linalg.norm(err_pos)
        if err_pos_norm > self.pos_vel:
            err_pos *= (self.pos_vel / err_pos_norm)
        self.err[:3] = err_pos

        # 姿态误差（四元数转旋转向量）
        r_site = R.from_quat(site_quat)
        r_target = R.from_quat(target_quat)
        r_err = r_target * r_site.inv()
        err_rotvec = r_err.as_rotvec()
        err_ori_norm = np.linalg.norm(err_rotvec)
        if err_ori_norm > self.rot_vel:
            err_rotvec *= (self.rot_vel / err_ori_norm)
        self.err[3:6] = err_rotvec

        # DLS求解
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.diag, self.err)
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max
        return dq