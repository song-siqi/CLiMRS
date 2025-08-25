import torch

def orientation_error(desired, current):
    desired = desired / torch.norm(desired, dim=-1, keepdim=True)
    current = current / torch.norm(current, dim=-1, keepdim=True)
    cc = torch.cat([current[..., :3] * -1, current[..., 3:]], dim=-1)
    q_r = quat_mul(desired, cc)
    angle = 2 * torch.atan2(torch.norm(q_r[..., :3], dim=-1), q_r[..., 3])
    axis = q_r[..., :3] / (torch.norm(q_r[..., :3], dim=-1, keepdim=True) + 1e-6)
    return axis * angle.unsqueeze(-1)

def quat_mul(q, r):
    x1, y1, z1, w1 = q.unbind(-1)
    x2, y2, z2, w2 = r.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dim=-1)

class FrankaOSCController:
    def __init__(self, kp=5, kv=None, gripper_open=0.04, gripper_close=0.0, close_thresh=0.05):
        self.kp = kp
        self.kv = 2 * (kp ** 0.5) if kv is None else kv
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.close_thresh = close_thresh

    def solve(self, pos_cur, orn_cur, dof_vel, pos_des, orn_des, jacobian, mm, cube_pos=None):
        m_inv = torch.linalg.pinv(mm)
        j_eef = jacobian
        jt = torch.transpose(j_eef, 1, 2)
        lambda_eef = torch.bmm(j_eef, torch.bmm(m_inv, jt))
        m_eef = torch.linalg.pinv(lambda_eef)

        orn_cur = orn_cur / torch.norm(orn_cur, dim=-1, keepdim=True)
        orn_err = orientation_error(orn_des, orn_cur)
        orn_err_norm = torch.norm(orn_err, dim=-1)

        pos_err = self.kp * (pos_des - pos_cur)
        orn_weight = 3
        dpose = torch.cat([pos_err, orn_weight * orn_err], -1)

        u = torch.bmm(jt, torch.bmm(m_eef, dpose.unsqueeze(-1))) - self.kv * torch.bmm(mm, dof_vel)
        u = u.squeeze(-1)

        if cube_pos is not None:
            dist = torch.norm(pos_cur - cube_pos, dim=1)
            gripper_targets = torch.where(
                dist < self.close_thresh,
                torch.full_like(dist, self.gripper_close),
                torch.full_like(dist, self.gripper_open)
            )
        else:
            gripper_targets = torch.full((pos_cur.shape[0],), self.gripper_open, device=pos_cur.device)

        return u, gripper_targets