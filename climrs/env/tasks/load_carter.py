import gym
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import torch
import math
import torch

# 初始化仿真
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60  # 仿真时间步长
sim_params.substeps = 2  # 子步数
sim_params.up_axis = gymapi.UP_AXIS_Z  # Z轴为向上方向
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)  # 重力
sim_params.use_gpu_pipeline = True  # 使用GPU加速
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 设置环境网格
spacing = 5.0  # 环境间距
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
num_per_row = 1  # 每行的环境数量
num_env = 1
# 创建环境
env = gym.create_env(sim, lower, upper, num_per_row)

# 加载资产
asset_root = "/home/pjlab/Desktop/urdf_model/carter"  # 资产文件夹路径
asset_file = "Carter_v1.urdf"  # 资产文件名（URDF或SDF文件）
asset_options = gymapi.AssetOptions()

asset_options.disable_gravity = False


robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
dof_props_asset = gym.get_asset_dof_properties(robot_asset)

# 设置物体的初始位置和姿态
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0, 0.0)  # 初始位置
pose.r = gymapi.Quat(0, 0, 0, 1)  # 初始姿态（无旋转）

# 创建动态物体
car_actor_handle = gym.create_actor(env, robot_asset, pose, "actor", 0, 0)

props = gym.get_actor_dof_properties(env, car_actor_handle)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)

plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)
gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time (synchronizes the physics simulation with the rendering rate)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

