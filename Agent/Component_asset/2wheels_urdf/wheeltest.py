# test_robot_wheels.py
import numpy as np
from isaacgym import gymapi, gymtorch
import torch
class WheelTestEnv:
    def __init__(self):
        # 初始化 Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # 设置仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        
        # 物理引擎参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # 创建仿真
        self.device = "cpu"
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # 创建环境
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        
        # 加载机器人
        self.robot_handle = self.load_robot()
        
        # 初始化张量
        self.prepare_tensors()
        
        # 创建viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.setup_camera()
        print("测试环境初始化完成")
    
    def setup_camera(self):
        """设置俯视相机视角"""
        cam_pos = gymapi.Vec3(-6, 0, 2)    
        cam_target = gymapi.Vec3(0, 0, 0.5)
        
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_robot(self):
        """加载机器人模型"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  # 确保基座不固定
        asset_options.disable_gravity = False
        
        asset_root = "2wheels_urdf"
        asset_file = "ranger_mini_v2/urdf/ranger_mini.urdf"
        
        try:
            robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            print(f"成功加载机器人模型: {asset_root}/{asset_file}")
        except Exception as e:
            print(f"加载机器人模型失败: {e}")
            return None
        
        # 创建机器人actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0.5)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        
        robot_handle = self.gym.create_actor(self.env, robot_asset, pose, "test_robot", 0, 0)
        
        # 获取DOF属性
        dof_props = self.gym.get_actor_dof_properties(self.env, robot_handle)
        dof_count = self.gym.get_actor_dof_count(self.env, robot_handle)
        
        print(f"机器人DOF数量: {dof_count}")
        
        # 设置DOF属性为速度控制
        for i in range(dof_count):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_VEL
            dof_props["stiffness"][i] = 0.0
            dof_props["damping"][i] = 100.0
            dof_props["friction"][i] = 0.1
            dof_props["velocity"][i] = 10.0  # 最大速度
            
        print(f"DOF属性: {dof_props}")
        
        self.gym.set_actor_dof_properties(self.env, robot_handle, dof_props)
        
        return robot_handle
    
    def prepare_tensors(self):
        """准备仿真张量"""
        self.gym.prepare_sim(self.sim)
        
        # 获取DOF状态张量
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        
        # 获取根状态张量
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_actor_root_state)
        
        print(f"DOF状态张量形状: {self.dof_states.shape}")
        print(f"根状态张量形状: {self.root_states.shape}")
    
    def set_wheel_velocities(self, left_vel, right_vel):
        """设置轮子速度"""
        if self.robot_handle is None:
            return
            
        dof_count = self.gym.get_actor_dof_count(self.env, self.robot_handle)
        
        if dof_count >= 2:
            # 使用numpy数组，更简单
            vel_targets = np.zeros(self.dof_states.shape[0])
            vel_targets[0] = left_vel   # 左轮
            vel_targets[1] = right_vel  # 右轮
            
            # 转换为tensor并设置
            vel_tensor = torch.from_numpy(vel_targets).float()
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_tensor))
            
            print(f"设置轮子速度: 左轮={left_vel:.2f}, 右轮={right_vel:.2f}")
    
    def step(self, left_vel, right_vel):
        """执行一个仿真步骤"""
        # 设置轮子速度
        self.set_wheel_velocities(left_vel, right_vel)
        
        # 步进仿真
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 刷新张量
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 更新viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        
        # 获取当前状态
        current_pos = self.root_states[0, 0:3].cpu().numpy()
        current_vel = self.dof_states[0:2, 1].cpu().numpy() if self.dof_states.shape[0] >= 2 else [0, 0]
        
        return current_pos, current_vel
    
    def run_test(self):
        """运行测试"""
        print("\n开始轮子转动测试...")
        
        test_cases = [
            (5.0, 5.0),    # 增加测试速度
            (3.0, 6.0),    # 右转
            (6.0, 3.0), 
            (3.0, 3.0), 
            (0.0, 0.0),    # 停止
        ]
        
        for i, (left_vel, right_vel) in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: 左轮={left_vel}, 右轮={right_vel}")
            
            # 运行100步
            for step in range(100):
                pos, vel = self.step(left_vel, right_vel)
                
                if step % 20 == 0:  # 每20步打印一次
                    print(f"  步骤 {step}: 位置={pos}, 轮子速度={vel}")
                
                # 检查退出
                if self.gym.query_viewer_has_closed(self.viewer):
                    return False
        
        print("\n测试完成！")
        return True
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'viewer'):
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, 'sim'):
            self.gym.destroy_sim(self.sim)

def main():
    """主函数"""
    try:
        # 创建测试环境
        test_env = WheelTestEnv()
        
        # 运行测试
        test_env.run_test()
        
        # 等待用户关闭
        print("\n按任意键或关闭窗口退出...")
        while not test_env.gym.query_viewer_has_closed(test_env.viewer):
            test_env.gym.step_graphics(test_env.sim)
            test_env.gym.draw_viewer(test_env.viewer, test_env.sim, True)
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'test_env' in locals():
            test_env.cleanup()

if __name__ == "__main__":
    main()