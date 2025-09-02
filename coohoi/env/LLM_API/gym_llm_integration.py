"""
Gym-LLM Integration Module
This module integrates Isaac Gym environment with LLM decision making system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from isaacgym import gymapi, gymtorch
import torch
from .ask_Llm import LLMWorkflow
from .split_llm_with_skills import parse_workflow_text, SKILL_MAP


class GymEnvironmentObserver:
    """观察器：从 Gym 环境提取状态信息"""
    
    def __init__(self, task):
        self.task = task
        self.device = task.device
        
    def get_environment_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        env_id = 0
        env_ptr = self.task.envs[0]
        
        root_state = self.task.gym.acquire_actor_root_state_tensor(self.task.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        state_info = {
            'area_positions': self._get_area_positions(env_ptr, all_root_state),
            'agent_positions': self._get_agent_positions(env_ptr, all_root_state),
            'component_positions': self._get_component_positions(env_ptr, all_root_state),
            'obstacle_positions': self._get_obstacle_positions(env_ptr, all_root_state),
            'franka_state': self._get_franka_state(env_ptr, all_root_state),
            'mobile_robot_states': self._get_mobile_robot_states(env_ptr, all_root_state)
        }
        
        return state_info
    
    def _get_area_positions(self, env_ptr, root_state) -> Dict[str, Tuple[float, float, float]]:
        """获取区域位置信息"""
        area_positions = {}

        if hasattr(self.task, 'component_cube_handles') and self.task.component_cube_handles:
            for i, handle in enumerate(self.task.component_cube_handles):
                idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                pos = root_state[idx, 0:3].cpu().numpy()
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

                if x >= 0 and y >= 0:
                    label = 'A'
                elif x < 0 and y >= 0:
                    label = 'B'
                elif x < 0 and y < 0:
                    label = 'C'
                else:
                    label = 'D'
                
                area_positions[label] = (x, y, z)
        
        return area_positions
    
    def _get_agent_positions(self, env_ptr, root_state) -> Dict[str, Tuple[float, float]]:
        """获取智能体位置信息"""
        agent_positions = {}
        
        if hasattr(self.task, 'humanoid_handles') and self.task.humanoid_handles:
            humanoid_idx = self.task.gym.get_actor_index(env_ptr, self.task.humanoid_handles[0], gymapi.DOMAIN_SIM)
            pos = root_state[humanoid_idx, 0:2].cpu().numpy()
            agent_positions["<humanoid> (101)"] = (float(pos[0]), float(pos[1]))
        
        # Franka 机械臂位置
        if hasattr(self.task, 'franka_handles') and self.task.franka_handles:
            franka_idx = self.task.gym.get_actor_index(env_ptr, self.task.franka_handles[0], gymapi.DOMAIN_SIM)
            pos = root_state[franka_idx, 0:2].cpu().numpy()
            agent_positions["<franka> (606)"] = (float(pos[0]), float(pos[1]))
        
        # 移动机器人位置
        mobile_robot_names = ["<wheeled robot1> (202)", "<wheeled robot2> (203)", "<wheeled robot3> (204)"]
        if hasattr(self.task, 'mobile_handles') and self.task.mobile_handles:
            for i, handle in enumerate(self.task.mobile_handles):
                if i < len(mobile_robot_names):
                    idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                    pos = root_state[idx, 0:2].cpu().numpy()
                    agent_positions[mobile_robot_names[i]] = (float(pos[0]), float(pos[1]))
        
        return agent_positions
    
    def _get_component_positions(self, env_ptr, root_state) -> Dict[str, Tuple[float, float, float]]:
        """获取组件位置信息"""
        component_positions = {}
        
        # 获取各个组件的位置
        component_mapping = {
            'trunk': (303, self.task.component_handles[2] if hasattr(self.task, 'component_handles') and len(self.task.component_handles) > 2 else None),
            'left wheel': (405, self.task.component_handles[0] if hasattr(self.task, 'component_handles') and len(self.task.component_handles) > 0 else None),
            'right wheel': (406, self.task.component_handles[1] if hasattr(self.task, 'component_handles') and len(self.task.component_handles) > 1 else None)
        }
        
        for name, (id_num, handle) in component_mapping.items():
            if handle is not None:
                idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                pos = root_state[idx, 0:3].cpu().numpy()
                component_positions[f"<{name}> ({id_num})"] = (float(pos[0]), float(pos[1]), float(pos[2]))
        
        return component_positions
    
    def _get_obstacle_positions(self, env_ptr, root_state) -> Dict[str, Tuple[float, float, float]]:
        """获取障碍物位置信息"""
        obstacle_positions = {}
        
        if hasattr(self.task, '_box_handles') and self.task._box_handles:
            # 获取障碍物位置
            for i, handle in enumerate(self.task._box_handles[:9]):  # 前9个是障碍物
                idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                pos = root_state[idx, 0:3].cpu().numpy()
                obstacle_positions[f"<obstacle_{i}> (507)"] = (float(pos[0]), float(pos[1]), float(pos[2]))
        
        return obstacle_positions
    
    def _get_franka_state(self, env_ptr, root_state) -> Dict[str, Any]:
        """获取 Franka 机械臂状态"""
        franka_state = {}
        
        if hasattr(self.task, 'franka_handles') and self.task.franka_handles:
            franka_idx = self.task.gym.get_actor_index(env_ptr, self.task.franka_handles[0], gymapi.DOMAIN_SIM)
            pos = root_state[franka_idx, 0:3].cpu().numpy()
            quat = root_state[franka_idx, 3:7].cpu().numpy()
            
            franka_state = {
                'position': (float(pos[0]), float(pos[1]), float(pos[2])),
                'orientation': (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                'task_stage': getattr(self.task, 'franka_task_stage', 0),
                'gripper_closed': getattr(self.task, 'gripper_closed', False)
            }
        
        return franka_state
    
    def _get_mobile_robot_states(self, env_ptr, root_state) -> Dict[str, Any]:
        """获取移动机器人状态"""
        mobile_states = {}
        
        if hasattr(self.task, 'mobile_handles') and self.task.mobile_handles:
            for i, handle in enumerate(self.task.mobile_handles):
                idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                pos = root_state[idx, 0:3].cpu().numpy()
                quat = root_state[idx, 3:7].cpu().numpy()
                
                mobile_states[f"robot_{i+1}"] = {
                    'position': (float(pos[0]), float(pos[1]), float(pos[2])),
                    'orientation': (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                    'waypoint_idx': getattr(self.task, f'current_wp_idx_{i+1}' if i > 0 else 'current_wp_idx', 0)
                }
        
        return mobile_states


class LLMDecisionExecutor:
    """执行器：将 LLM 决策转换为 Gym 环境动作"""
    
    def __init__(self, task):
        self.task = task
        self.llm_workflow = None
        self.last_decision = None
        
    def initialize_llm(self, area_positions: Dict, agent_positions: Dict):
        """初始化 LLM 工作流"""
        self.llm_workflow = LLMWorkflow(area_positions, agent_positions, {})
        
    def get_llm_decision(self, question: str = "请给出组装方案") -> str:
        """获取 LLM 决策"""
        if self.llm_workflow is None:
            raise ValueError("LLM workflow not initialized")
        
        answer = self.llm_workflow.ask_llm(question)
        self.last_decision = answer
        return answer
    
    def execute_decision(self, decision_text: str) -> bool:
        """执行 LLM 决策"""
        try:
            # 解析工作流文本
            steps = parse_workflow_text(decision_text)
            
            # 执行每个步骤
            for step in steps:
                self.execute_skill_step(step)
            
            return True
        except Exception as e:
            print(f"Error executing decision: {e}")
            return False
    
    def execute_skill_step(self, step: Dict[str, Any]):
        """执行单个技能步骤"""
        print(f"Executing: {step['skill']}, robot: {step['robot_name']}, area: {step['area_name']}, target: {step['target_name']}")
        
        func = SKILL_MAP.get(step['skill'])
        if func:
            func(self.task, step['robot_name'], step['area_name'], step['target_name'])
        else:
            print(f"Unknown skill: {step['skill']}")


class GymLLMIntegration:
    """Gym-LLM 整合主类"""
    
    def __init__(self, task):
        self.task = task
        self.observer = GymEnvironmentObserver(task)
        self.executor = LLMDecisionExecutor(task)
        self.update_frequency = 10  # LLM 更新频率
        self.step_counter = 0
        
    def initialize(self):
        """初始化整合系统"""
        # 获取初始环境状态
        env_state = self.observer.get_environment_state()
        
        # 初始化 LLM 工作流
        self.executor.initialize_llm(
            env_state['area_positions'],
            env_state['agent_positions']
        )
        
        print("Gym-LLM Integration initialized successfully")
    
    def update(self, question: str = "请给出组装方案") -> bool:
        """更新系统：观察环境，获取决策，执行动作"""
        self.step_counter += 1
        
        # 按频率更新 LLM 决策
        if self.step_counter % self.update_frequency == 0:
            # 更新环境状态
            env_state = self.observer.get_environment_state()
            
            # 更新 LLM 工作流状态
            if self.executor.llm_workflow:
                self.executor.llm_workflow.area_positions = env_state['area_positions']
                self.executor.llm_workflow.agent_positions = env_state['agent_positions']
            
            # 获取 LLM 决策
            try:
                decision = self.executor.get_llm_decision(question)
                print(f"LLM Decision: {decision}")
                
                # 执行决策
                success = self.executor.execute_decision(decision)
                if success:
                    print("Decision executed successfully")
                else:
                    print("Failed to execute decision")
                
                return success
            except Exception as e:
                print(f"Error in LLM update: {e}")
                return False
        
        return True
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息（用于调试）"""
        return self.observer.get_environment_state()
    
    def set_update_frequency(self, frequency: int):
        """设置 LLM 更新频率"""
        self.update_frequency = frequency
        print(f"LLM update frequency set to {frequency}")
