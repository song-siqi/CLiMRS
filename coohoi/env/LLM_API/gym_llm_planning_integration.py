"""
Gym-LLM Planning Integration System
整合 LLM/dev_revision 的规划系统与 gym 环境的执行系统
"""

import os
import sys
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from isaacgym import gymapi, gymtorch
from LLM.dev_revision.llm_agents.feedback_agent import FeedbackAgent
from LLM.dev_revision.arena import ArenaMultiAgent
from .gym_llm_integration import GymLLMIntegration

LLM_AVAILABLE = True

class AgentPriority(Enum):
    HUMAN = 3      
    FRANKA = 2     
    MOBILE = 1     


@dataclass
class AgentState:
    id: str
    name: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    priority: AgentPriority
    current_action: Optional[str] = None
    is_executing: bool = False
    last_update: float = 0.0


@dataclass
class CollisionRisk:
    agent1_id: str
    agent2_id: str
    distance: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommended_action: str
    priority_agent: str


class GymLLMPlanningIntegration:
    """
    Gym-LLM 规划整合系统
    将 LLM 规划与 gym 环境执行结合起来
    """
    
    def __init__(self, task, config_path: str = None):
        self.task = task
        self.config_path = config_path
        
        self.llm_arena = None
        self.llm_agents = {}
        self.planning_thread = None
        self.planning_active = False
        
        self.agent_states = {}
        self.agent_priorities = {}
        self.collision_thresholds = {
            'low': 2.0,      
            'medium': 1.5,   
            'high': 1.0,     
            'critical': 0.5  
        }
        self.mobile_robot_stop_flags = {}
        
        self.current_plan = []
        self.plan_execution_index = 0
        self.plan_status = "idle"  
            
        self.safety_mode = True
        self.emergency_stop = False
        
        self._initialize_agent_priorities()
        self._initialize_llm_system()

        self.llm_control_agents = {}  
        self._initialize_llm_control()
    
    def _initialize_llm_control(self):
        try:

            if hasattr(self.task, 'franka_handles') and self.task.franka_handles:
                self.llm_control_agents['franka'] = GymLLMIntegration(self.task)
                
            if hasattr(self.task, 'humanoid_handles') and self.task.humanoid_handles:
                self.llm_control_agents['humanoid'] = GymLLMIntegration(self.task)
                
        except Exception as e:
            print(f"Failed to initialize LLM control: {e}")

    def _initialize_agent_priorities(self):
        if hasattr(self.task, 'humanoid_handles') and self.task.humanoid_handles:
            self.agent_priorities['humanoid'] = AgentPriority.HUMAN
        
        if hasattr(self.task, 'franka_handles') and self.task.franka_handles:
            self.agent_priorities['franka'] = AgentPriority.FRANKA
        
        if hasattr(self.task, 'mobile_handles') and self.task.mobile_handles:
            for i, _ in enumerate(self.task.mobile_handles):
                self.agent_priorities[f'mobile_{i}'] = AgentPriority.MOBILE
    
    def _initialize_llm_system(self):
        if not LLM_AVAILABLE:
            print("LLM planning system not available")
            return
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    raw = json.load(f)
                if isinstance(raw, list) and raw:
                    env_config = raw[0]
                elif isinstance(raw, dict):
                    env_config = raw
                else:
                    env_config = {}
                
                self._create_llm_agents(env_config)
                self._create_planning_arena(env_config)
                print("LLM planning system initialized successfully")
            else:
                print(f"Config file not found: {self.config_path}")
                
        except Exception as e:
            print(f"Failed to initialize LLM system: {e}")
    
    def _create_llm_agents(self, env_config):
        agent_configs = [
            {
                'agent_id': 0,
                'args': self._create_mock_args(),
                'agent_node': {'id': 'humanoid', 'class_name': 'humanoid'},
                'init_graph': env_config.get('init_graph', {})
            },
            {
                'agent_id': 1,
                'args': self._create_mock_args(),
                'agent_node': {'id': 'franka', 'class_name': 'robot_arm'},
                'init_graph': env_config.get('init_graph', {})
            }
        ]
        
        for config in agent_configs:
            try:
                agent = FeedbackAgent(**config)
                self.llm_agents[config['agent_node']['id']] = agent
            except Exception as e:
                print(f"Failed to create agent {config['agent_node']['id']}: {e}")
    
    def _create_mock_args(self):
        class MockArgs:
            def __init__(self, task):
                self.task = task.cfg.get('task', 'HumanoidAMPCarryObjectObstacle')
                self.num_envs = task.cfg.get('num_envs', 1)
                self.cfg_env = task.cfg.get('cfg_env', 'coohoi/data/cfg/humanoid_carrybox.yaml')
                self.cfg_train = task.cfg.get('cfg_train', 'coohoi/data/cfg/train/amp_humanoid_task.yaml')
                self.motion_file = task.cfg.get('motion_file', 'coohoi/data/motions/coohoi_data/coohoi_data.yaml')
                self.checkpoint = task.cfg.get('checkpoint', 'coohoi/data/models/Humanoid.pth')
                
                self.debug = False
                self.source = 'llm_module'
                self.lm_id = 'gpt-4o-mini'
                self.max_tokens = 1000
                self.t = 0.7
                self.n = 1
                self.env = task.cfg.get('env', 'env0')
                self.api_key = None
                self.organization = None
                self.oracle_prompt_path = None
                self.agent_selection_prompt_path = None
                self.quadrotor_prompt_path = None
                self.robot_dog_prompt_path = None
                self.robot_arm_prompt_path = None
                self.judge_prompt_path = None
        
        return MockArgs()
    
    def _create_planning_arena(self, env_config):
        try:
            def env_fn():
                return self._create_mock_env_info(env_config)

            def agent_fn(args_llm):
                return FeedbackAgent(**args_llm)

            self.llm_arena = ArenaMultiAgent(
                environment_fn=env_fn,
                agent_fn=list(self.llm_agents.values()),
                args=self._create_mock_args()
            )
            
        except Exception as e:
            print(f"Failed to create planning arena: {e}")
    
    def _create_mock_env_info(self, env_config):
        class MockEnvInfo:
            def __init__(self, config):
                self.task_goal = config.get('task_goal', {})
                self.goal_instruction = config.get('goal_instruction', '')
                self.init_graph = config.get('init_graph', {})
        
        return MockEnvInfo(env_config)
    
    def start_planning(self, task_description: str = "请给出组装方案"):
        if not self.llm_arena:
            print("llm_arena planning system not available")
            return False
        
        try:
            self.planning_active = True
            self.plan_status = "planning"
            
            self.planning_thread = threading.Thread(
                target=self._run_planning,
                args=(task_description,)
            )
            self.planning_thread.start()
            
            print("LLM planning started in background")
            return True
            
        except Exception as e:
            print(f"Failed to start planning: {e}")
            self.planning_active = False
            self.plan_status = "idle"
            return False
    
    def _run_planning(self, task_description: str):
        try:
            self.current_plan = self._generate_example_plan()
            self.plan_execution_index = 0
            self.plan_status = "executing"
            
            print(f"Planning completed: {len(self.current_plan)} steps")
            
        except Exception as e:
            print(f"Planning failed: {e}")
            self.plan_status = "idle"
        finally:
            self.planning_active = False
    
###### distance -> status -> action ######
    # 更新agent状态
    def update_agent_states(self):
        try:
            env_ptr = self.task.envs[0]
            root_state = self.task.gym.acquire_actor_root_state_tensor(self.task.sim)
            all_root_state = gymtorch.wrap_tensor(root_state)
            
            current_time = time.time()

            if hasattr(self.task, 'humanoid_handles') and self.task.humanoid_handles:
                humanoid_idx = self.task.gym.get_actor_index(env_ptr, self.task.humanoid_handles[0], gymapi.DOMAIN_SIM)
                pos = all_root_state[humanoid_idx, 0:3].cpu().numpy()
                vel = all_root_state[humanoid_idx, 7:10].cpu().numpy()
                
                self.agent_states['humanoid'] = AgentState(
                    id='humanoid',
                    name='Humanoid Robot',
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
                    priority=AgentPriority.HUMAN,
                    last_update=current_time
                )
            
            if hasattr(self.task, 'franka_handles') and self.task.franka_handles:
                franka_idx = self.task.gym.get_actor_index(env_ptr, self.task.franka_handles[0], gymapi.DOMAIN_SIM)
                pos = all_root_state[franka_idx, 0:3].cpu().numpy()
                vel = all_root_state[franka_idx, 7:10].cpu().numpy()
                
                self.agent_states['franka'] = AgentState(
                    id='franka',
                    name='Franka Robot',
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
                    priority=AgentPriority.FRANKA,
                    last_update=current_time
                )
            
            if hasattr(self.task, 'mobile_handles') and self.task.mobile_handles:
                for i, handle in enumerate(self.task.mobile_handles):
                    idx = self.task.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
                    pos = all_root_state[idx, 0:3].cpu().numpy()
                    vel = all_root_state[idx, 7:10].cpu().numpy()
                    
                    self.agent_states[f'mobile_{i}'] = AgentState(
                        id=f'mobile_{i}',
                        name=f'Mobile Robot {i+1}',
                        position=(float(pos[0]), float(pos[1]), float(pos[2])),
                        velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
                        priority=AgentPriority.MOBILE,
                        last_update=current_time
                    )
            
        except Exception as e:
            print(f"Failed to update agent states: {e}")
    
    def check_collision_risks(self) -> List[CollisionRisk]:
        collision_risks = []
        
        try:
            agent_list = list(self.agent_states.values())
            
            for i, agent1 in enumerate(agent_list):
                for j, agent2 in enumerate(agent_list[i+1:], i+1):
                    pos1 = np.array(agent1.position[:2])  # only x, y
                    pos2 = np.array(agent2.position[:2])
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    risk_level = 'low'
                    if distance <= self.collision_thresholds['critical']:
                        risk_level = 'critical'
                    elif distance <= self.collision_thresholds['high']:
                        risk_level = 'high'
                    elif distance <= self.collision_thresholds['medium']:
                        risk_level = 'medium'
                    
                    if risk_level != 'low':
                        if agent1.priority.value > agent2.priority.value:
                            priority_agent = agent1.id
                            recommended_action = f"Stop {agent2.name}"
                        else:
                            priority_agent = agent2.id
                            recommended_action = f"Stop {agent1.name}"
                        
                        collision_risk = CollisionRisk(
                            agent1_id=agent1.id,
                            agent2_id=agent2.id,
                            distance=distance,
                            risk_level=risk_level,
                            recommended_action=recommended_action,
                            priority_agent=priority_agent
                        )
                        collision_risks.append(collision_risk)
            
        except Exception as e:
            print(f"Failed to check collision risks: {e}")
        
        return collision_risks
    
    def execute_safety_control(self, collision_risks: List[CollisionRisk]):
        if not self.safety_mode:
            return
        
        for k in list(self.mobile_robot_stop_flags.keys()):
            self.mobile_robot_stop_flags[k] = False
        
        for risk in collision_risks:
            if risk.risk_level in ['high', 'critical']:
                a1, a2 = risk.agent1_id, risk.agent2_id

                if a1.startswith('mobile') and a2.startswith('mobile'):
                    self._stop_mobile_robot(a1)
                    self._stop_mobile_robot(a2)
                elif a1.startswith('mobile') and not a2.startswith('mobile'):
                    self._stop_mobile_robot(a1)
                elif a2.startswith('mobile') and not a1.startswith('mobile'):
                    self._stop_mobile_robot(a2)
    
    def _execute_safety_action(self, risk: CollisionRisk):
        return
    
    def _stop_mobile_robot(self, robot_id: str):
        try:
            self.mobile_robot_stop_flags[robot_id] = True
        except Exception as e:
            print(f"Failed to stop {robot_id}: {e}")
    
    def execute_current_plan(self):
        if not self.current_plan or self.plan_status != "executing":
            return
        
        try:
            if self.plan_execution_index < len(self.current_plan):
                current_step = self.current_plan[self.plan_execution_index]
                
                if self._can_execute_step(current_step):
                    success = self._execute_plan_step(current_step)
                    if success:
                        self.plan_execution_index += 1
                        print(f"✅ Step {current_step['step']} completed")
                    else:
                        print(f"❌ Step {current_step['step']} failed")
                else:
                    print(f"⏸️  Step {current_step['step']} paused due to safety concerns")
            
            if self.plan_execution_index >= len(self.current_plan):
                self.plan_status = "completed"
                print("🎉 Plan execution completed!")
                
        except Exception as e:
            print(f"Failed to execute plan: {e}")
    
    def _can_execute_step(self, step: Dict) -> bool:
        try:
            collision_risks = self.check_collision_risks()
            high_risks = [r for r in collision_risks if r.risk_level in ['high', 'critical']]
            
            if high_risks:
                step_agent = step['agent']
                for risk in high_risks:
                    if step_agent in [risk.agent1_id, risk.agent2_id]:
                        return False
            
            return True
            
        except Exception as e:
            print(f"Failed to check step execution: {e}")
            return False
    
    def _execute_plan_step(self, step: Dict) -> bool:
        try:
            agent_id = step['agent']
            action = step['action']
            
            print(f"Executing: {agent_id} -> {action}")
            
            if agent_id.startswith('mobile'):
                return self._execute_mobile_robot_action(agent_id, action, step['target'])
            elif agent_id == 'franka':
                return self._execute_franka_action(action, step['target'])
            elif agent_id == 'humanoid':
                return self._execute_humanoid_action(action, step['target'])
            else:
                print(f"Unknown agent type: {agent_id}")
                return False
                
        except Exception as e:
            print(f"Failed to execute plan step: {e}")
            return False
    
    def _execute_mobile_robot_action(self, agent_id: str, action: str, target: str) -> bool:
        try:
            idx_map = {
                'mobile_0': ("<wheeled robot1> (202)", 'A'),
                'mobile_1': ("<wheeled robot2> (203)", 'B'),
                'mobile_2': ("<wheeled robot3> (204)", 'D'),
            }
            robot_name, default_area = idx_map.get(agent_id, (None, None))
            area = default_area
            if action.endswith('_A'):
                area = 'A'
            elif action.endswith('_B'):
                area = 'B'
            elif action.endswith('_D'):
                area = 'D'
            if robot_name and hasattr(self.task, 'explore_area') and area:
                self.task.explore_area(robot_name, area)
                return True
            return False
        except Exception as e:
            print(f"Failed to execute mobile robot action: {e}")
            return False
    
    def _execute_franka_action(self, action: str, target: str) -> bool:
        try:
            if action == 'pick_and_place':
                is_right = 'right' in (target or '').lower()
                is_left = 'left' in (target or '').lower()

                if hasattr(self.task, 'gripper_closed'):
                    self.task.gripper_closed = False
                if hasattr(self.task, 'franka_task_stage'):
                    self.task.franka_task_stage = 0
                if hasattr(self.task, 'franka_task_stage_1'):
                    self.task.franka_task_stage_1 = 0
                # select which FSM to run by counter flag used in env
                if is_right:
                    self.task.franka_counter = 2
                elif is_left:
                    # first wheel path uses default counter == 0
                    self.task.franka_counter = 0
                else:
                    # default to left sequence if unspecified
                    self.task.franka_counter = 0
                print(f"Franka FSM armed. counter={self.task.franka_counter}, target={target}")
                return True
            else:
                print(f"Unknown Franka action: {action}")
                return False
        except Exception as e:
            print(f"Failed to execute Franka action: {e}")
            return False
    
    def _execute_humanoid_action(self, action: str, target: str) -> bool:
        try:
            if action == 'carry_object':
                print(f"Humanoid executing carry: {target}")
                # 这里应该调用实际的人形机器人控制逻辑
                return True
            else:
                print(f"Unknown humanoid action: {action}")
                return False
                
        except Exception as e:
            print(f"Failed to execute humanoid action: {e}")
            return False
    
    def update(self, task_description: str = "请给出组装方案"):
        try:
            self.update_agent_states()
            
            collision_risks = self.check_collision_risks()
            self.execute_safety_control(collision_risks)
            
            if not self.current_plan and self.plan_status == "idle":
                self.start_planning(task_description)
            
            if self.plan_status == "executing":
                self.execute_current_plan()
            
            for agent_id, controller in self.llm_control_agents.items():
                if self.current_plan and self.plan_execution_index < len(self.current_plan):
                    current_step = self.current_plan[self.plan_execution_index]
                    if current_step['agent'] == agent_id:
                        controller.step()
            
            if collision_risks:
                print(f"⚠️  Collision risks detected: {len(collision_risks)}")
                for risk in collision_risks:
                    print(f"   {risk.agent1_id} <-> {risk.agent2_id}: {risk.risk_level} risk")
            
        except Exception as e:
            print(f"Failed to update integration system: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'planning_active': self.planning_active,
            'plan_status': self.plan_status,
            'plan_progress': f"{self.plan_execution_index}/{len(self.current_plan)}",
            'agent_count': len(self.agent_states),
            'safety_mode': self.safety_mode,
            'emergency_stop': self.emergency_stop
        }
    
    def set_safety_mode(self, enabled: bool):
        self.safety_mode = enabled
        print(f"Safety mode: {'enabled' if enabled else 'disabled'}")
    
    def set_collision_thresholds(self, thresholds: Dict[str, float]):
        self.collision_thresholds.update(thresholds)
        print("Collision thresholds updated")

