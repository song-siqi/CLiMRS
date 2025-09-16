import os
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_rotate
import requests
import json
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import re
import math
from scipy.spatial.transform import Rotation as R, Slerp
from controllers.franka_osc_controller import FrankaOSCController
from controllers.kinematics import FrankaIKGym
from rrt_algorithms.planpath import plan_paths_for_cars_and_boxes, plan_paths_for_boxes_to_franka_area

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from LLM.dev_revision.arena import ArenaMultiAgent
    from LLM.dev_revision.llm_agents.feedback_agent import FeedbackAgent
    from LLM.dev_revision.llm_agents.oracle_planner import OraclePlanner
    LLM_DEV_REVISION_AVAILABLE = True
except ImportError:
    LLM_DEV_REVISION_AVAILABLE = False
    from env.LLM_API.ask_Llm import LLMWorkflow
    from env.LLM_API.split_llm_with_skills import parse_workflow_text, SKILL_MAP
    from env.LLM_API.gym_llm_integration import GymLLMIntegration
    from env.LLM_API.gym_llm_planning_integration import GymLLMPlanningIntegration

# TODO:
# 单LLM做多决策
# self.component_handles[2] body
# self.component_handles[0] first wheel   self.component_handles[1] second wheel

class Pose:
    def __init__(self, pos, quat):
        self.pos = np.array(pos)
        self.quat = np.array(quat)

def slerp(q0, q1, t):
    key_rots = R.from_quat([q0, q1])
    slerp_obj = Slerp([0, 1], key_rots)
    return slerp_obj([t]).as_quat()[0]

def interpolate(start: Pose, end: Pose, step_size: float = None, num_steps: int = None):
    path = []
    for i in range(num_steps + 1):
        t = i / num_steps
        pos = (1 - t) * start.pos + t * end.pos
        quat = slerp(start.quat, end.quat, t)
        path.append(Pose(pos, quat))
    return path

def convert_wz(w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if not (w.shape == z.shape):
        raise ValueError("w 和 z 的形状必须相同")
    quat_x = torch.zeros_like(z)
    quat_y = torch.zeros_like(z)
    return torch.stack([quat_x, quat_y, z, w], dim=-1)


class LLMManager:
    def __init__(self, task_instance, enable_llm=True):
        self.task = task_instance
        self.enable_llm = enable_llm
        self.llm_update = 200
        self.llm_mode_active = False
        self.arena_multi_agent = None
        self.llm_planning_integration = None
        self.llm_integration = None
        if self.enable_llm:
            self._init_llm_systems()
    
    def _init_llm_systems(self):
        if LLM_DEV_REVISION_AVAILABLE:
            self.arena_multi_agent = self._init_arena_multi_agent()
        else:
            self.llm_planning_integration = GymLLMPlanningIntegration(self.task)
            self.llm_integration = GymLLMIntegration(self.task)
            if self.llm_integration:
                self.llm_integration.initialize()
    
    def _init_arena_multi_agent(self):
        from types import SimpleNamespace
        
        args = SimpleNamespace()
        args.debug = False
        args.source = 'llm_module'
        args.lm_id = 'gpt-4o-mini'
        args.max_tokens = 1000
        args.t = 0.7
        args.n = 1
        args.env = 'env0'
        args.api_key = None
        args.organization = None
        args.oracle_prompt_path = 'LLM/dev_revision/prompt/oracle_prompt.txt'
        args.agent_selection_prompt_path = 'LLM/dev_revision/prompt/agent_selection_prompt.txt'
        args.quadrotor_prompt_path = 'LLM/dev_revision/prompt/quadrotor_prompt.txt'
        args.mobile_car_prompt_path = 'LLM/dev_revision/prompt/mobile_car_prompt.txt'
        args.humanoid_prompt_path = 'LLM/dev_revision/prompt/humanoid_prompt.txt'
        args.robot_arm_prompt_path = 'LLM/dev_revision/prompt/robot_arm_prompt.txt'
        args.judge_prompt_path = 'LLM/dev_revision/prompt/judge_prompt.txt'
        args.select_agents = False
        
        def env_fn():
            task_ref = self.task  # 捕获真实任务的引用（LLMManager.task）  
            class MockEnv:
                def __init__(self):
                    self.task = task_ref  
                    self.task_goal = {"on_trunk(303)_left wheel(405)": [1], "on_trunk(303)_right wheel(406)": [1]}
                    self.goal_instruction = "Assemble robot components: attach left and right wheels to trunk"
                    self.steps = 0
                    self.task_id = 1
                    self.env_id = 0
                    self.task_name = "robot_assembly"
                    self.ground_truth_step_num = 10
                    self.num_agent = 5  # humanoid + franka + 3 mobile_cars
                    self.agent_states = {}  # 跟踪每个agent的状态
                    self.id_name_dict = {
                        0: ('humanoid', 101), 
                        1: ('franka', 606), 
                        2: ('mobile_car_1', 201),  # 处理left wheel
                        3: ('mobile_car_2', 202),  # 处理right wheel  
                        4: ('mobile_car_3', 203)   # 处理trunk
                    }
                    
                def get_observations(self):
                    try:
                        if hasattr(self, 'get_positions_for_prompt') and callable(self.get_positions_for_prompt):
                            area_positions, agent_positions = self.get_positions_for_prompt(0, self.envs[0])
                    except:
                        area_positions = {'A': (1.0, 2.0, 0.0), 'B': (-1.0, 2.0, 0.0), 'C': (-1.0, -2.0, 0.0), 'D': (1.0, -2.0, 0.0)}
                        agent_positions = {
                            "<humanoid> (101)": (0.0, 0.0), 
                            "<franka> (606)": (0.0, 1.0),
                            "<mobile_car_1> (201)": (1.0, 1.0),
                            "<mobile_car_2> (202)": (-1.0, 1.0),
                            "<mobile_car_3> (203)": (0.0, -1.0)
                        }

                    executed_actions = getattr(self, 'executed_actions', {})
                    
                    obs = []
                    for i in range(5): 
                        if i == 0:  # humanoid agent
                            available_actions = [
                                "[walk] <humanoid> (101) move to selected area",
                                "[carry] <humanoid> (101) carry <obstacles> (507)",
                                "[wait] <humanoid> (101) wait"
                            ]
                        elif i == 1:  # franka agent
                            available_actions = [
                                "[check] <franka> (606) check <trunk> (303)",
                                "[check] <franka> (606) check <left wheel> (405)",
                                "[check] <franka> (606) check <right wheel> (406)",
                                "[pick] <franka> (606) pick and place <left wheel> (405) on <trunk> (303)",
                                "[pick] <franka> (606) pick and place <right wheel> (406) on <trunk> (303)",
                                "[wait] <franka> (606) wait"
                            ]
                        elif i == 2:  # mobile_car_1 agent (left wheel)
                            agent_key = "mobile_car_1(201)"
                            if hasattr(self, 'task') and hasattr(self.task, 'agent_states'):
                                agent_state = self.task.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            else:
                                agent_state = self.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            
                            available_actions = []
                            if agent_state['status'] == 'idle':
                                available_actions.append("[move] <mobile_car_1> (201) move to component location using RRT path")
                            elif agent_state['status'] == 'moved':
                                available_actions.append("[push] <mobile_car_1> (201) push selected component to franka area")
                            
                            available_actions.append("[wait] <mobile_car_1> (201) wait")
                        elif i == 3:  # mobile_car_2 agent (right wheel)
                            agent_key = "mobile_car_2(202)"
                            if hasattr(self, 'task') and hasattr(self.task, 'agent_states'):
                                agent_state = self.task.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            else:
                                agent_state = self.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            
                            available_actions = []
                            if agent_state['status'] == 'idle':
                                available_actions.append("[move] <mobile_car_2> (202) move to component location using RRT path")
                            elif agent_state['status'] == 'moved':
                                available_actions.append("[push] <mobile_car_2> (202) push selected component to franka area")
                            
                            available_actions.append("[wait] <mobile_car_2> (202) wait")
                        elif i == 4:  # mobile_car_3 agent (trunk)
                            agent_key = "mobile_car_3(203)"
                            if hasattr(self, 'task') and hasattr(self.task, 'agent_states'):
                                agent_state = self.task.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            else:
                                agent_state = self.agent_states.get(agent_key, {'status': 'idle', 'last_action': None})
                            
                            available_actions = []
                            if agent_state['status'] == 'idle':
                                available_actions.append("[move] <mobile_car_3> (203) move to component location using RRT path")
                            elif agent_state['status'] == 'moved':
                                available_actions.append("[push] <mobile_car_3> (203) push selected component to franka area")
                            
                            available_actions.append("[wait] <mobile_car_3> (203) wait")
                        else:
                            available_actions = []
                            
                        agent_obs = {
                            "nodes": [
                                {"id": 101, "class_name": "humanoid", "category": "Agents", "properties": ["MOVABLE"], "states": []},
                                {"id": 606, "class_name": "robot arm", "category": "Agents", "properties": ["ON_HIGH_SURFACE"], "states": []},
                                {"id": 201, "class_name": "mobile_car_1", "category": "Agents", "properties": ["MOVABLE"], "states": []},
                                {"id": 202, "class_name": "mobile_car_2", "category": "Agents", "properties": ["MOVABLE"], "states": []},
                                {"id": 203, "class_name": "mobile_car_3", "category": "Agents", "properties": ["MOVABLE"], "states": []},
                                {"id": 303, "class_name": "trunk", "category": "Objects", "properties": ["GRABABLE"], "states": []},
                                {"id": 405, "class_name": "left wheel", "category": "Objects", "properties": ["GRABABLE"], "states": []},
                                {"id": 406, "class_name": "right wheel", "category": "Objects", "properties": ["GRABABLE"], "states": []}
                            ],
                            "edges": [],
                            "agent_in_room_id": 1,
                            "available_actions": available_actions
                        }
                        obs.append(agent_obs)
                    return obs
                    
                def step(self, class_name, agent_id, action, task_goal):
                    self.steps += 1
                    
                    if not hasattr(self, 'executed_actions'):
                        self.executed_actions = {}
                    if not hasattr(self, 'dialogue_history'):
                        self.dialogue_history = ""
                    if not hasattr(self, 'total_dialogue_history'):
                        self.total_dialogue_history = []
                    
                    agent_key = f"{class_name}({agent_id})"
                    
                    if agent_key not in self.executed_actions:
                        self.executed_actions[agent_key] = []
                    
                    self.executed_actions[agent_key].append(action)
                    
                    if agent_key not in self.agent_states:
                        self.agent_states[agent_key] = {'status': 'idle', 'last_action': None}
                    
                    if hasattr(self, 'task') and hasattr(self.task, 'agent_states'):
                        if "[move]" in action:
                            self.task.agent_states[agent_key] = {'status': 'moved', 'last_action': 'move'}
                            self.agent_states[agent_key] = {'status': 'moved', 'last_action': 'move'}
                        elif "[push]" in action:
                            self.task.agent_states[agent_key] = {'status': 'pushed', 'last_action': 'push'}
                            self.agent_states[agent_key] = {'status': 'pushed', 'last_action': 'push'}
                        elif "[observe]" in action:
                            self.task.agent_states[agent_key] = {'status': 'observed', 'last_action': 'observe'}
                            self.agent_states[agent_key] = {'status': 'observed', 'last_action': 'observe'}
                    else:                       
                        if "[move]" in action:
                            self.agent_states[agent_key] = {'status': 'moved', 'last_action': 'move'}
                        elif "[push]" in action:
                            self.agent_states[agent_key] = {'status': 'pushed', 'last_action': 'push'}
                        elif "[observe]" in action:
                            self.agent_states[agent_key] = {'status': 'observed', 'last_action': 'observe'}

                    task_results = []
                    satisfied = []
                    unsatisfied = []
                    
                    if "[move]" in action:
                        action_result = f"{agent_key} has successfully moved to component location and is ready for push action"
                        satisfied.append(action_result)
                        task_results.append({
                            'agent': agent_key,
                            'action': action, 
                            'status': 'completed',
                            'next_available': ['push', 'wait']
                        })
                        
                    elif "[push]" in action:
                        action_result = f"{agent_key} has successfully pushed component to franka assembly area"
                        satisfied.append(action_result)
                        task_results.append({
                            'agent': agent_key,
                            'action': action, 
                            'status': 'completed',
                            'next_available': ['wait']
                        })
                    
                    elif "[observe]" in action:
                        area_match = re.search(r'area <([^>]+)>', action)
                        if area_match:
                            area = area_match.group(1)
                            action_result = f"{agent_key} observed area {area} and found components"
                            satisfied.append(action_result)
                            task_results.append({
                                'agent': agent_key,
                                'action': action,
                                'status': 'completed', 
                                'next_available': ['move', 'wait']
                            })
                    
                    done = False
                    if len(self.executed_actions) >= 3:  
                        move_count = sum(1 for actions in self.executed_actions.values() 
                                       for action in actions if "[move]" in action)
                        if move_count >= 3:  
                            done = True
                    
                    return done, task_results, satisfied, unsatisfied, self.steps
                    
            env = MockEnv()
            env.envs = [None]
            env.get_positions_for_prompt = self.task.get_positions_for_prompt
            
            return env
        
        agent_configs = [
            {
                'agent_id': 0,
                'args': args,
                'agent_node': {'id': 101, 'class_name': 'humanoid'},
                'init_graph': {'nodes': [], 'edges': []}
            },
            {
                'agent_id': 1,
                'args': args, 
                'agent_node': {'id': 606, 'class_name': 'robot_arm'},
                'init_graph': {'nodes': [], 'edges': []}
            },
            {
                'agent_id': 2,
                'args': args,
                'agent_node': {'id': 201, 'class_name': 'mobile_car_1'}, 
                'init_graph': {'nodes': [], 'edges': []}
            },
            {
                'agent_id': 3,
                'args': args,
                'agent_node': {'id': 202, 'class_name': 'mobile_car_2'}, 
                'init_graph': {'nodes': [], 'edges': []}
            },
            {
                'agent_id': 4,
                'args': args,
                'agent_node': {'id': 203, 'class_name': 'mobile_car_3'}, 
                'init_graph': {'nodes': [], 'edges': []}
            }
        ]
        
        agents = [FeedbackAgent(**config) for config in agent_configs]
        
        arena = ArenaMultiAgent(env_fn, agents, args)
        return arena
    
    def update(self, step_count):
        if not self.enable_llm:
            return
        
        if step_count % self.llm_update == 0:
            if self.arena_multi_agent:
                if not self._any_robot_executing_waypoints():
                    action, message = self.run_llm_planning(step_count)
                    if action and action != "None":
                        self.llm_mode_active = True
                        self.task._execute_llm_action(action, message)
            elif self.llm_planning_integration:
                self.llm_planning_integration.update("请给出组装方案")
    
    def run_llm_planning(self, step_count):
        if self.arena_multi_agent:
            try:
                self.arena_multi_agent.dialogue_history = getattr(self.task, 'dialogue_history', "")
                self.arena_multi_agent.total_dialogue_history = getattr(self.task, 'total_dialogue_history', [])
                
                done, task_results, satisfied, unsatisfied, agent_id, agent_action, agent_message, steps = self.arena_multi_agent.step()
                
                self.task.dialogue_history = getattr(self.arena_multi_agent, 'dialogue_history', "")
                self.task.total_dialogue_history = getattr(self.arena_multi_agent, 'total_dialogue_history', [])
                
                print(f"Oracle output: {agent_message}")
                return agent_action, agent_message
            except Exception as e:
                print(f"LLM planning failed: {e}")
                return None, None
        return None, None

    def _any_robot_executing_waypoints(self):
        task = self.task
        # 检查move或push动作是否在执行
        if hasattr(task, 'llm_action_type') and task.llm_action_type in ["move", "push"] and hasattr(task, 'current_robot_id'):
            robot_id = task.current_robot_id
            simple_robot_id = robot_id - 200 if robot_id > 200 else robot_id
            
            if simple_robot_id == 1:
                waypoints = getattr(task, 'waypoints', None)
                current_idx = getattr(task, 'current_wp_idx', 0)
            elif simple_robot_id == 2:
                waypoints = getattr(task, 'waypoints_2', None)
                current_idx = getattr(task, 'current_wp_idx_2', 0)
            elif simple_robot_id == 3:
                waypoints = getattr(task, 'waypoints_3', None)
                current_idx = getattr(task, 'current_wp_idx_3', 0)
            else:
                return False

            if waypoints is not None and len(waypoints) > 0 and current_idx < len(waypoints):
                return True
        
        # 检查是否有push模式激活
        if hasattr(task, '_llm_push_mode') and getattr(task, '_llm_push_mode', False):
            return True
                
        return False
    
    def update_agent_states(self):
        if self.llm_planning_integration:
            self.llm_planning_integration.update_agent_states()
    
    def check_collision_risks(self):
        if self.llm_planning_integration:
            return self.llm_planning_integration.check_collision_risks()
        return []
    
    def execute_safety_control(self, collision_risks):
        if self.llm_planning_integration:
            self.llm_planning_integration.execute_safety_control(collision_risks)
    
    def get_mobile_robot_stop_flags(self):
        if self.llm_planning_integration:
            return getattr(self.llm_planning_integration, 'mobile_robot_stop_flags', {})
        return {}
    
    def get_plan_status(self):
        if self.arena_multi_agent:
            return 'executing' if self.llm_mode_active else 'idle'
        elif self.llm_planning_integration:
            return getattr(self.llm_planning_integration, 'plan_status', 'idle')
        return 'idle'
    
    def get_current_plan(self):
        if self.llm_planning_integration:
            return getattr(self.llm_planning_integration, 'current_plan', [])
        return []
    
    def get_plan_execution_index(self):
        if self.llm_planning_integration:
            return getattr(self.llm_planning_integration, 'plan_execution_index', 0)
        return 0

class HumanoidAMPCarryObjectObstacle(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.current_wp_idx = 0
        self.current_wp_idx_2 = 0
        self.current_wp_idx_3 = 0
        self.next_wp_idx = 0

        self.num_envs = cfg["env"]["numEnvs"]
        self.franka_counter = 0
        self.franka_count = 0
        self.absorbed = 0
        self.absorbed1 = 0

        self._box_dist_min = 0.5
        self._box_dist_max = 5
        self._target_dist_min = 1.5
        self._target_dist_max = 5

        # scaling object size
        self._box_min_scale = 1.0
        self._box_max_scale = 1.0
        self.scaling_factor = self._box_min_scale + \
            (self._box_max_scale - self._box_min_scale) * \
            torch.rand(self.num_envs)

        # scaling object weight
        self._box_min_weight = 0.8
        self._box_max_weight = 0.8
        self.scaling_factor_weight = self._box_min_weight + \
            (self._box_max_weight - self._box_min_weight) * \
            torch.rand(self.num_envs)

        self._default_box_width_size = 0.5
        self._default_box_length_size = 0.5
        self._default_box_height_size = 0.5

        self.obs_add_noise = True
        self.noise_level = 0.1
        
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._width_box_size = torch.zeros(self.num_envs).to(device)
        self._length_box_size = torch.zeros(self.num_envs).to(device)
        self._height_box_size = torch.zeros(self.num_envs).to(device)
        
        self.llm_manager = LLMManager(self, enable_llm=True)
        self.is_ask_llm = False

        self.target_position = torch.tensor([1.0, 7.5]).to(device)
        self.update_pos = torch.tensor([0.02,0.02]).to(device)

        self.controller = FrankaOSCController()
        self.ik_solver = FrankaIKGym()
        self.franka_task_stage = 0
        self.franka_task_stage_1 = 0
        self.franka_path = None
        self.franka_path_step = 0
        self.franka_gripper_target = None
        self.franka_gripper_steps = 0
        self.franka_gripper_max_steps = 0
        self.gripper_closed = False
        self.control_step_counter = 0
        self.wait_counter = 0

        self.current_area_positions = {}
        
        # 初始化目标ID相关属性
        self.current_target_id = None
        self.current_target_name = None
        self.llm_action_type = None
        
        # 初始化对话历史用于LLM状态上下文
        self.dialogue_history = ""
        self.total_dialogue_history = []
        
        # 添加agent状态管理 - 跟踪每个mobile car的状态
        self.agent_states = {
            'mobile_car_1(201)': {'status': 'idle', 'last_action': None},
            'mobile_car_2(202)': {'status': 'idle', 'last_action': None}, 
            'mobile_car_3(203)': {'status': 'idle', 'last_action': None}
        }
        
        # 初始化robot相关属性
        self.current_robot_name = None
        self.current_robot_id = None
        
        # 创建target_id到处理函数的映射
        self.target_id_handlers = {
            0: self._handle_component_0,  # wheel1
            1: self._handle_component_1,  # wheel2  
            2: self._handle_component_2,  # trunk/body
        }
        
        # 创建target_id到robot的映射 (三个mobile_car分别处理不同的component)
        self.target_id_to_robot_mapping = {
            0: 'robot1',    # left wheel (405) -> robot1 处理
            1: 'robot2',    # right wheel (406) -> robot2 处理  
            2: 'robot3',    # trunk (303) -> robot3 处理
        }
        
        # LLM robot名称到实际robot名称的映射
        self.llm_robot_name_mapping = {
            'mobile_car_1': 'robot1',    # mobile_car_1 (201) -> robot1
            'mobile_car_2': 'robot2',    # mobile_car_2 (202) -> robot2
            'mobile_car_3': 'robot3',    # mobile_car_3 (203) -> robot3
            'mobile_car': 'robot1',      # 向后兼容旧格式
            'humanoid': 'robot2',        # 如果有humanoid相关动作
            'franka': 'franka',          # franka保持原名
            'robot_arm': 'franka',       # robot_arm也是franka
        }
        

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.wait_steps = int(1.0 / self.dt)
        self.spacing = cfg["env"]['envSpacing']
        self.reset_time = 0
        self.log_success = False

        if cfg['env']['eval_mode']:
            # for calculate success rate and distance error and execution time
            self.log_success = True
            self._distance_to_target = [
                [] for _ in range(self.num_envs)]
            self.log_success_rate = []
            self.log_success_precision = []

        if cfg['env']['save_motions']:
            self.save_motion_for_blender = False
            self._save_all_state = True
            self.record_frame_number = 600

            self.output_dict = {}
            self.output_dict['trans'] = np.zeros(
                [self.record_frame_number, 15, 3])
            self.output_dict['rot'] = np.zeros(
                [self.record_frame_number, 15, 4])
            self.output_dict['obj_pos'] = np.zeros(
                [self.record_frame_number, 3])
            self.output_dict['obj_rot'] = np.zeros(
                [self.record_frame_number, 4])

            if self._save_all_state:
                self.output_dict['root_pos'] = np.zeros(
                    [self.record_frame_number, 3])
                self.output_dict['root_rot'] = np.zeros(
                    [self.record_frame_number, 4])
                self.output_dict['dof_pos'] = np.zeros(
                    [self.record_frame_number, 28])
        self.record_step = 0

        width_half_size = self._width_box_size / 2.0
        length_half_size = self._length_box_size / 2.0
        height_half_size = self._height_box_size / 2.0

        lfus = torch.stack([-length_half_size, width_half_size, height_half_size], dim=1)
        lfds = torch.stack([-length_half_size, width_half_size, -height_half_size], dim=1)
        lbus = torch.stack([-length_half_size, -width_half_size, height_half_size], dim=1)
        lbds = torch.stack([-length_half_size, -width_half_size, -height_half_size], dim=1)
        rfus = torch.stack([length_half_size, width_half_size, height_half_size], dim=1)
        rfds = torch.stack([length_half_size, width_half_size, -height_half_size], dim=1)
        rbus = torch.stack([length_half_size, -width_half_size, height_half_size], dim=1)
        rbds = torch.stack([length_half_size, -width_half_size, -height_half_size], dim=1)

        stand_points_left = torch.stack(
            [-length_half_size - 0.2, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        stand_points_right = torch.stack(
            [length_half_size + 0.2, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_left = torch.stack(
            [-length_half_size + width_half_size, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_right = torch.stack(
            [length_half_size - width_half_size, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)

        self.box_bps = torch.stack([lfus, lfds, lbus, lbds, rfus, rfds, rbus, rbds], dim=0)

        self.stand_held_points_offset = torch.stack([stand_points_left, stand_points_right, held_points_left, held_points_right], dim=0)

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        lift_body_names = cfg["env"]["liftBodyNames"]
        self._lift_body_ids = self._build_lift_body_ids_tensor(lift_body_names)

        self._build_box_tensors()
        self._build_target_state_tensors()
        self._reset_target([0])
        
    def _build_lift_body_ids_tensor(self, lift_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in lift_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._box_asset = []
        self.obstacle_asset = []
        self.obstacle_asset2 = []
        self.obstacle_asset3 = []
        self.obstacle_asset4 = []
        self.obs_asset1 = []
        self.obs_asset2 = []
        self.obs_asset3 = []
        self.obs_asset4 = []
        self.obs_box1 = []
        self.obs_box2 = []
        self.obs_box3 = []
        self.obs_component = []

        self._box_handles = []
        self.franka_handles = []
        self.franka_hand_indices = []
        self.table_handles = []
        self.franka_body_handles = []
        self.franka_cube_handles = []
        self.component_handles = []
        self.mobile_handles = []
        self.component_cube_handles = []
        self.franka_cylinder_rb_idxs = []
        self.franka_cube_size = 0.04
        
        self._load_box_asset()
        self._load_obstacle()

        super()._create_envs(num_envs, spacing, num_per_row)     
        return

# Build the actual box and the blocks corresponding to the parts
    def _load_box_asset(self):
        width_box_size = self._default_box_width_size
        length_box_size = self._default_box_length_size
        height_box_size = self._default_box_height_size
        self.asset_density = torch.zeros(self.num_envs).to(self.device)

        for env_id in range(self.num_envs):
            scaling_factor_l = self.scaling_factor[env_id]
            scaling_factor_w = self.scaling_factor[env_id]
            scaling_factor_h = self.scaling_factor[env_id]
            scaling_factor_weight = self.scaling_factor_weight[env_id]

            box_length = scaling_factor_l * length_box_size
            box_width = scaling_factor_w * width_box_size
            box_height = scaling_factor_h * height_box_size

            asset_options = gymapi.AssetOptions()
            asset_options.density = scaling_factor_weight * 50.0 / \
                (scaling_factor_l * scaling_factor_w * scaling_factor_h)
            self.asset_density[env_id] = asset_options.density

            self.obs_box1.append(self.gym.create_box(
                self.sim, box_length, box_width, box_height, asset_options))
            self.obs_box2.append(self.gym.create_box(
                self.sim, box_length, box_width, box_height, asset_options))
            self.obs_box3.append(self.gym.create_box(
                self.sim, box_length, box_width, box_height, asset_options))
        
            self.obs_component.append(self.gym.create_box(
                self.sim, 0.6, 0.5, box_height-0.2, asset_options))
            
        return

# build obstacle
    def _load_obstacle(self):
        width_box_size = self._default_box_width_size * 2
        length_box_size = self._default_box_length_size *2
        height_box_size = self._default_box_height_size

        for env_id in range(self.num_envs):
            box_length =  length_box_size
            box_width = width_box_size
            box_height = height_box_size

            asset_options = gymapi.AssetOptions()

            self.obstacle_asset.append(self.gym.create_box(
                self.sim, box_length, box_width*22, box_height, asset_options))
            self.obstacle_asset2.append(self.gym.create_box(
                self.sim, box_length, box_width*22, box_height, asset_options))
            self.obstacle_asset3.append(self.gym.create_box(
                self.sim, box_length*13, box_width, box_height, asset_options))
            self.obstacle_asset4.append(self.gym.create_box(
                self.sim, box_length*13, box_width, box_height, asset_options))
            
            self.obs_asset1.append(self.gym.create_box(
                self.sim, box_length*5.0, box_width*0.25, box_height, asset_options))
            self.obs_asset2.append(self.gym.create_box(
                self.sim, box_length*5.0, box_width*0.25, box_height, asset_options))
            self.obs_asset3.append(self.gym.create_box(
                self.sim, box_length*0.25, box_width*5.0, box_height, asset_options))
            self.obs_asset4.append(self.gym.create_box(
                self.sim, box_length*0.25, box_width*5.0, box_height, asset_options))

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_box(env_id, env_ptr)
        self._reset_components(env_id, env_ptr)
        self._build_franka(env_id, env_ptr)
        self._build_franka_table(env_id, env_ptr)
        self._build_franka_body(env_id, env_ptr)
        self._build_left_table(env_id, env_ptr)
        self._build_franka_cube(env_id, env_ptr)
        self._build_mobile_robots_cube(env_id, env_ptr)
        self._build_mobile_robots_cube2(env_id, env_ptr)
        self._build_mobile_robots_body(env_id, env_ptr)

        self._build_mobile_robots(env_id, env_ptr)   # wheel_1
        self._build_mobile_robots_2(env_id, env_ptr) # body
        self._build_mobile_robots_3(env_id, env_ptr) # wheel_2
        
        return

# Box position entity (should be aligned with the real object), add entity
    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 3.0

        default_pose2 = gymapi.Transform()
        default_pose2.p.x = -7.0
        default_pose2.p.y = 0.0
        default_pose2.p.z = 0.25

        default_pose3 = gymapi.Transform()
        default_pose3.p.x = 7.0
        default_pose3.p.y = 0.0
        default_pose3.p.z = 0.25

        default_pose4 = gymapi.Transform()
        default_pose4.p.x = 0.0
        default_pose4.p.y = 10.5
        default_pose4.p.z = 0.25

        default_pose5 = gymapi.Transform()
        default_pose5.p.x = 0.0
        default_pose5.p.y = -10.5
        default_pose5.p.z = 0.25

        default_pose6 = gymapi.Transform()
        default_pose6.p.x = 4.0
        default_pose6.p.y = 6.0
        default_pose6.p.z = 0.25

        default_pose7 = gymapi.Transform()
        default_pose7.p.x = -4.0
        default_pose7.p.y = -6.0
        default_pose7.p.z = 0.25

        default_pose8 = gymapi.Transform()
        default_pose8.p.x = -3.0
        default_pose8.p.y = 7.5
        default_pose8.p.z = 0.25

        default_pose9 = gymapi.Transform()
        default_pose9.p.x = 3.0
        default_pose9.p.y = -7.5
        default_pose9.p.z = 0.25

        default_pose10 = gymapi.Transform()
        default_pose10.p.x = 0.5
        default_pose10.p.y = 6.0
        default_pose10.p.z = 0.25

        default_pose11 = gymapi.Transform()
        default_pose11.p.x = -0.5
        default_pose11.p.y = 6.0
        default_pose11.p.z = 0.25

        default_pose12 = gymapi.Transform()
        default_pose12.p.x = -1.5
        default_pose12.p.y = 6.0
        default_pose12.p.z = 0.25

        # all 1.0
        scaling_factor_l = self.scaling_factor[env_id]
        scaling_factor_w = self.scaling_factor[env_id]
        scaling_factor_h = self.scaling_factor[env_id]

        self._width_box_size[env_id] = scaling_factor_w * \
            self._default_box_width_size
        self._length_box_size[env_id] = scaling_factor_l * \
            self._default_box_length_size
        self._height_box_size[env_id] = scaling_factor_h * \
            self._default_box_height_size

        obs_box_handle = self.gym.create_actor(
            env_ptr, self.obs_box1[env_id], default_pose10, "box", col_group, col_filter, segmentation_id)
        obs_box_handle2 = self.gym.create_actor(
            env_ptr, self.obs_box2[env_id], default_pose11, "box", col_group, col_filter, segmentation_id)
        obs_box_handle3 = self.gym.create_actor(
            env_ptr, self.obs_box3[env_id], default_pose12, "box", col_group, col_filter, segmentation_id)
        
        
        box_handle2 = self.gym.create_actor(
            env_ptr, self.obstacle_asset[env_id], default_pose2, "cube", col_group, col_filter, segmentation_id)
        box_handle3 = self.gym.create_actor(
            env_ptr, self.obstacle_asset2[env_id], default_pose3, "cube", col_group, col_filter, segmentation_id)
        box_handle4 = self.gym.create_actor(
            env_ptr, self.obstacle_asset3[env_id], default_pose4, "cube", col_group, col_filter, segmentation_id)
        box_handle5 = self.gym.create_actor(
            env_ptr, self.obstacle_asset4[env_id], default_pose5, "cube", col_group, col_filter, segmentation_id)
        box_handle6 = self.gym.create_actor(
            env_ptr, self.obs_asset1[env_id], default_pose6, "cube", col_group, col_filter, segmentation_id)
        box_handle7 = self.gym.create_actor(
            env_ptr, self.obs_asset2[env_id], default_pose7, "cube", col_group, col_filter, segmentation_id)
        box_handle8 = self.gym.create_actor(
            env_ptr, self.obs_asset3[env_id], default_pose8, "cube", col_group, col_filter, segmentation_id)
        box_handle9 = self.gym.create_actor(
            env_ptr, self.obs_asset4[env_id], default_pose9, "cube", col_group, col_filter, segmentation_id)

        props = self.gym.get_actor_dof_properties(env_ptr, obs_box_handle)
        props['friction'].fill(5.0)
        self.gym.set_actor_dof_properties(env_ptr, obs_box_handle, props)
        self._box_handles.append(obs_box_handle)
        self._box_handles.append(box_handle2)
        self._box_handles.append(box_handle3)
        self._box_handles.append(box_handle4)
        self._box_handles.append(box_handle5)
        self._box_handles.append(box_handle6)
        self._box_handles.append(box_handle7)
        self._box_handles.append(box_handle8)
        self._box_handles.append(box_handle9)
        self._box_handles.append(obs_box_handle2)
        self._box_handles.append(obs_box_handle3)

        return
    
    def _build_target_state_tensors(self):
        self._target_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self._target_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self.tar_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.tar_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        return

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()
        # now box states1
        self._box_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self.box_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self._box_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        self._box_pos = self._box_states[..., :3]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._box_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        
    def _build_box_tensors_2(self):
        num_actors = self.get_num_actors_per_env()
        # now box states1
        self._box_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        
        self.box_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self._box_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        self._box_pos = self._box_states[..., :3]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._box_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

    ################## agent new ################## 

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_box(env_ids)
        self._reset_cubes_mobile()
        self.rrt_plan()   
        return

    # box id
    def _reset_box(self, env_ids):
        rand_theta = 2 * np.pi *torch.tensor(0.25,device=self._box_states.device)
        axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
        rand_rot = quat_from_angle_axis(rand_theta, axis)
        self._box_states[env_ids, 3:7] = rand_rot
        self._box_states[env_ids, 7:] = 0.0
        return
   
    # The location of the parts (objects) that need to be moved
    def _reset_components(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.x = 4.0
        default_pose.p.y = 8.0
        default_pose.p.z = 0.25

        component_handle = self.gym.create_actor(
            env_ptr, self.obs_component[env_id], default_pose, "wheel_1", col_group, col_filter, segmentation_id)
        self.component_cube_handles.append(component_handle)    
        return

    def _reset_cubes_mobile(self):
        if hasattr(self, '_cubes_initialized') and self._cubes_initialized:
            return
        random.seed(65)
        corners = [(4, 8), (-4, 8), (-4, -8), (4, -8)]
        selected = random.sample(corners, 3)
        env_ptr = self.envs[0]

        cube_handles = [self.component_cube_handles[0], self.component_cube_handles[1], self.component_cube_handles[2]]
        mobile_handles = [self.mobile_handles[0], self.mobile_handles[1], self.mobile_handles[2]]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_state)

        for handle, pos in zip(cube_handles, selected):
            idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
            root_state[idx, 0] = pos[0]    # x
            root_state[idx, 1] = pos[1]    # y
            root_state[idx, 2] = 0.25      # z
        
        min_dist = 4.0  
        car_positions = []
        while len(car_positions) < 3:
            x = random.uniform(-5, -1) + random.uniform(0, 5)
            y = random.uniform(-3, 5)
            pos = (x, y)
            if all(math.sqrt((x - px)**2 + (y - py)**2) >= min_dist for px, py in car_positions):
                car_positions.append(pos)

        for handle, pos in zip(mobile_handles[1:], car_positions):
            idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
            root_state[idx, 0] = pos[0]
            root_state[idx, 1] = pos[1]
            root_state[idx, 2] = 0.25
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
        self._cubes_initialized = True

    def rrt_plan(self):
        env_ptr = self.envs[0]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_state)
        
        if self.llm_manager.enable_llm:
            self.llm_manager.update_agent_states()
            collision_risks = self.llm_manager.check_collision_risks()
            self.llm_manager.execute_safety_control(collision_risks)
            if any(self.llm_manager.get_mobile_robot_stop_flags().values()):
                print("⚠️  LLM系统检测到碰撞风险，部分小车已停止，跳过路径规划")
                return
        
        if not hasattr(self, '_waypoints_initialized') or not self._waypoints_initialized:
            car_handles = [self.mobile_handles[0], self.mobile_handles[1], self.mobile_handles[2]]
            box_handles = [self.component_cube_handles[0], self.component_cube_handles[1], self.component_cube_handles[2]]
            all_obstacle_handles = self._box_handles

            paths1 = plan_paths_for_cars_and_boxes(
                self.gym, self.sim, self.device, env_ptr, root_state, car_handles, box_handles, all_obstacle_handles
            )
            if not paths1 or any(p is None for p in paths1):
                raise RuntimeError("RRT planning failed")
            end_positions = [p[-1] for p in paths1]
            paths2 = plan_paths_for_boxes_to_franka_area(
                self.gym, self.sim, self.device, env_ptr, root_state, car_handles, box_handles, self.franka_handles[0], all_obstacle_handles, start_positions=end_positions
            )
            paths = []
            for p1, p2 in zip(paths1, paths2):
                if p1 is None: p1 = []
                if p2 is None: p2 = []
                paths.append(p1 + p2)
            self.waypoints = torch.tensor(paths1[0], device=self.device, dtype=torch.float32)
            self.waypoints_2 = torch.tensor(paths1[1], device=self.device, dtype=torch.float32)
            self.waypoints_3 = torch.tensor(paths1[2], device=self.device, dtype=torch.float32)
            self._waypoints_initialized = True
            self.rrt_paths = paths
            self.rrt_paths_initial = paths1  
            
            if self.llm_manager.enable_llm:
                print("✅ RRT路径规划完成，LLM系统已更新")

    # franka location entity
    def _build_franka(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.x = -0.05
        default_pose.p.y = -3.0
        default_pose.p.z = 0.0
        asset_options = gymapi.AssetOptions()

        asset_root = "Agent" 
        asset_file = "franka_description/robots/franka_panda.urdf"  
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True

        franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props['lower']
        franka_upper_limits = franka_dof_props['upper']
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)

        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        franka_dof_props["stiffness"][:7].fill(0.0)
        franka_dof_props["damping"][:7].fill(0.0)
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        default_dof_pos[:7] = franka_mids[:7]
        default_dof_pos[7:] = franka_upper_limits[7:]
        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        franka_handle = self.gym.create_actor(env_ptr, franka_asset, default_pose, "franka_panda", col_group, col_filter, segmentation_id)
        self.franka_handles.append(franka_handle)

        self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_handle, default_dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(env_ptr, franka_handle, default_dof_pos)

        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]
        self.franka_hand_indices.append(self.franka_hand_index)
        self.franka_dof_props = franka_dof_props
        self.franka_default_dof_pos = default_dof_pos
        self.franka_default_dof_state = default_dof_state
        return

    # robot desk(skip)
    def _build_franka_table(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        table_dims = gymapi.Vec3(0.6, 1.2, 0.3)
        default_pose.p.x = 0.45
        default_pose.p.y = -30.4
        default_pose.p.z = 0.5 * table_dims.z
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_handle = self.gym.create_actor(env_ptr, table_asset, default_pose, "table", col_group, col_filter, segmentation_id)
        self.table_handles.append(table_handle)
        return
    
    def _build_franka_body(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        table_dims = gymapi.Vec3(0.6, 1.2, 0.3)
        self.franka_cube_size = 0.45
        default_pose.p.x = 0.45
        default_pose.p.y = -3.0
        default_pose.p.z = table_dims.z - 0.1
        default_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = True

        asset_root = "Agent/Component_asset/2wheels_urdf"
        asset_file = "body.urdf"
        body_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        body_handle = self.gym.create_actor(env_ptr, body_asset, default_pose, "wheel_body", col_group, col_filter, segmentation_id)
        body_rb_idx = self.gym.get_actor_rigid_body_index(env_ptr, body_handle, 0, gymapi.DOMAIN_SIM)
        self.franka_body_handles.append(body_handle)
        
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, body_handle)
        for p in props:
            p.friction = 2.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, body_handle, props)

        return

    # robot desk(skip)
    def _build_left_table(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        table_dims = gymapi.Vec3(0.5, 1.7, 0.3)
        default_pose.p.x = -50
        default_pose.p.y = -1.4
        default_pose.p.z = 0.5 * table_dims.z
        default_pose.r = gymapi.Quat.from_euler_zyx(0, 0, np.pi/2)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        table_handle = self.gym.create_actor(env_ptr, table_asset, default_pose, "left_table", col_group, col_filter, segmentation_id)
        self.table_handles.append(table_handle)
        return

    # robot desk(skip)
    def _build_franka_cube(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        table_dims = gymapi.Vec3(0.5, 1.7, 0.3)
        self.franka_cube_size = 0.04
        default_pose.p.x = -50
        default_pose.p.y = -1.4
        default_pose.p.z = table_dims.z + 0.5 * self.franka_cube_size
        default_pose.r = gymapi.Quat.from_euler_zyx(0, 0, -np.pi)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False

        asset_root = "Agent/Component_asset/2wheels_urdf"
        asset_file = "wheel_only.urdf"
        wheel_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        wheel_handle = self.gym.create_actor(env_ptr, wheel_asset, default_pose, "franka_wheel", col_group, col_filter, segmentation_id)
        wheel_rb_idx = self.gym.get_actor_rigid_body_index(env_ptr, wheel_handle, 0, gymapi.DOMAIN_SIM)
        self.franka_cube_handles.append(wheel_handle)
        
        cylinder_rb_idx = self.gym.find_actor_rigid_body_handle(env_ptr, wheel_handle, "cylinder")
        self.franka_cylinder_rb_idxs.append(cylinder_rb_idx)
        
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, wheel_handle)
        for p in props:
            p.friction = 2.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, wheel_handle, props)

        return
    
    # Create the blocks for the robot's torso and wheels
    def _build_mobile_robots_body(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        self.franka_cube_size = 0.5
        default_pose.p.x = 4.0
        default_pose.p.y = -8.5
        default_pose.p.z = 1.0 

        default_pose2 = gymapi.Transform()
        default_pose2.p.x = -4.0
        default_pose2.p.y = -8.0
        default_pose2.p.z = 0.25

        default_pose3 = gymapi.Transform()
        default_pose3.p.x = 4.0
        default_pose3.p.y = -8.0
        default_pose3.p.z = 0.25

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False

        asset_root = "Agent/Component_asset/Omni.SLDASM"
        asset_file = "Omni.urdf"
        wheel_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.wheel2 = self.gym.create_box(
                self.sim, 0.6, 0.45, 0.3, asset_options)
        self.body_cube = self.gym.create_box(
                self.sim, 0.6, 0.3, 0.3, asset_options)
        
        component_cube_handle = self.gym.create_actor(
            env_ptr, self.wheel2, default_pose2, "wheel_2", col_group, col_filter, segmentation_id)
        component_cube_handle2 = self.gym.create_actor(
            env_ptr, self.body_cube, default_pose3, "robot_body", col_group, col_filter, segmentation_id)
        
        wheel_handle = self.gym.create_actor(env_ptr, wheel_asset, default_pose, "franka_body", col_group, col_filter, segmentation_id)

        self.component_handles.append(wheel_handle)
        self.component_cube_handles.append(component_cube_handle)
        self.component_cube_handles.append(component_cube_handle2)
        
        cylinder_rb_idx = self.gym.find_actor_rigid_body_handle(env_ptr, wheel_handle, "cylinder")
        self.franka_cylinder_rb_idxs.append(cylinder_rb_idx)
        
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, wheel_handle)
        for p in props:
            p.friction = 2.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, wheel_handle, props)

        return

    # Create the robot's second wheel
    def _build_mobile_robots_cube2(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        self.franka_cube_size = 0.04
        default_pose.p.x = -4.0
        default_pose.p.y = -8.0
        default_pose.p.z = 0.5
        default_pose.r = gymapi.Quat.from_euler_zyx(0, 0, -np.pi)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = False

        asset_root = "Agent/Component_asset/2wheels_urdf"
        asset_file = "wheel_only.urdf"
        wheel_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        wheel_handle = self.gym.create_actor(env_ptr, wheel_asset, default_pose, "franka_wheel2", col_group, col_filter, segmentation_id)
        self.component_handles.append(wheel_handle)
        
        cylinder_rb_idx = self.gym.find_actor_rigid_body_handle(env_ptr, wheel_handle, "cylinder")
        self.franka_cylinder_rb_idxs.append(cylinder_rb_idx)
        
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, wheel_handle)
        for p in props:
            p.friction = 2.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, wheel_handle, props)

        return
    
    # Creating the robot's first wheel
    def _build_mobile_robots_cube(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        table_dims = gymapi.Vec3(0.5, 1.7, 0.5)
        self.franka_cube_size = 0.04
        default_pose.p.x = 4.0
        default_pose.p.y = 8.0
        default_pose.p.z = table_dims.z 
        default_pose.r = gymapi.Quat.from_euler_zyx(0, 0, -np.pi)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = False

        asset_root = "Agent/Component_asset/2wheels_urdf"
        asset_file = "wheel_only.urdf"
        wheel_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        wheel_handle = self.gym.create_actor(env_ptr, wheel_asset, default_pose, "franka_wheel", col_group, col_filter, segmentation_id)
        self.component_handles.append(wheel_handle)
        
        cylinder_rb_idx = self.gym.find_actor_rigid_body_handle(env_ptr, wheel_handle, "cylinder")
        self.franka_cylinder_rb_idxs.append(cylinder_rb_idx)
        
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, wheel_handle)
        for p in props:
            p.friction = 2.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, wheel_handle, props)
        return
    
    def prepare_tensors(self):
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)

        franka_dof_start = self.gym.get_actor_dof_index(self.envs[0], self.franka_handles[0], 0, gymapi.DOMAIN_SIM)
        franka_dof_end = franka_dof_start + self.gym.get_actor_dof_count(self.envs[0], self.franka_handles[0])
        self.franka_dof_states = self.dof_states[franka_dof_start:franka_dof_end]
        
        self.franka_dof_pos = self.franka_dof_states[:, 0]
        self.franka_dof_vel = self.franka_dof_states[:, 1]

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _rb_states = gymtorch.wrap_tensor(_rb_states)
        self.franka_rb_states = _rb_states[franka_dof_start:franka_dof_end]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka_panda")
        self.franka_jacobian = gymtorch.wrap_tensor(_jacobian)
        self.j_eef = self.franka_jacobian[:, self.franka_hand_index - 1, :]

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka_panda")
        self.mm = gymtorch.wrap_tensor(_massmatrix)

        self.franka_hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)
        self.franka_pos_action = torch.zeros_like(self.franka_dof_pos).squeeze(-1)
        self.franka_effort_action = torch.zeros_like(self.franka_pos_action)

        return self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm
        
    # define take and place   
    def close_gripper(self):
        if self.franka_gripper_target is not None and self.franka_gripper_steps < self.franka_gripper_max_steps:
            all_tensor = self.pd_tar[0]
            for i, env in enumerate(self.envs):
                franka_dof_start = self.gym.get_actor_dof_index(env, self.franka_handles[i], 0, gymapi.DOMAIN_SIM)
                all_tensor[franka_dof_start+7] = self.franka_gripper_target
                all_tensor[franka_dof_start+8] = self.franka_gripper_target
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_tensor))

    def path_follow(self, pose):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()

        pos_cur = self.franka_rb_states[self.franka_hand_index, :3].unsqueeze(0)
        orn_cur = self.franka_rb_states[self.franka_hand_index, 3:7].unsqueeze(0)
        dof_vel = self.franka_dof_states[:, 1].reshape(len(self.envs), 9, 1)
        pos_des = torch.from_numpy(pose.pos.astype(np.float32)).unsqueeze(0).to(self.device)
        orn_des = torch.from_numpy(pose.quat.astype(np.float32)).unsqueeze(0).to(self.device)

        u, gripper_targets = self.controller.solve(
            pos_cur, orn_cur, dof_vel, pos_des, orn_des,
            self.j_eef, self.mm, cube_pos=None
        )
        u[:, 7:] = 0
        if self.gripper_closed:
            gripper_targets[:] = 0.01  
        else:
            gripper_targets[:] = 0.04

        total_dofs = self.dof_states.shape[0]
        all_u = torch.zeros(total_dofs, device=u.device, dtype=u.dtype)
        franka_dof_start = self.gym.get_actor_dof_index(self.envs[0], self.franka_handles[0], 0, gymapi.DOMAIN_SIM)
        franka_dof_end = franka_dof_start + self.gym.get_actor_dof_count(self.envs[0], self.franka_handles[0])
        all_u[franka_dof_start:franka_dof_end] = u[0]  
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_u))
        all_tensor = self.pd_tar[0]
        for i, env in enumerate(self.envs):
            franka_dof_start = self.gym.get_actor_dof_index(env, self.franka_handles[i], 0, gymapi.DOMAIN_SIM)
            all_tensor[franka_dof_start+7] = gripper_targets[i].item()
            all_tensor[franka_dof_start+8] = gripper_targets[i].item()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_tensor))
        
    def set_franka_path(self, path , duration=150.0):
        self.franka_path = path
        self.franka_path_time = 0.0
        self.franka_path_num = len(path)
        self.franka_path_duration = duration  

    def _step_franka_path(self, step_scale = 65 , gap = 5.0 , dist_gap = 0.02):
        if hasattr(self, 'franka_path') and self.franka_path_time < self.franka_path_duration:
            idx = int(self.franka_path_time * (self.franka_path_num - 1) / self.franka_path_duration)
            idx = min(idx, self.franka_path_num - 1)
            pose = self.franka_path[idx]
            self.path_follow(pose)
            self.franka_path_time += self.dt * step_scale
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            cur_pos = self.franka_rb_states[self.franka_hand_index, :3].cpu().numpy()
            target_pos = self.franka_path[-1].pos
            dist = np.linalg.norm(cur_pos - target_pos)
            weight = np.array([1.0, 1.0, 0.2])
            dist = np.linalg.norm((cur_pos - target_pos) * weight)
            if dist < dist_gap:
                return True
            if self.franka_path_time >= self.franka_path_duration:
                self.franka_path_duration += gap  
            return False
        return True
            
    def _step_franka_gripper(self, close=True):
        if close:
            self.franka_gripper_target = 0.01
        else:
            self.franka_gripper_target = 0.04
        if self.franka_gripper_steps < 5:
            self.close_gripper()
            self.franka_gripper_steps += 1
            return False
        else:
            self.franka_gripper_steps = 0
            return True

    def apply_magnetic_force(self,range = 0.2863 , mag = False):
        if not self.franka_cube_handles or not self.franka_body_handles:
            return
        env_ptr = self.envs[0]
        wheel_handle = self.component_handles[0]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        cube_root_idx = self.gym.get_actor_index(self.envs[0], wheel_handle, gymapi.DOMAIN_SIM)
        cube_state_tensor = all_root_state[cube_root_idx]
        body_rb_idx = self.gym.get_actor_index(self.envs[0], body_handle, gymapi.DOMAIN_SIM)
        body_state_tensor = all_root_state[body_rb_idx]
        wheel_pos = cube_state_tensor[0:3].unsqueeze(0).cpu().numpy()
        body_pos = body_state_tensor[0:3].unsqueeze(0).cpu().numpy()+ np.array([0.0, 0.165, 0.0]) 
        diff = body_pos - wheel_pos
        dist = np.linalg.norm(diff)
        magnet_range = range
        if mag:
            root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_state = gymtorch.wrap_tensor(root_state)
            wheel_root_idx = self.gym.get_actor_index(env_ptr, wheel_handle, gymapi.DOMAIN_SIM)
            root_state[wheel_root_idx, 0:3] = torch.tensor(body_pos, device=root_state.device, dtype=root_state.dtype)
            cube_quat = np.array([0.707, 0.0, 0.0, 0.707])
            root_state[wheel_root_idx, 3:7] = torch.tensor(cube_quat, device=root_state.device, dtype=root_state.dtype)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
            return
        if dist < magnet_range:
            root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_state = gymtorch.wrap_tensor(root_state)
            wheel_root_idx = self.gym.get_actor_index(env_ptr, wheel_handle, gymapi.DOMAIN_SIM)
            
            root_state[wheel_root_idx, 0:3] = torch.tensor(body_pos, device=root_state.device, dtype=root_state.dtype)
            cube_quat = np.array([-0.707, 0.0, 0.0, 0.707])
            root_state[wheel_root_idx, 3:7] = torch.tensor(cube_quat, device=root_state.device, dtype=root_state.dtype)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
            return True
        else:
            return False

    def apply_magnetic_force_1(self, range = 0.2863 , mag = False):
        if not self.franka_cube_handles or not self.franka_body_handles:
            return
        env_ptr = self.envs[0]
        wheel_handle = self.component_handles[1]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        cube_root_idx = self.gym.get_actor_index(self.envs[0], wheel_handle, gymapi.DOMAIN_SIM)
        cube_state_tensor = all_root_state[cube_root_idx]
        body_rb_idx = self.gym.get_actor_index(self.envs[0], body_handle, gymapi.DOMAIN_SIM)
        body_state_tensor = all_root_state[body_rb_idx]
        wheel_pos = cube_state_tensor[0:3].unsqueeze(0).cpu().numpy() 
        body_pos = body_state_tensor[0:3].unsqueeze(0).cpu().numpy()+ np.array([0.0, -0.18, 0.0]) 
        diff = body_pos - wheel_pos
        dist = np.linalg.norm(diff)
        magnet_range = range

        if mag:
            root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_state = gymtorch.wrap_tensor(root_state)
            wheel_root_idx = self.gym.get_actor_index(env_ptr, wheel_handle, gymapi.DOMAIN_SIM)
            root_state[wheel_root_idx, 0:3] = torch.tensor(body_pos, device=root_state.device, dtype=root_state.dtype)
            cube_quat = np.array([-0.707, 0.0, 0.0, 0.707])
            root_state[wheel_root_idx, 3:7] = torch.tensor(cube_quat, device=root_state.device, dtype=root_state.dtype)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
            return 

        if dist < magnet_range:
            root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_state = gymtorch.wrap_tensor(root_state)
            wheel_root_idx = self.gym.get_actor_index(env_ptr, wheel_handle, gymapi.DOMAIN_SIM)
            
            root_state[wheel_root_idx, 0:3] = torch.tensor(body_pos, device=root_state.device, dtype=root_state.dtype)
            cube_quat = np.array([0.707, 0.0, 0.0, 0.707])
            root_state[wheel_root_idx, 3:7] = torch.tensor(cube_quat, device=root_state.device, dtype=root_state.dtype)

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
            return True
        else:
            return False
    
    # path planning
    def _plan_franka_path_to_pre_grasp(self, cube_handle):
        print("Calling _plan_franka_path_to_pre_grasp")
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        cube_root_idx = self.gym.get_actor_index(self.envs[0], cube_handle, gymapi.DOMAIN_SIM)
        cube_state_tensor = all_root_state[cube_root_idx]

        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long,device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
        
        self.franka_init_pos = cur_pos[0].cpu().numpy().copy()
        self.franka_init_quat = cur_orn[0].cpu().numpy().copy()

        cube_pos = cube_state_tensor[0:3].unsqueeze(0)
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])

        pre_cube_pos = cube_pos + torch.tensor([0.0, -0.04, 0.15], device=cube_pos.device, dtype=cube_pos.dtype)
        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(pre_cube_pos[0].cpu().numpy(), cube_quat)
        
        path = interpolate(start_pose, end_pose, num_steps=50)
        self.set_franka_path(path, duration=50.0)  

    def _plan_franka_path_to_grasp(self, cube_handle):
        print("Calling _plan_franka_path_to_grasp")
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        # cube_handle = self.component_handles[0]
        cube_root_idx = self.gym.get_actor_index(self.envs[0], cube_handle, gymapi.DOMAIN_SIM)
        cube_state_tensor = all_root_state[cube_root_idx]
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long,device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
        cube_pos = cube_state_tensor[0:3].unsqueeze(0)
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube_pos = cube_pos + torch.tensor([0.0, -0.04, 0.02], device=cube_pos.device, dtype=cube_pos.dtype)
        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(cube_pos[0].cpu().numpy(), cube_quat)
        path = interpolate(start_pose, end_pose, num_steps=10)
        self.set_franka_path(path, duration=2.0)  

    def _plan_franka_path_to_pre_place(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
        env_ptr = self.envs[0]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        body_root_idx = self.gym.get_actor_index(env_ptr, body_handle, gymapi.DOMAIN_SIM)

        cube_quat = np.array([0.707, 0.0, 0.0, 0.707]) # left
        place_pos = all_root_state[body_root_idx, 0:3] 
        pre_place_pos = place_pos + torch.tensor([-0.0, 0.25, 0.2], device=self.device)  

        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(pre_place_pos.cpu().numpy(), cube_quat)
        path = interpolate(start_pose, end_pose, num_steps=120)
        self.set_franka_path(path, duration=200.0)

    def _plan_franka_path_to_place(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
 
        env_ptr = self.envs[0]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        body_root_idx = self.gym.get_actor_index(env_ptr, body_handle, gymapi.DOMAIN_SIM)
        place_pos = all_root_state[body_root_idx, 0:3]+torch.tensor([0.0, 0.146, 0.0], device=self.device)
        cube_quat = np.array([0.707, 0.0, 0.0, 0.707])
        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(place_pos.cpu().numpy(), cube_quat)
        path = interpolate(start_pose, end_pose, num_steps=20)
        self.set_franka_path(path, duration=30.0)

    def _plan_franka_path_to_lift(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
        lift_pos = self.franka_init_pos
        lift_orn = self.franka_init_quat 

        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(lift_pos, lift_orn)
        path = interpolate(start_pose, end_pose, num_steps=20)  
        self.set_franka_path(path,duration=20.0)

    def _franka_take_and_place_fsm(self, cube_handle):
        # print("FSM stage:", self.franka_task_stage)
        if self.franka_task_stage == 0:
            self._plan_franka_path_to_pre_grasp(cube_handle)
            self.franka_task_stage = 1
        elif self.franka_task_stage == 1:
            finished = self._step_franka_path()
            if finished:
                self.franka_task_stage = 2
        elif self.franka_task_stage == 2:
            self._plan_franka_path_to_grasp(cube_handle)
            self.franka_task_stage = 3
        elif self.franka_task_stage == 3:
            finished = self._step_franka_path(step_scale = 1, gap = 0.5, dist_gap = 0.0105)
            if finished:
                self.franka_task_stage = 4
        elif self.franka_task_stage == 4:
            finished = self._step_franka_gripper(close=True)
            if finished:
                self.gripper_closed = True
                self.franka_task_stage = 4.5
        elif self.franka_task_stage == 4.5:
                self._plan_franka_path_to_pre_place()
                self.franka_task_stage = 5
        elif self.franka_task_stage == 5:
            finished = self._step_franka_path(dist_gap = 0.0159)
            if finished:
                self.franka_task_stage = 6 
        elif self.franka_task_stage == 6:
            self._plan_franka_path_to_place()
            self.franka_task_stage = 7
        elif self.franka_task_stage == 7:
            absorbed = self.apply_magnetic_force()
            finished = self._step_franka_path(step_scale = 1, gap = 0.5, dist_gap = 0.01)
            if absorbed or finished:
                self.franka_task_stage = 8
                self.absorbed = 1
        elif self.franka_task_stage == 8:
            finished = self._step_franka_gripper(close=False)
            if finished:
                self.gripper_closed = False
                self.franka_task_stage = 9
        elif self.franka_task_stage == 9:
            self._plan_franka_path_to_lift()
            self.franka_task_stage = 10
        elif self.franka_task_stage == 10:
            finished = self._step_franka_path()
            if finished:
                self.franka_count = 1
                if hasattr(self, 'franka_path') and self.franka_path:
                    pose = self.franka_path[-1]
                    self.path_follow(pose)               
    
    # path planning
    def plan_franka_path_to_pre_place(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
        env_ptr = self.envs[0]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        body_root_idx = self.gym.get_actor_index(env_ptr, body_handle, gymapi.DOMAIN_SIM)

        cube_quat = np.array([-0.707, 0.0, 0.0, 0.707]) #右
        place_pos = all_root_state[body_root_idx, 0:3] 
        pre_place_pos = place_pos + torch.tensor([-0.0, -0.25, 0.2], device=self.device)  

        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(pre_place_pos.cpu().numpy(), cube_quat)
        path = interpolate(start_pose, end_pose, num_steps=120)
        self.set_franka_path(path, duration=300.0)

    def plan_franka_path_to_place(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]
 
        env_ptr = self.envs[0]
        body_handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        body_root_idx = self.gym.get_actor_index(env_ptr, body_handle, gymapi.DOMAIN_SIM)
        place_pos = all_root_state[body_root_idx, 0:3]+torch.tensor([0.0, -0.146, 0.0], device=self.device)
 
        cube_quat = np.array([-0.707, 0.0, 0.0, 0.707])
        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(place_pos.cpu().numpy(), cube_quat)
        path = interpolate(start_pose, end_pose, num_steps=20)
        self.set_franka_path(path, duration=30.0)

    def plan_franka_path_to_lift(self):
        self.franka_dof_states, self.franka_rb_states, self.j_eef, self.mm = self.prepare_tensors()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_idxs = torch.tensor(self.franka_hand_indices, dtype=torch.long, device=self.device)
        cur_pos = self.franka_rb_states[hand_idxs, :3]
        cur_orn = self.franka_rb_states[hand_idxs, 3:7]

        lift_pos = self.franka_init_pos
        lift_orn = self.franka_init_quat 

        start_pose = Pose(cur_pos[0].cpu().numpy(), cur_orn[0].cpu().numpy())
        end_pose = Pose(lift_pos, lift_orn)
        path = interpolate(start_pose, end_pose, num_steps=20)  
        self.set_franka_path(path,duration=20.0)
                
    def _franka_take_and_place_fsm2(self, cube_handle):
        print("FSM stage:", self.franka_task_stage_1)
        if self.franka_task_stage_1 == 0:
            self._plan_franka_path_to_pre_grasp(cube_handle)
            self.franka_task_stage_1 = 1
        elif self.franka_task_stage_1 == 1:
            finished = self._step_franka_path()
            if finished:
                self.franka_task_stage_1 = 2
        elif self.franka_task_stage_1 == 2:
            self._plan_franka_path_to_grasp(cube_handle)
            self.franka_task_stage_1 = 3
            self.franka_task_stage = 3
        elif self.franka_task_stage_1 == 3:
            finished = self._step_franka_path(step_scale = 1, gap = 0.5, dist_gap = 0.0105)
            if finished:
                self.franka_task_stage_1 = 4
        elif self.franka_task_stage_1 == 4:
            finished = self._step_franka_gripper(close=True)
            if finished:
                self.gripper_closed = True
                self.franka_task_stage_1 = 4.5
        elif self.franka_task_stage_1 == 4.5:
                self.plan_franka_path_to_pre_place()
                self.franka_task_stage_1 = 5
        elif self.franka_task_stage_1 == 5:
            finished = self._step_franka_path(dist_gap = 0.0159)
            if finished:
                self.franka_task_stage_1 = 6 
        elif self.franka_task_stage_1 == 6:
            self.plan_franka_path_to_place()
            self.franka_task_stage_1 = 7
        elif self.franka_task_stage_1 == 7:
            absorbed = self.apply_magnetic_force_1()
            finished = self._step_franka_path(step_scale = 1, gap = 0.5, dist_gap = 0.01)
            if absorbed or finished:
                self.franka_task_stage_1 = 8
                self.absorbed1 = 1
        elif self.franka_task_stage_1 == 8:
            finished = self._step_franka_gripper(close=False)
            if finished:
                self.gripper_closed = False
                self.franka_task_stage_1 = 9
        elif self.franka_task_stage_1 == 9:
            self.plan_franka_path_to_lift()
            self.franka_task_stage_1 = 10
        elif self.franka_task_stage_1 == 10:
            finished = self._step_franka_path()
            if finished:
                self.franka_task_stage_1 = 0
                import pdb; pdb.set_trace() 

    ## mobile robots ##
    def keep_cube_attached_to_box(self, cube_handle, box_handle):
        # Magnetic logic, component_handles is the component, box_handles is the corresponding box.
        if not self.component_handles or not self._box_handles:
            return   
        env_ptr = self.envs[0]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_state)

        cube_root_idx = self.gym.get_actor_index(env_ptr, cube_handle, gymapi.DOMAIN_SIM)
        box_root_idx = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
        box_pos = root_state[box_root_idx, 0:3]
        box_quat = root_state[box_root_idx, 3:7]
        box_quat = box_quat.unsqueeze(0)
        offset = torch.tensor([0.0, 0.0, 0.2], device=box_pos.device).unsqueeze(0)
        offset_world = quat_rotate(box_quat, offset).squeeze(0)
        target_cube_pos = box_pos + offset_world
        target_cube_quat = box_quat.squeeze(0)
        current_cube_pos = root_state[cube_root_idx, 0:3]
        current_cube_quat = root_state[cube_root_idx, 3:7]
        alpha = 0.8
        new_cube_pos = alpha * target_cube_pos + (1 - alpha) * current_cube_pos
        new_cube_quat = slerp(current_cube_quat.cpu().numpy(), target_cube_quat.cpu().numpy(), alpha)
        root_state[cube_root_idx, 0:3] = new_cube_pos
        root_state[cube_root_idx, 3:7] = torch.tensor(new_cube_quat, device=box_pos.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
    
    def keep_cube_attached_to_box_1(self):
        self.keep_cube_attached_to_box(self.component_handles[0], self.component_cube_handles[0])

    def keep_cube_attached_to_box_2(self):
        """body cube to wheel"""
        if not self.component_handles or not self.component_cube_handles:
            return
        env_ptr = self.envs[0]
        cube_handle = self.component_handles[2]
        box_handle = self.component_cube_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_state)
        cube_root_idx = self.gym.get_actor_index(env_ptr, cube_handle, gymapi.DOMAIN_SIM)
        box_root_idx = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
        box_pos = root_state[box_root_idx, 0:3]
        box_quat = root_state[box_root_idx, 3:7]
        offset = torch.tensor([0.0, 0.0, 0.2], device=box_pos.device).unsqueeze(0)   #new
        box_quat_unsq = box_quat.unsqueeze(0)
        offset_world = quat_rotate(box_quat_unsq, offset).squeeze(0)
        target_cube_pos = box_pos + offset_world
        current_cube_pos = root_state[cube_root_idx, 0:3]
        alpha = 0.6
        new_cube_pos = alpha * target_cube_pos + (1 - alpha) * current_cube_pos
        root_state[cube_root_idx, 0:3] = new_cube_pos
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))
    
    def keep_cube_attached_to_box_3(self):
        self.keep_cube_attached_to_box(self.component_handles[1], self.component_cube_handles[1])

    def fix_component_2_quat(self):
        env_ptr = self.envs[0]
        handle = self.component_handles[2]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_state)
        idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
        root_state[idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_state.device, dtype=root_state.dtype)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state))       

    def _build_mobile_robots(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.x = -2.0
        default_pose.p.y = -2.0
        default_pose.p.z = 0.25
        asset_options = gymapi.AssetOptions()

        asset_root = "Agent"  
        asset_file = "tracer_mini/urdf/tracer_mini.urdf"  

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        car_actor_handle = self.gym.create_actor(env_ptr, robot_asset, default_pose, "mobile_robots", col_group, col_filter, segmentation_id)

        self.mobile_handles.append(car_actor_handle)

        return
    
    def _build_mobile_robots_2(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.x = 2.0
        default_pose.p.y = -2.0
        default_pose.p.z = 0.25
        asset_options = gymapi.AssetOptions()

        asset_root = "Agent"  
        asset_file = "tracer_mini/urdf/tracer_mini.urdf" 

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        car_actor_handle = self.gym.create_actor(env_ptr, robot_asset, default_pose, "mobile_robots2", col_group, col_filter, segmentation_id)
        self.mobile_handles.append(car_actor_handle)

        return
    
    def _build_mobile_robots_3(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        default_pose.p.x = 2.0
        default_pose.p.y = 2.0
        default_pose.p.z = 0.25
        asset_options = gymapi.AssetOptions()

        asset_root = "Agent"  
        asset_file = "tracer_mini/urdf/tracer_mini.urdf"  # 资产文件名（URDF或SDF文件

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        car_actor_handle = self.gym.create_actor(env_ptr, robot_asset, default_pose, "mobile_robots3", col_group, col_filter, segmentation_id)
        self.mobile_handles.append(car_actor_handle)

        return

    def _reset_car_target(self, waypoints, robot_id=1):
        path = torch.tensor(
            [
            [5.0, 8.0],
            [1.0, 8.0],
            [1.4, 8.0],
            [1.4, 9.0],
            [0.405, 9.0],
            [0.405, -1.95]
            ], device=self.device)
        if robot_id == 1:
            self.waypoints = path
        elif robot_id == 2:
            self.waypoints_2 = path
        elif robot_id == 3:
            self.waypoints_3 = path
        
        self.final_waypoints_1 = torch.tensor(
            [
            [0.49, -1.5],
            [-1.5, -1.5],
            [-1.5, -2.4],
            [2.0,-2.4]
            ], device=self.device)
        return

    def _reset_car_target2(self, waypoints, robot_id=1):
        path = torch.tensor(
            [
                [4.0, -9.0],
                [4.0, -3.5],
                [4.0, -4.5],
                [5.0, -4.5],
                [5.0, -3.0],
                [1.0, -3.0],
            ], device=self.device)
        if robot_id == 1:
            self.waypoints = path
        elif robot_id == 2:
            self.waypoints_2 = path
        elif robot_id == 3:
            self.waypoints_3 = path
        return

    def _reset_car_target3(self, waypoints, robot_id=1):
        path = torch.tensor(
            [
                [-5.0, -8.0],
                [-0.8, -8.0],
                [-2.0, -8.0],
                [-2.0, -9.0],
                [-0.2, -9.0],
                [-0.2, -4.25]
            ], device=self.device)
        if robot_id == 1:
            self.waypoints = path
        elif robot_id == 2:
            self.waypoints_2 = path
        elif robot_id == 3:
            self.waypoints_3 = path
        return

    def _reset_car_target4(self, waypoints, robot_id=1):
        path = torch.tensor(
            [
                [-4.0, 9.0],
                [-4.0, -2.5],
                [-4.0, -1.0],
                [-5.0, -1.0],
                [-5.0, -3.0],
                [-1.1, -3.0]
            ], device=self.device)
        if robot_id == 1:
            self.waypoints = path
        elif robot_id == 2:
            self.waypoints_2 = path
        elif robot_id == 3:
            self.waypoints_3 = path
        return

    def _update_mobile_robots(self):
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        mobile_robot_state_tensor = all_root_state[-1]
        mobile_robot_2_state_tensor = all_root_state[-2]
        mobile_robot_3_state_tensor = all_root_state[-3]
        component_wheel_1_state_tensor = all_root_state[-14]
        component_wheel_2_state_tensor = all_root_state[-5]
        component_body_state_tensor = all_root_state[-6]
        
        step_size = torch.norm(self.update_pos)
        
        self._update_robot_1(mobile_robot_state_tensor, component_wheel_1_state_tensor, 
                            component_wheel_2_state_tensor, component_body_state_tensor, 
                            all_root_state, step_size)
        
        self._update_robot_2(mobile_robot_2_state_tensor, component_wheel_2_state_tensor,
                            all_root_state, step_size)
        
        self._update_robot_3(mobile_robot_3_state_tensor, component_body_state_tensor,
                            all_root_state, step_size)
        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_root_state))

    def _update_robot_1(self, mobile_robot_state_tensor, component_wheel_1_state_tensor,
                   component_wheel_2_state_tensor, component_body_state_tensor,
                   all_root_state, step_size):
        if self.current_wp_idx >= len(self.waypoints):
            return
            
        current_waypoint = self.waypoints[self.current_wp_idx]
        robot_pos = mobile_robot_state_tensor[0:2]
        
        component_states = [component_wheel_1_state_tensor, component_wheel_2_state_tensor, component_body_state_tensor]
        component_idxs = [-14, -5, -6]
        box_positions = torch.stack([cs[0:2] for cs in component_states], dim=0)
        dists = torch.norm(robot_pos.unsqueeze(0) - box_positions, dim=1)
        nearest_box_idx = torch.argmin(dists)
        nearest_box_pos = box_positions[nearest_box_idx]
        nearest_box_state = component_states[nearest_box_idx]
        nearest_box_global_idx = component_idxs[nearest_box_idx]

        robot_to_box_dist = torch.norm(robot_pos - nearest_box_pos)
        robot_to_waypoint_dist = torch.norm(robot_pos - current_waypoint)
        
        push_threshold = 0.6
        
        if robot_to_box_dist < push_threshold and robot_to_waypoint_dist > 0.05:
            self._smart_push_box(mobile_robot_state_tensor, nearest_box_state, nearest_box_global_idx,
                            current_waypoint, all_root_state, step_size)
        else:
            self._move_to_waypoint(mobile_robot_state_tensor, current_waypoint, all_root_state, step_size)

        if robot_to_waypoint_dist < 0.05:
            self.keep_cube_attached_to_box_1()
            self.keep_cube_attached_to_box_3()
            self.current_wp_idx += 1
            # 检查是否完成了所有waypoints
            if self.current_wp_idx >= len(self.waypoints):
                if hasattr(self, 'agent_states') and 'mobile_car_1(201)' in self.agent_states:
                    # 如果是push模式，完成后重置push状态
                    if getattr(self, '_llm_push_mode', False):
                        self.agent_states['mobile_car_1(201)'] = {'status': 'pushed', 'last_action': 'push'}
                        print(f"✅ Robot 1 完成push waypoints，推动完成")
                        if hasattr(self, 'current_robot_id') and (self.current_robot_id == 201 or self.current_robot_id == 1):
                            self._llm_push_mode = False
                            self.llm_action_type = None
                            print(f"🔄 Robot 1 push完成，重置LLM状态")
                    else:
                        self.agent_states['mobile_car_1(201)'] = {'status': 'moved', 'last_action': 'move'}
                        print(f"✅ Robot 1 完成所有RRT waypoints，状态更新为'moved'，push动作已可用")
            if self.current_wp_idx == 1:
                self._reset_target2([0])

    def _update_robot_2(self, mobile_robot_2_state_tensor, component_wheel_2_state_tensor,
                    all_root_state, step_size):
        component_wheel_1_state_tensor = all_root_state[-14]
        component_body_state_tensor = all_root_state[-6]

        if self.current_wp_idx_2 >= len(self.waypoints_2):
            return
            
        current_waypoint = self.waypoints_2[self.current_wp_idx_2]
        robot_pos = mobile_robot_2_state_tensor[0:2]
        
        component_states = [component_wheel_1_state_tensor, component_wheel_2_state_tensor, component_body_state_tensor]
        component_idxs = [-14, -5, -6]
        box_positions = torch.stack([cs[0:2] for cs in component_states], dim=0)
        dists = torch.norm(robot_pos.unsqueeze(0) - box_positions, dim=1)
        nearest_box_idx = torch.argmin(dists)
        nearest_box_pos = box_positions[nearest_box_idx]
        nearest_box_state = component_states[nearest_box_idx]
        nearest_box_global_idx = component_idxs[nearest_box_idx]

        robot_to_box_dist = torch.norm(robot_pos - nearest_box_pos)
        robot_to_waypoint_dist = torch.norm(robot_pos - current_waypoint)
        
        push_threshold = 0.6
        
        if robot_to_box_dist < push_threshold and robot_to_waypoint_dist > 0.05:
            self._smart_push_box(mobile_robot_2_state_tensor, nearest_box_state, nearest_box_global_idx,
                            current_waypoint, all_root_state, step_size, robot_id=2)
        else:
            self._move_to_waypoint(mobile_robot_2_state_tensor, current_waypoint, all_root_state, step_size, robot_id=2)

        if robot_to_waypoint_dist < 0.05:
            self.current_wp_idx_2 += 1
            # 检查是否完成了所有waypoints
            if self.current_wp_idx_2 >= len(self.waypoints_2):
                if hasattr(self, 'agent_states') and 'mobile_car_2(202)' in self.agent_states:
                    # 如果是push模式，完成后重置push状态
                    if getattr(self, '_llm_push_mode', False):
                        self.agent_states['mobile_car_2(202)'] = {'status': 'pushed', 'last_action': 'push'}
                        print(f"✅ Robot 2 完成push waypoints，推动完成")
                        if hasattr(self, 'current_robot_id') and (self.current_robot_id == 202 or self.current_robot_id == 2):
                            self._llm_push_mode = False
                            self.llm_action_type = None
                            print(f"🔄 Robot 2 push完成，重置LLM状态")
                    else:
                        self.agent_states['mobile_car_2(202)'] = {'status': 'moved', 'last_action': 'move'}
                        print(f"✅ Robot 2 完成所有RRT waypoints，状态更新为'moved'，push动作已可用")

    def _update_robot_3(self, mobile_robot_3_state_tensor, component_body_state_tensor,
                    all_root_state, step_size):
        component_wheel_1_state_tensor = all_root_state[-14]
        component_wheel_2_state_tensor = all_root_state[-5]
        if self.current_wp_idx_3 >= len(self.waypoints_3):
            return
            
        current_waypoint = self.waypoints_3[self.current_wp_idx_3]
        robot_pos = mobile_robot_3_state_tensor[0:2]
        
        component_wheel_1_state_tensor = all_root_state[-14]
        component_wheel_2_state_tensor = all_root_state[-5]
        component_states = [component_wheel_1_state_tensor, component_wheel_2_state_tensor, component_body_state_tensor]
        component_idxs = [-14, -5, -6]
        box_positions = torch.stack([cs[0:2] for cs in component_states], dim=0)
        dists = torch.norm(robot_pos.unsqueeze(0) - box_positions, dim=1)
        nearest_box_idx = torch.argmin(dists)
        nearest_box_pos = box_positions[nearest_box_idx]
        nearest_box_state = component_states[nearest_box_idx]
        nearest_box_global_idx = component_idxs[nearest_box_idx]

        robot_to_box_dist = torch.norm(robot_pos - nearest_box_pos)
        robot_to_waypoint_dist = torch.norm(robot_pos - current_waypoint)
        
        push_threshold = 0.6
        
        if robot_to_box_dist < push_threshold and robot_to_waypoint_dist > 0.05:
            self._smart_push_box(mobile_robot_3_state_tensor, nearest_box_state, nearest_box_global_idx,
                            current_waypoint, all_root_state, step_size, robot_id=3)
        else:
            self._move_to_waypoint(mobile_robot_3_state_tensor, current_waypoint, all_root_state, step_size, robot_id=3)

        if robot_to_waypoint_dist < 0.05:
            self.keep_cube_attached_to_box_3()
            self.current_wp_idx_3 += 1
            if self.current_wp_idx_3 >= len(self.waypoints_3):
                if hasattr(self, 'agent_states') and 'mobile_car_3(203)' in self.agent_states:
                    # 如果是push模式，完成后重置push状态
                    if getattr(self, '_llm_push_mode', False):
                        self.agent_states['mobile_car_3(203)'] = {'status': 'pushed', 'last_action': 'push'}
                        print(f"✅ Robot 3 完成push waypoints，推动完成")
                        if hasattr(self, 'current_robot_id') and (self.current_robot_id == 203 or self.current_robot_id == 3):
                            self._llm_push_mode = False
                            self.llm_action_type = None
                            print(f"🔄 Robot 3 push完成，重置LLM状态")
                    else:
                        self.agent_states['mobile_car_3(203)'] = {'status': 'moved', 'last_action': 'move'}
                        print(f"✅ Robot 3 完成所有RRT waypoints，状态更新为'moved'，push动作已可用")

    def _push_nearest_box(self, robot_state, component_states, component_idxs, direction, step_size, all_root_state):
        box_pos = torch.stack([cs[0:2] for cs in component_states], dim=0)
        dists = torch.norm(robot_state[0:2].unsqueeze(0) - box_pos, dim=1)
        
        threshold = 0.48
        mask = dists < threshold
        if mask.any():
            i = int(torch.nonzero(mask)[0])
            cs = component_states[i]
            cs[0:2] += direction * step_size
            all_root_state[component_idxs[i]] = cs

    def _handle_robot_2_final_push(self, mobile_robot_2_state_tensor, component_wheel_2_state_tensor,
                                all_root_state, step_size):
        final_waypoints = torch.tensor([[1.0, -3.1]], device=self.device)
        direction_2 = final_waypoints[0] - mobile_robot_2_state_tensor[0:2]
        direction_2 = direction_2 / torch.norm(direction_2)
        
        if torch.norm(mobile_robot_2_state_tensor[0:2] - final_waypoints[0:2]) > 0.05:
            if torch.norm(mobile_robot_2_state_tensor[0:2] - component_wheel_2_state_tensor[0:2]) < 0.57:
                component_wheel_2_state_tensor[0:2] += direction_2 * step_size / 2
                all_root_state[-5] = component_wheel_2_state_tensor
            mobile_robot_2_state_tensor[0:2] += direction_2 * step_size / 2
            all_root_state[-2] = mobile_robot_2_state_tensor

    def _smart_push_box(self, robot_state, box_state, box_global_idx, target_waypoint, all_root_state, step_size, robot_id=1):
        self.keep_cube_attached_to_box_1()
        self.keep_cube_attached_to_box_3()
        robot_pos = robot_state[0:2]
        box_pos = box_state[0:2]
        
        direction = target_waypoint - robot_pos
        direction = direction / torch.norm(direction)
        
        robot_state[0:2] += direction * step_size
        robot_to_box_dist = torch.norm(robot_pos - box_pos)

        if robot_to_box_dist < 0.5:
            box_state[0:2] += direction * step_size
            all_root_state[box_global_idx] = box_state
        
        dx, dy = direction
        angle = torch.atan2(dy, dx) + np.pi / 2
        half_angle = angle / 2.0
        w = torch.cos(half_angle)
        z = torch.sin(half_angle)
        new_quat = convert_wz(w, z)
        
        robot_state_idx = -robot_id
        all_root_state[robot_state_idx, 3:7] = new_quat

    def _move_to_waypoint(self, robot_state, waypoint, all_root_state, step_size, robot_id=1):
        self.keep_cube_attached_to_box_1()
        self.keep_cube_attached_to_box_3()
        
        robot_pos = robot_state[0:2]
        direction = waypoint - robot_pos
        direction = direction / torch.norm(direction)

        robot_state[0:2] += direction * step_size
        
        dx, dy = direction
        angle = torch.atan2(dy, dx) + np.pi / 2
        half_angle = angle / 2.0
        w = torch.cos(half_angle)
        z = torch.sin(half_angle)
        new_quat = convert_wz(w, z)
        
        robot_state_idx = -robot_id  
        all_root_state[robot_state_idx, 3:7] = new_quat

    def _reset_target(self, env_ids):
        n = len(env_ids)
        random_numbers = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)
        
        # rand_theta = 2 * np.pi * random_numbers
        rand_theta = np.pi *torch.tensor(0.2,device=self._target_pos.device)
        if self.is_ask_llm:
        # 换成英文，强调严格按照规定格式，看作成功率，输出结构化
            answer = self.ask_llm(f"当前有一个差速机器人要移动到目标点:[3.5,7.5]，但是在{self._box_states[env_ids, 0]}处有一个障碍物，并且在[4.0,6.0]处有一个长方体障碍物，尺寸为[1.0,5.0]，人形只能对小障碍物做搬运，不能搬运长方体，请考虑效率，给出想要把障碍物搬运到的目标点，只输出坐标")
            match = re.search(r'\\boxed\{([^\}]+)\}', answer)
            # import pdb; pdb.set_trace()
            ans = match.group(1)
            ans = re.search(r'\[([^\]]+)\]', ans)
            coordinates = ans.group(1).split(',')

            # 将字符串转换为浮点数
            x = float(coordinates[0].strip())
            y = float(coordinates[1].strip())

            self._target_pos[env_ids, 0] = torch.tensor(x)
            self._target_pos[env_ids, 1] = torch.tensor(y)

            print(f"目标点坐标为：{x},{y}")
        else:
            # rand_dist = (self._target_dist_max - self._target_dist_min) * torch.rand(
            #     [n], dtype=self._target_pos.dtype, device=self._target_pos.device) + self._target_dist_min
            # self._target_pos[env_ids, 0] = rand_dist * \
            #     torch.cos(rand_theta) + self._box_states[env_ids, 0]
            # self._target_pos[env_ids, 1] = rand_dist * \
            #     torch.sin(rand_theta) + self._box_states[env_ids, 1]
            self._target_pos[env_ids, 0] = 4.0
            self._target_pos[env_ids, 1] = 4.0
            
        self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0
        
        axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
        rand_rot = quat_from_angle_axis(rand_theta, axis)
        self._target_rot[env_ids] = rand_rot
        return

    def _reset_target2(self, env_ids):
        n = len(env_ids)
        random_numbers = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)
        rand_theta = 2 * np.pi * random_numbers
        if self.is_ask_llm:
        # 换成英文，强调严格按照规定格式，看作成功率，输出结构化
            answer = self.ask_llm(f"当前有一个差速机器人要移动到目标点:[3.5,7.5]，但是在{self._box_states[env_ids, 0]}处有一个障碍物，并且在[4.0,6.0]处有一个长方体障碍物，尺寸为[1.0,5.0]，人形只能对小障碍物做搬运，不能搬运长方体，请考虑效率，给出想要把障碍物搬运到的目标点，只输出坐标")
            match = re.search(r'\\boxed\{([^\}]+)\}', answer)
            # import pdb; pdb.set_trace()
            ans = match.group(1)
            ans = re.search(r'\[([^\]]+)\]', ans)
            coordinates = ans.group(1).split(',')

            # 将字符串转换为浮点数
            x = float(coordinates[0].strip())
            y = float(coordinates[1].strip())

            self._target_pos[env_ids, 0] = torch.tensor(x)
            self._target_pos[env_ids, 1] = torch.tensor(y)

            print(f"目标点坐标为：{x},{y}")
        else:
            self._target_pos[env_ids, 0] = -2.0
            self._target_pos[env_ids, 1] = 4.0
            
        self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0
        
        axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
        rand_rot = quat_from_angle_axis(rand_theta, axis)
        self._target_rot[env_ids] = rand_rot
        return
    
    ## LLM Integrated ##
    def get_positions_for_prompt(self, env_id, env_ptr):
        """return LLM API (area_positions, agent_positions)"""
        root = self.gym.acquire_actor_root_state_tensor(self.sim)
        root = gymtorch.wrap_tensor(root)
        area_positions = {}
        agent_positions = {}
        car_handles = [self.mobile_handles[0], self.mobile_handles[1], self.mobile_handles[2]]
        box_handles = [self.component_cube_handles[0], self.component_cube_handles[1], self.component_cube_handles[2]]
        
        for h in box_handles:
            idx = self.gym.get_actor_index(env_ptr, h, gymapi.DOMAIN_SIM)
            pos = root[idx, 0:3].cpu().numpy()
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
        
        self.current_area_positions = area_positions.copy()
        
        keys = ["<wheeled robot1> (202)", "<wheeled robot2> (203)", "<wheeled robot3> (204)"]
        for name, h in zip(keys, car_handles):
            idx = self.gym.get_actor_index(env_ptr, h, gymapi.DOMAIN_SIM)
            pos = root[idx, 0:2].cpu().numpy()
            agent_positions[name] = (float(pos[0]), float(pos[1]))
        
        return area_positions, agent_positions
    
    def move_robot_to_component(self, robot_id):
        if robot_id == 1:
            waypoints = getattr(self, 'waypoints', None)
            wp_idx = getattr(self, 'current_wp_idx', 0)
            handle = self.mobile_handles[0]
            agent_key = 'mobile_car_1(201)'
        elif robot_id == 2:
            waypoints = getattr(self, 'waypoints_2', None)
            wp_idx = getattr(self, 'current_wp_idx_2', 0)
            handle = self.mobile_handles[1]
            agent_key = 'mobile_car_2(202)'
        elif robot_id == 3:
            waypoints = getattr(self, 'waypoints_3', None)
            wp_idx = getattr(self, 'current_wp_idx_3', 0)
            handle = self.mobile_handles[2]
            agent_key = 'mobile_car_3(203)'
        else:
            print(f"⚠️ 未知robot_id: {robot_id}")
            return

        if waypoints is None:
            print(f"⚠️ Robot {robot_id} waypoints未初始化，尝试生成路径")
            self._generate_rrt_path_for_robot(robot_id)
            return
        
        if len(waypoints) == 0:
            print(f"⚠️ Robot {robot_id} waypoints为空")
            return
            
        # 检查移动是否已经完成
        if wp_idx >= len(waypoints):
            # 移动已完成，确保状态为'moved'
            if hasattr(self, 'agent_states') and agent_key in self.agent_states:
                current_status = self.agent_states[agent_key].get('status', 'idle')
                if current_status != 'moved':
                    self.agent_states[agent_key] = {'status': 'moved', 'last_action': 'move'}
                    print(f"✅ Robot {robot_id} 移动完成，状态更新为'moved'，push动作已可用")
            return

        env_ptr = self.envs[0]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
        robot_state = all_root_state[idx]

        if wp_idx < len(waypoints):
            target = waypoints[wp_idx]
            pos = robot_state[0:2].cpu().numpy()
            
            # 确保target也是numpy array
            if isinstance(target, torch.Tensor):
                target_pos = target[:2].cpu().numpy()
            else:
                target_pos = np.array(target[:2])
            
            direction = target_pos - pos
            norm = np.linalg.norm(direction)
            if norm > 1e-3:
                direction = direction / norm

                step = 0.05
                new_pos = pos + direction * min(step, norm)
                robot_state[0:2] = torch.tensor(new_pos, device=robot_state.device, dtype=robot_state.dtype)

                all_root_state[idx] = robot_state
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_root_state))

            if norm < 0.1:
                if robot_id == 1:
                    self.current_wp_idx += 1
                elif robot_id == 2:
                    self.current_wp_idx_2 += 1
                elif robot_id == 3:
                    self.current_wp_idx_3 += 1

    def _generate_rrt_path_for_robot(self, robot_id):
        if robot_id == 1 and hasattr(self, 'current_target_name'):
            if self.current_target_name == "left_wheel":
                area = self._get_component_area("left_wheel")
                self._assign_waypoints_by_area("mobile_car_1", area)
        elif robot_id == 2 and hasattr(self, 'current_target_name'):
            if self.current_target_name == "right_wheel":  
                area = self._get_component_area("right_wheel")
                self._assign_waypoints_by_area("mobile_car_2", area)
        elif robot_id == 3 and hasattr(self, 'current_target_name'):
            if self.current_target_name == "trunk":
                area = self._get_component_area("trunk")
                self._assign_waypoints_by_area("mobile_car_3", area)
        
    def push_box_with_robot(self, robot_id):
        self._generate_push_waypoints_for_robot(robot_id)
        
        if robot_id == 1:
            self.current_wp_idx = 0
            agent_key = 'mobile_car_1(201)'
        elif robot_id == 2:
            self.current_wp_idx_2 = 0
            agent_key = 'mobile_car_2(202)'
        elif robot_id == 3:
            self.current_wp_idx_3 = 0
            agent_key = 'mobile_car_3(203)'
        else:
            print(f"⚠️ 未知robot_id: {robot_id}")
            return
            
        # 开始push时更新状态
        if hasattr(self, 'agent_states') and agent_key in self.agent_states:
            self.agent_states[agent_key] = {'status': 'pushing', 'last_action': 'push'}
            print(f"🚀 Robot {robot_id} 开始推动，状态更新为'pushing'")
            
        return True
            
    def execute_robot_action(self, robot_id, action_type):
        if action_type == "move":
            self.move_robot_to_component(robot_id)
        elif action_type == "push":
            self.push_box_with_robot(robot_id)
        else:
            print(f"⚠️ 未知动作类型: {action_type}")
    
    def _execute_llm_action(self, agent_action, agent_message):
        try:
            robot_match = re.search(r'<([^>]+)>\s*\((\d+)\)', agent_action)
            action_type = None
            
            if "[move]" in agent_action:
                action_type = "move"
            elif "[push]" in agent_action:
                action_type = "push"
            elif "[check]" in agent_action:
                action_type = "check"
            elif "[pick]" in agent_action:
                action_type = "pick"
            elif "[wait]" in agent_action:
                action_type = "wait"
            elif "[walk]" in agent_action:
                action_type = "walk"
            elif "[carry]" in agent_action:
                action_type = "carry"
            elif "[movetowards]" in agent_action:
                action_type = "movetowards"
            
            if robot_match and action_type:
                robot_name = robot_match.group(1)
                robot_id = int(robot_match.group(2))
                
                target_match = re.search(r'<([^>]+)>\s*\((\d+)\)', agent_action.split('> (')[1] if '> (' in agent_action and agent_action.count('<') > 1 else '')
                target_name = target_match.group(1) if target_match else ""
                
                self._set_current_context(robot_name, robot_id, target_name)
                self._dispatch_action(action_type, robot_name, target_name, robot_id)
                
            else:
                print(f"⚠️ 未识别的LLM动作格式: {agent_action}")
                
        except Exception as e:
            print(f"❌ 执行LLM动作失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_current_context(self, robot_name, robot_id, target_name=""):
        self.current_robot_name = robot_name
        self.current_robot_id = robot_id
        
        if robot_id == 201 or "left wheel" in target_name.lower():
            self.current_target_id = 0
            self.current_target_name = "left_wheel"
        elif robot_id == 202 or "right wheel" in target_name.lower():
            self.current_target_id = 1
            self.current_target_name = "right_wheel"
        elif robot_id == 203 or "trunk" in target_name.lower():
            self.current_target_id = 2
            self.current_target_name = "trunk"
    
    def _dispatch_action(self, action_type, robot_name, target_name, robot_id):
        self.llm_action_type = action_type
        
        if action_type == "move":
            if "mobile_car" in robot_name:
                simple_robot_id = robot_id - 200 if robot_id > 200 else robot_id
                # 开始移动时更新状态
                agent_key = f'mobile_car_{simple_robot_id}({robot_id})'
                if hasattr(self, 'agent_states') and agent_key in self.agent_states:
                    self.agent_states[agent_key] = {'status': 'moving', 'last_action': 'move'}
                    print(f"🚶 {robot_name} 开始移动，状态更新为'moving'")
                self.execute_robot_action(simple_robot_id, "move")
            else:
                area = self._get_target_area(target_name)
                self.move_robot(robot_name, area)
                
        elif action_type == "push":
            self._llm_push_mode = True
            self.llm_action_type = "push"  # 设置push动作类型，类似move的逻辑
            if "mobile_car" in robot_name:
                simple_robot_id = robot_id - 200 if robot_id > 200 else robot_id
                self.execute_robot_action(simple_robot_id, "push")
            else:
                self.push_component(robot_name, target_name)
                
        elif action_type == "check":
            if "franka" in robot_name:
                self.franka_check(robot_name, target_name)
            else:
                print(f"Check action for {robot_name} on {target_name}")
                
        elif action_type == "pick":
            print(f"Pick action: {robot_name} picking {target_name}")
            
        elif action_type == "wait":
            self.wait_agent(robot_name)
            
        elif action_type == "walk":
            area = self._get_target_area(target_name)
            self.walk_humanoid(robot_name, area)
            
        elif action_type == "carry":
            self.carry_obstacle(robot_name, target_name)
            
        elif action_type == "movetowards":
            area = self._get_target_area(target_name)
            self.move_robot(robot_name, area)
    
    def _get_target_area(self, target_name):
        if "trunk" in target_name.lower():
            return "D"
        elif "left" in target_name.lower() and "wheel" in target_name.lower():
            return "B"
        elif "right" in target_name.lower() and "wheel" in target_name.lower():
            return "C"
        else:
            return "A"

    # LLM 技能执行方法
    def explore_area(self, robot_name, area_name):
        target_id = getattr(self, 'current_target_id', None)
        actual_robot_name = self._convert_llm_robot_name(robot_name, target_id)
        self._assign_waypoints_by_area(actual_robot_name, area_name)
        
    def move_robot(self, robot_name, area_name):
        print(f"🚶 Moving {robot_name} to area {area_name}")
        target_id = getattr(self, 'current_target_id', None)
        actual_robot_name = self._convert_llm_robot_name(robot_name, target_id)
        self._assign_waypoints_by_area(actual_robot_name, area_name)
        
    def walk_humanoid(self, robot_name, area_name):
        """人形机器人行走"""
        print(f"Humanoid {robot_name} walking to area {area_name}")
        target_id = getattr(self, 'current_target_id', None)
        actual_robot_name = self._convert_llm_robot_name(robot_name, target_id)
        self._assign_waypoints_by_area(actual_robot_name, area_name)
        
    def carry_obstacle(self, robot_name, target_name):
        """搬运障碍物"""
        print(f"{robot_name} carrying obstacle {target_name}")
        target_id = getattr(self, 'current_target_id', None)
        actual_robot_name = self._convert_llm_robot_name(robot_name, target_id)
        # 这里可以添加具体的搬运逻辑
        
    def push_component(self, robot_name, target_name):
        """推动组件"""
        print(f"{robot_name} pushing component {target_name}")
        target_id = getattr(self, 'current_target_id', None)
        actual_robot_name = self._convert_llm_robot_name(robot_name, target_id)

        
    def franka_check(self, robot_name, target_name):
        print(f"Franka {robot_name} checking {target_name}")
        
        root = self.gym.acquire_actor_root_state_tensor(self.sim)
        root = gymtorch.wrap_tensor(root)
        
        franka_idx = self.gym.get_actor_index(self.envs[0], self.franka_handles[0], gymapi.DOMAIN_SIM)
        franka_state_tensor = root[franka_idx]
        
        near_thresh = 0.6
        
        if "left wheel" in target_name.lower():
            if hasattr(self, 'component_cube_handles') and len(self.component_cube_handles) > 0:
                component_idx = self.gym.get_actor_index(self.envs[0], self.component_cube_handles[0], gymapi.DOMAIN_SIM)
                component_state_tensor = root[component_idx]
                distance = torch.norm(component_state_tensor[0:2] - franka_state_tensor[0:2])
                d = float(distance.item())
                
                if d < near_thresh:
                    print(f"✅ Left wheel has arrived at franka area (distance: {d:.3f})")
                    return True
                else:
                    print(f"❌ Left wheel not yet at franka area (distance: {d:.3f}, threshold: {near_thresh})")
                    return False
        
        elif "right wheel" in target_name.lower():
            if hasattr(self, 'component_cube_handles') and len(self.component_cube_handles) > 1:
                component_idx = self.gym.get_actor_index(self.envs[0], self.component_cube_handles[1], gymapi.DOMAIN_SIM)
                component_state_tensor = root[component_idx]
                distance = torch.norm(component_state_tensor[0:2] - franka_state_tensor[0:2])
                d = float(distance.item())
                
                if d < near_thresh:
                    print(f"✅ Right wheel has arrived at franka area (distance: {d:.3f})")
                    return True
                else:
                    print(f"❌ Right wheel not yet at franka area (distance: {d:.3f}, threshold: {near_thresh})")
                    return False
        
        else:
            print(f"⚠️ Unknown target: {target_name}")
            return False

    def wait_agent(self, robot_name):
        """等待智能体"""
        print(f"Agent {robot_name} waiting")
        
    def _get_component_area(self, component_name):
        """动态获取组件所在的区域"""
        try:
            # 获取当前的组件位置
            root = self.gym.acquire_actor_root_state_tensor(self.sim)
            root = gymtorch.wrap_tensor(root)
            
            # 组件handle映射
            component_handles = {
                'left_wheel': self.component_cube_handles[0] if hasattr(self, 'component_cube_handles') and len(self.component_cube_handles) > 0 else None,
                'right_wheel': self.component_cube_handles[1] if hasattr(self, 'component_cube_handles') and len(self.component_cube_handles) > 1 else None,
                'trunk': self.component_cube_handles[2] if hasattr(self, 'component_cube_handles') and len(self.component_cube_handles) > 2 else None,
            }
            
            # 根据组件名称获取对应的handle
            handle = None
            if 'left' in component_name.lower() or 'wheel1' in component_name.lower():
                handle = component_handles.get('left_wheel')
            elif 'right' in component_name.lower() or 'wheel2' in component_name.lower():
                handle = component_handles.get('right_wheel')
            elif 'trunk' in component_name.lower() or 'body' in component_name.lower():
                handle = component_handles.get('trunk')
            
            if handle is None:
                print(f"⚠️ 无法找到组件 {component_name} 的handle，使用默认区域")
                return 'A'  # 默认区域
            
            # 获取组件的实际位置
            idx = self.gym.get_actor_index(self.envs[0], handle, gymapi.DOMAIN_SIM)
            pos = root[idx, 0:3].cpu().numpy()
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            
            # 根据位置确定区域 (与get_positions_for_prompt保持一致)
            if x >= 0 and y >= 0:
                area = 'A'
            elif x < 0 and y >= 0:
                area = 'B'
            elif x < 0 and y < 0:
                area = 'C'
            else:
                area = 'D'
            
            return area
            
        except Exception as e:
            print(f"❌ 获取组件 {component_name} 位置失败: {e}")
            return 'A'  # 出错时返回默认区域
        
    def _convert_llm_robot_name(self, llm_robot_name, target_id=None):
        """将LLM规划的robot名称转换为实际的robot名称"""
        # 优先使用直接的名称映射
        if llm_robot_name in self.llm_robot_name_mapping:
            actual_name = self.llm_robot_name_mapping[llm_robot_name]
            print(f"🔄 名称转换: {llm_robot_name} -> {actual_name}")
            return actual_name
        # 回退到基于target_id的映射（向后兼容）
        elif llm_robot_name == 'mobile_car' and target_id is not None:
            if target_id in self.target_id_to_robot_mapping:
                actual_name = self.target_id_to_robot_mapping[target_id]
                print(f"🔄 Mobile car名称转换: {llm_robot_name}(target_id={target_id}) -> {actual_name}")
                return actual_name
            else:
                print(f"⚠️ 未找到target_id={target_id}的mobile_car映射，使用robot1")
                return 'robot1'
        else:
            print(f"⚠️ 未找到{llm_robot_name}的映射，使用原名称")
            return llm_robot_name
        
    def _handle_component_0(self):
        """处理组件0 (wheel1)"""
        self.franka_counter = 0
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        component_wheel_1_state_tensor = all_root_state[-14]  # box1
        franka_root_idx = self.gym.get_actor_index(self.envs[0], self.franka_handles[0], gymapi.DOMAIN_SIM)
        franka_state_tensor = all_root_state[franka_root_idx]
        dist0 = torch.norm(component_wheel_1_state_tensor[0:2] - franka_state_tensor[0:2])
        d0 = float(dist0.item())
        near_thresh = 0.6
        
        if d0 < near_thresh:
            self._franka_take_and_place_fsm(self.component_handles[0])
        else:
            self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
    
    def _handle_component_1(self):
        """处理组件1 (wheel2)"""
        self.franka_counter = 2
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        component_wheel_2_state_tensor = all_root_state[-5]   # box2
        franka_root_idx = self.gym.get_actor_index(self.envs[0], self.franka_handles[0], gymapi.DOMAIN_SIM)
        franka_state_tensor = all_root_state[franka_root_idx]
        dist1 = torch.norm(component_wheel_2_state_tensor[0:2] - franka_state_tensor[0:2])
        d1 = float(dist1.item())
        near_thresh = 0.6
        
        if d1 < near_thresh:
            self._franka_take_and_place_fsm2(self.component_handles[1])
        else:
            self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
    
    def _handle_component_2(self):
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        component_body_state_tensor = all_root_state[-6]      # box3
        franka_root_idx = self.gym.get_actor_index(self.envs[0], self.franka_handles[0], gymapi.DOMAIN_SIM)
        franka_state_tensor = all_root_state[franka_root_idx]
        dist2 = torch.norm(component_body_state_tensor[0:2] - franka_state_tensor[0:2])
        d2 = float(dist2.item())
        near_thresh = 0.6
        
        if d2 < near_thresh:
            self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
        else:
            self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
     
    def _assign_waypoints_by_area(self, robot_name: str, area_name: str):
        import torch as _torch
        area_targets = {
            'A': _torch.tensor([4.0,  8.0], device=self.device),
            'B': _torch.tensor([-4.0, 8.0], device=self.device),
            'C': _torch.tensor([-4.0,-8.0], device=self.device),
            'D': _torch.tensor([4.0, -8.0], device=self.device),
            'franka_area': _torch.tensor([0.0, -1.0], device=self.device), 
        }
        
        if hasattr(self, 'current_area_positions') and area_name in self.current_area_positions:
            real_pos = self.current_area_positions[area_name]
            target = _torch.tensor([real_pos[0], real_pos[1]], device=self.device)
        elif area_name == 'franka_area':
            franka_positions = {
                'robot1': _torch.tensor([-1.5, -1.0], device=self.device),  
                'robot2': _torch.tensor([1.5, -1.0], device=self.device),   
                'robot3': _torch.tensor([0.0, -0.5], device=self.device),   
            }
            
            robot_key = None
            if 'robot1' in robot_name:
                robot_key = 'robot1'
            elif 'robot2' in robot_name:
                robot_key = 'robot2'
            elif 'robot3' in robot_name:
                robot_key = 'robot3'
                
            if robot_key:
                target = franka_positions[robot_key]
            else:
                target = area_targets['franka_area'] 
        elif area_name in area_targets:
            target = area_targets[area_name]
        else:
            print(f"Area {area_name} not found, skipping waypoint assignment")
            return
        
        path = _torch.stack([target], dim=0)

        robot_id = 1
        if 'robot1' in robot_name:
            robot_id = 1
        elif 'robot2' in robot_name:
            robot_id = 2
        elif 'robot3' in robot_name:
            robot_id = 3
        self._set_robot_waypoints(robot_id, path)
        
    def _set_robot_waypoints(self, robot_id: int, path):
        if robot_id == 1:
            self.waypoints = path
            self.current_wp_idx = 0
            if hasattr(self, '_waypoints_reset'):
                self._waypoints_reset = False
        elif robot_id == 2:
            self.waypoints_2 = path
            self.current_wp_idx_2 = 0
            if hasattr(self, '_waypoints_reset_2'):
                self._waypoints_reset_2 = False
        elif robot_id == 3:
            self.waypoints_3 = path
            self.current_wp_idx_3 = 0
            if hasattr(self, '_waypoints_reset_3'):
                self._waypoints_reset_3 = False

    def _check_llm_safety_and_update(self, robot_id=1):
        if self.llm_manager.enable_llm:
            self.llm_manager.update_agent_states()
            collision_risks = self.llm_manager.check_collision_risks()
            self.llm_manager.execute_safety_control(collision_risks)
            robot_key = f'mobile_{robot_id-1}'
            return self.llm_manager.get_mobile_robot_stop_flags().get(robot_key, False)
        return False
    
    def _get_rrt_path(self, robot_id=1, path_type="initial"):
        if not hasattr(self, '_waypoints_initialized') or not self._waypoints_initialized:
            return None
        
        if path_type == "initial":
            if robot_id == 1 and hasattr(self, 'waypoints') and self.waypoints is not None:
                return self.waypoints
            elif robot_id == 2 and hasattr(self, 'waypoints_2') and self.waypoints_2 is not None:
                return self.waypoints_2
            elif robot_id == 3 and hasattr(self, 'waypoints_3') and self.waypoints_3 is not None:
                return self.waypoints_3
        elif path_type == "complete":
            if hasattr(self, 'rrt_paths') and self.rrt_paths is not None:
                if robot_id == 1 and len(self.rrt_paths) > 0:
                    return torch.tensor(self.rrt_paths[0], device=self.device, dtype=torch.float32)
                elif robot_id == 2 and len(self.rrt_paths) > 1:
                    return torch.tensor(self.rrt_paths[1], device=self.device, dtype=torch.float32)
                elif robot_id == 3 and len(self.rrt_paths) > 2:
                    return torch.tensor(self.rrt_paths[2], device=self.device, dtype=torch.float32)
        
        return None
    
    def _is_robot_stopped_by_llm(self, robot_id=1):
        if self.llm_manager.enable_llm:
            robot_key = f'mobile_{robot_id-1}'
            return self.llm_manager.get_mobile_robot_stop_flags().get(robot_key, False)
        return False
    
    def _get_safe_path_for_robot(self, robot_id=1, path_type="return"):
        is_stopped = self._is_robot_stopped_by_llm(robot_id)
        
        rrt_path = self._get_rrt_path(robot_id, "initial")
        if rrt_path is not None:
            if is_stopped:
                rrt_path_np = rrt_path.cpu().numpy()
                safe_path = self._add_safety_waypoints(rrt_path_np, robot_id)
                return torch.tensor(safe_path, device=self.device, dtype=torch.float32)
            else:
                return rrt_path
        
        if path_type == "return":
            if robot_id == 1:
                if is_stopped:
                    return torch.tensor(
                        [
                        [5.0, 8.0],
                        [3.0, 8.0],  
                        [1.0, 8.0],
                        [1.4, 8.0],
                        [1.4, 9.0],
                        [0.405, 9.0],
                        [0.405, -1.95]
                        ], device=self.device)
                else:
                    return torch.tensor(
                        [
                        [5.0, 8.0],
                        [1.0, 8.0],
                        [1.4, 8.0],
                        [1.4, 9.0],
                        [0.405, 9.0],
                        [0.405, -1.95]
                        ], device=self.device)
            elif robot_id == 2:
                if is_stopped:
                    return torch.tensor(
                        [
                            [4.0, -9.0],
                            [4.0, -6.0],  
                            [4.0, -3.5],
                            [4.0, -4.5],
                            [5.0, -4.5],
                            [5.0, -3.0],
                            [1.0, -3.0],
                        ], device=self.device)
                else:
                    return torch.tensor(
                        [
                            [4.0, -9.0],
                            [4.0, -3.5],
                            [4.0, -4.5],
                            [5.0, -4.5],
                            [5.0, -3.0],
                            [1.0, -3.0],
                        ], device=self.device)
            elif robot_id == 3:
                if is_stopped:
                    return torch.tensor(
                        [
                            [-5.0, -8.0],
                            [-3.0, -8.0],  
                            [-0.8, -8.0],
                            [-2.0, -8.0],
                            [-2.0, -9.0],
                            [-0.2, -9.0],
                            [-0.2, -4.25]
                        ], device=self.device)
                else:
                    return torch.tensor(
                        [
                            [-5.0, -8.0],
                            [-0.8, -8.0],
                            [-2.0, -8.0],
                            [-2.0, -9.0],
                            [-0.2, -9.0],
                            [-0.2, -4.25]
                        ], device=self.device)
        return torch.tensor([], device=self.device)
    
    def _add_safety_waypoints(self, rrt_path, robot_id):
        if len(rrt_path) < 2:
            return rrt_path
        
        safe_path = []
        for i in range(len(rrt_path) - 1):
            current_point = rrt_path[i]
            next_point = rrt_path[i + 1]
            
            safe_path.append(current_point)
            
            distance = np.linalg.norm(next_point - current_point)
            
            if distance > 2.0: 
                mid_point = (current_point + next_point) / 2
                safe_path.append(mid_point)
        
        safe_path.append(rrt_path[-1])
        
        if robot_id == 1:
            if len(safe_path) > 0:
                start_point = safe_path[0]
                safe_start = start_point + np.array([0.5, 0.5]) 
                safe_path.insert(0, safe_start)
        elif robot_id == 2:
            if len(safe_path) > 0:
                start_point = safe_path[0]
                safe_start = start_point + np.array([0.3, -0.3])  
                safe_path.insert(0, safe_start)
        elif robot_id == 3:
            if len(safe_path) > 0:
                start_point = safe_path[0]
                safe_start = start_point + np.array([-0.3, 0.3])  
                safe_path.insert(0, safe_start)
        
        return np.array(safe_path)
    
    def _update_mobile_robots(self):
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        mobile_robot_state_tensor = all_root_state[-1]
        mobile_robot_2_state_tensor = all_root_state[-2]
        mobile_robot_3_state_tensor = all_root_state[-3]
        component_wheel_1_state_tensor = all_root_state[-14]
        component_wheel_2_state_tensor = all_root_state[-5]
        component_body_state_tensor = all_root_state[-6]
        
        step_size = torch.norm(self.update_pos)
        
        self._update_robot_1(mobile_robot_state_tensor, component_wheel_1_state_tensor, 
                            component_wheel_2_state_tensor, component_body_state_tensor, 
                            all_root_state, step_size)
        
        self._update_robot_2(mobile_robot_2_state_tensor, component_wheel_2_state_tensor,
                            all_root_state, step_size)
        
        self._update_robot_3(mobile_robot_3_state_tensor, component_body_state_tensor,
                            all_root_state, step_size)
        
    def _generate_push_waypoints_for_robot(self, robot_id):
        env_ptr = self.envs[0]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        # 获取robot位置
        if robot_id == 1:
            robot_handle = self.mobile_handles[0]
        elif robot_id == 2:
            robot_handle = self.mobile_handles[1] 
        elif robot_id == 3:
            robot_handle = self.mobile_handles[2]
        else:
            return
            
        robot_idx = self.gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
        robot_pos = all_root_state[robot_idx, 0:2].cpu().numpy()
        
        # 获取所有箱子位置，选择最近的箱子（与_update_robot_X逻辑一致）
        box_handles = [self.component_cube_handles[0], self.component_cube_handles[1], self.component_cube_handles[2]]
        box_positions = []
        for box_handle in box_handles:
            box_idx = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
            box_pos = all_root_state[box_idx, 0:2].cpu().numpy()
            box_positions.append(box_pos)
        
        # 选择最近的箱子
        distances = [((robot_pos[0] - bp[0])**2 + (robot_pos[1] - bp[1])**2)**0.5 for bp in box_positions]
        nearest_idx = distances.index(min(distances))
        box_pos = box_positions[nearest_idx]
        
        # 调试信息：显示Robot和所有箱子的位置
        box_names = ["left_wheel", "right_wheel", "trunk"]
        print(f"🤖 Robot {robot_id} 位置: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
        for i, (bp, dist) in enumerate(zip(box_positions, distances)):
            print(f"  📦 {box_names[i]} 位置: ({bp[0]:.1f}, {bp[1]:.1f}), 距离: {dist:.2f}")
        print(f"  ✅ 选择最近的: {box_names[nearest_idx]} (距离: {distances[nearest_idx]:.2f})")
        
        # 根据box位置生成推箱子waypoints
        if box_pos[0] >= 0 and box_pos[1] >= 0:
            self._reset_car_target([0], robot_id=robot_id)
        elif box_pos[0] >= 0 and box_pos[1] < 0:
            self._reset_car_target2([0], robot_id=robot_id)
        elif box_pos[0] < 0 and box_pos[1] < 0:
            self._reset_car_target3([0], robot_id=robot_id)  
        elif box_pos[0] < 0 and box_pos[1] >= 0:
            self._reset_car_target4([0], robot_id=robot_id)
            
        box_names = ["left_wheel", "right_wheel", "trunk"]
        print(f"🚀 Robot {robot_id} 选择最近的箱子: {box_names[nearest_idx]}，位置: ({box_pos[0]:.1f}, {box_pos[1]:.1f})")

    def _update_specific_robot(self, robot_id):
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        
        step_size = torch.norm(self.update_pos)
        
        if robot_id == 1:
            mobile_robot_state_tensor = all_root_state[-1]
            component_wheel_1_state_tensor = all_root_state[-14]
            component_wheel_2_state_tensor = all_root_state[-5]
            component_body_state_tensor = all_root_state[-6]
            self._update_robot_1(mobile_robot_state_tensor, component_wheel_1_state_tensor, 
                                component_wheel_2_state_tensor, component_body_state_tensor, 
                                all_root_state, step_size)
        elif robot_id == 2:
            mobile_robot_2_state_tensor = all_root_state[-2]
            component_wheel_2_state_tensor = all_root_state[-5]
            self._update_robot_2(mobile_robot_2_state_tensor, component_wheel_2_state_tensor,
                                all_root_state, step_size)
        elif robot_id == 3:
            mobile_robot_3_state_tensor = all_root_state[-3]
            component_body_state_tensor = all_root_state[-6]
            self._update_robot_3(mobile_robot_3_state_tensor, component_body_state_tensor,
                                all_root_state, step_size)
        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(all_root_state))

    def _update_mobile_robots_llm_mode(self):
        if getattr(self, '_llm_waiting_for_plan', True):
            return
        
        current_skill = self._get_current_executing_skill()
        if current_skill in ['A', 'B', 'C', 'D']:
            return
        
        if not self.llm_manager.enable_llm:
            return

        plan_status = self.llm_manager.get_plan_status()
        if plan_status != 'executing':
            return

        if getattr(self, '_llm_push_mode', False):
            if hasattr(self, 'current_robot_id'):
                robot_full_id = self.current_robot_id
                simple_robot_id = robot_full_id - 200 if robot_full_id > 200 else robot_full_id
                self._update_specific_robot(simple_robot_id)
            return
        
        # 持续执行移动动作 - 只移动LLM指定的robot
        if hasattr(self, 'llm_action_type') and self.llm_action_type == "move":
            if hasattr(self, 'current_robot_id'):
                robot_full_id = self.current_robot_id
                simple_robot_id = robot_full_id - 200 if robot_full_id > 200 else robot_full_id
                self._update_specific_robot(simple_robot_id)
                return
        
        # 持续执行push动作 - 只push LLM指定的robot
        if hasattr(self, 'llm_action_type') and self.llm_action_type == "push":
            if hasattr(self, 'current_robot_id'):
                robot_full_id = self.current_robot_id
                simple_robot_id = robot_full_id - 200 if robot_full_id > 200 else robot_full_id
                self._update_specific_robot(simple_robot_id)
                return  

    def _get_current_executing_skill(self):
        try:
            if self.llm_manager.enable_llm:
                current_plan = self.llm_manager.get_current_plan()
                execution_index = self.llm_manager.get_plan_execution_index()
                
                if current_plan and 0 <= execution_index < len(current_plan):
                    current_step = current_plan[execution_index]
                    skill = current_step.get('skill', 'Unknown')
                    return skill
            
            return 'Unknown'
        except Exception as e:
            print(f"⚠️  获取当前技能失败: {e}")
            return 'Unknown'

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        box_env_ids_int32 = self._box_actor_ids[env_ids]
        reset_env_ids_int32 = box_env_ids_int32

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(reset_env_ids_int32), len(reset_env_ids_int32))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        
        # LLM规划模式 - 从一开始就等待LLM规划
        if self.llm_manager.enable_llm:
            # 初始化LLM等待状态
            if not hasattr(self, '_llm_mode_initialized'):
                self._llm_mode_initialized = True
                self.llm_manager.llm_mode_active = True
                self._llm_waiting_for_plan = True
            
            # 定期调用LLM更新
            self.llm_manager.update(int(self.progress_buf[0]))
            
            # 检查LLM规划状态
            plan_status = self.llm_manager.get_plan_status()
            if plan_status == "planning":
                self.llm_manager.llm_mode_active = True
                self._llm_waiting_for_plan = True
                if self.progress_buf[0] % 500 == 0:
                    print(f"⏳ LLM规划中，机器人等待...")
                    
            elif plan_status == "executing":
                self.llm_manager.llm_mode_active = True
                self._llm_waiting_for_plan = False
                if not hasattr(self, '_llm_execution_started'):
                    self._llm_execution_started = True
                    
            elif plan_status == "completed":
                self.llm_manager.llm_mode_active = True
                self._llm_waiting_for_plan = False
                if not hasattr(self, '_llm_execution_completed'):
                    self._llm_execution_completed = True
                    print(f"✅ LLM规划执行完成！")
                    
            else:
                self.llm_manager.llm_mode_active = True
                self._llm_waiting_for_plan = True

        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        all_root_state = gymtorch.wrap_tensor(root_state)
        self.fix_component_2_quat()
        self.keep_cube_attached_to_box_2()

        franka_root_idx = self.gym.get_actor_index(self.envs[0], self.franka_handles[0], gymapi.DOMAIN_SIM)
        franka_state_tensor = all_root_state[franka_root_idx]

        component_wheel_1_state_tensor = all_root_state[-14]  # box1
        component_wheel_2_state_tensor = all_root_state[-5]   # box2
        component_body_state_tensor = all_root_state[-6]      # box3

        dist0 = torch.norm(component_wheel_1_state_tensor[0:2] - franka_state_tensor[0:2])
        dist1 = torch.norm(component_wheel_2_state_tensor[0:2] - franka_state_tensor[0:2])
        dist2 = torch.norm(component_body_state_tensor[0:2] - franka_state_tensor[0:2])
        d0 = float(dist0.item())
        d1 = float(dist1.item())
        d2 = float(dist2.item())
        near_thresh = 0.6
        # The distance between parts and franka
        if self.absorbed == 1:
            self.apply_magnetic_force(range = 2.0,mag=True)
        if self.absorbed1 == 1:
            self.apply_magnetic_force_1(range = 2.0, mag=True) 
        
        if not (
            (self.franka_counter == 0 and d0 < near_thresh) or 
            (self.franka_counter == 2 and d1 < near_thresh)
        ):
            self.keep_cube_attached_to_box_1()
            self.keep_cube_attached_to_box_3()
        
        if not getattr(self.llm_manager, 'llm_mode_active', False):
            if self.wait_counter < self.wait_steps:
                self.wait_counter += 1
            else:
                if self.franka_counter == 0:
                    if d0 < near_thresh:
                        self._franka_take_and_place_fsm(self.component_handles[0])
                    else:
                        self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
                elif self.franka_counter == 2:
                    if d1 < near_thresh:
                        self._franka_take_and_place_fsm2(self.component_handles[1])
                    else:
                        self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
                else:
                    self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
        else:
            if getattr(self, '_llm_waiting_for_plan', True):
                self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
            else:
                if hasattr(self, 'current_target_id') and self.current_target_id is not None:
                    if self.current_target_id in self.target_id_handlers:
                        self.target_id_handlers[self.current_target_id]()
                    else:
                        self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
                else:
                    # 没有target_id时，执行原有的逻辑
                    if self.wait_counter < self.wait_steps:
                        self.wait_counter += 1
                    else:
                        if self.franka_counter == 0:
                            if d0 < near_thresh:
                                self._franka_take_and_place_fsm(self.component_handles[0])
                            else:
                                self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
                        elif self.franka_counter == 2:
                            if d1 < near_thresh:
                                self._franka_take_and_place_fsm2(self.component_handles[1])
                            else:
                                self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
                        else:
                            self.gym.set_dof_position_target_tensor(self.sim, self.pd_tar_tensor)
        
        if getattr(self.llm_manager, 'llm_mode_active', False):
            self._update_mobile_robots_llm_mode()
        
        return

    def update_standing_and_held_points(self, box_states, tar_pos, tar_rot, env_ids=None):
        if env_ids is None:
            box_pos = box_states[..., 0:3]
            box_rot = box_states[..., 3:7]
            # hard code: set standing points to the left
            self.box_standing_points[:] = box_pos + \
                quat_rotate(box_rot, self.stand_held_points_offset[0])
            self.box_standing_points[..., 2] = 0.0
            self.box_held_points[:] = box_pos + \
                quat_rotate(box_rot, self.stand_held_points_offset[2])
            # hard code: set standing points to the left
            self.tar_standing_points[:] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[0])
            self.tar_held_points[:] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[2])

        else:
            box_pos = box_states[env_ids, 0:3]
            box_rot = box_states[env_ids, 3:7]
            tar_pos = tar_pos[env_ids]
            tar_rot = tar_rot[env_ids]
            self.box_standing_points[env_ids] = box_pos + \
                quat_rotate(box_rot,
                            self.stand_held_points_offset[0][env_ids])
            self.box_standing_points[env_ids, 2] = 0.0
            self.box_held_points[env_ids] = box_pos + \
                quat_rotate(box_rot,
                            self.stand_held_points_offset[2][env_ids])
            self.tar_standing_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[0][env_ids])
            self.tar_held_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[2][env_ids])
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_bps = self.box_bps
            tar_pos = self._target_pos
            tar_rot = self._target_rot
            # Note: Update Standing points and held points only can be after the box_states is updated
            self.update_standing_and_held_points(box_states, tar_pos, tar_rot)
            box_standing_points = self.box_standing_points
            tar_standing_points = self.tar_standing_points
            density = self.asset_density
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            tar_pos = self._target_pos[env_ids]
            tar_rot = self._target_rot[env_ids]
            box_bps = self.box_bps[:, env_ids, :]
            self.update_standing_and_held_points(
                self._box_states, self._target_pos, self._target_rot, env_ids)
            box_standing_points = self.box_standing_points[env_ids]
            tar_standing_points = self.tar_standing_points[env_ids]
            density = self.asset_density[env_ids]

        obs = compute_carrybox_observations(
            root_states, box_states, tar_pos, tar_rot, box_bps, box_standing_points, tar_standing_points, 
            density
        )

        self.record_step += 1
        if self.log_success:
            distance_to_target = torch.norm(box_states[..., 0:3] - tar_pos, dim=-1)
            self._distance_to_target = np.array(distance_to_target.cpu().numpy())
            success_env_mask = self._distance_to_target< 0.2
            success_env_id = np.where(success_env_mask)

            if self._distance_to_target[success_env_mask].size > 0:
                mean_error = self._distance_to_target[success_env_mask].mean()
            else:
                mean_error = 100

            self.log_success_rate.append(success_env_mask.sum() / self.num_envs)
            self.log_success_precision.append(mean_error)
            print("Max Success rate: ", max(self.log_success_rate))
            print("Max Success precision: ", min(self.log_success_precision))

            print(self.record_step)
        return obs

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 76
        return obs_size

    def _compute_reset(self):
        box_pos = self._box_states[..., 0:3]
        tar_pos = self._target_pos[..., 0:3]
        prev_box_pos = self._prev_box_pos
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, self._rigid_body_pos, self._box_contact_forces,
            self._lift_body_ids, self.max_episode_length,
            self._enable_early_termination, self._termination_heights,
            box_pos, tar_pos, prev_box_pos, dt_tensor, hand_positions
        )
        return

    def _compute_reward(self, actions):
        # LLM动作执行已经移到 LLMManager.update() 中，这里只需要处理机器人更新
        if getattr(self.llm_manager, 'llm_mode_active', False):
            self._update_mobile_robots_llm_mode()
        else:
            self._update_mobile_robots()
        obstacle_reward_w = 0.1
        walk_pos_reward_w = 0.1
        walk_vel_reward_w = 0.1
        walk_face_reward_w = 0.1
        held_hand_reward_w = 0.4
        held_height_reward_w = 0.0
        carry_box_reward_pos_far_w = 0.1
        carry_box_reward_velocity_w = 0.0
        carry_box_reward_pos_near_w = 0.2
        carry_box_face_reward_w = 0.2
        carry_box_dir_reward_w = 0.1
        putdown_reward_w = 0.1

        box_pos = self._box_states[..., 0:3]  # Box position
        box_height = box_pos[..., 2]
        box_rot = self._box_states[..., 3:7]  # Box rotation
        prev_box_pos = self._prev_box_pos
        box_standing_pos = self.box_standing_points
        box_held_pos = self.box_held_points
        held_point_height = box_held_pos[..., 2]
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        root_pos = self._humanoid_root_states[..., 0:3]  # 3d state
        root_rot = self._humanoid_root_states[..., 3:7]  # 4d state
        prev_root_pos = self._prev_root_pos
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]
        tar_pos = self._target_pos
        tar_rot = self._target_rot

        walk_pos_reward, walk_vel_reward, walk_face_reward = compute_walk_reward(
            root_pos, root_rot, prev_root_pos, box_standing_pos, dt_tensor)

        held_hand_reward = compute_contact_reward(
            hand_positions, box_held_pos, root_pos, box_standing_pos, box_pos, tar_pos)

        height_reward = compute_height_reward(held_point_height)
        
        carry_box_reward_pos_far, carry_box_reward_velocity, \
            carry_box_reward_pos_near, carry_box_face_reward, \
            carry_box_dir_reward, put_down_height_reward = compute_carry_reward(
                root_pos, root_rot, box_pos, box_rot, prev_box_pos, tar_pos, tar_rot, held_point_height, dt_tensor)
        
        obs_reward = compute_obs_reward(
            root_pos, root_rot, prev_root_pos, box_standing_pos, dt_tensor)


        self.rew_buf[:] = walk_pos_reward_w * walk_pos_reward + \
            walk_vel_reward_w * walk_vel_reward + \
            walk_face_reward_w * walk_face_reward + \
            held_hand_reward_w * held_hand_reward + \
            held_height_reward_w * height_reward + \
            carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
            carry_box_reward_velocity_w * carry_box_reward_velocity + \
            carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
            carry_box_face_reward_w * carry_box_face_reward + \
            carry_box_dir_reward_w * carry_box_dir_reward + \
            putdown_reward_w * put_down_height_reward

        walk_reward = walk_pos_reward_w * walk_pos_reward + \
            walk_vel_reward_w * walk_vel_reward + \
            walk_face_reward_w * walk_face_reward
        contact_reward = held_hand_reward_w * held_hand_reward
        carry_reward = carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
            carry_box_reward_velocity_w * carry_box_reward_velocity + \
            carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
            carry_box_face_reward_w * carry_box_face_reward + \
            carry_box_dir_reward_w * carry_box_dir_reward + \
            putdown_reward_w * put_down_height_reward

        box_half_height = self._height_box_size / 2.0
        height_diff = compute_box_raise_height(box_half_height, box_height)
        return

    def _update_task(self):
        return

    def _reset_task(self, env_ids):
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        tar_pos = self._target_pos
        tar_rot = self._target_rot

        box_bps = self.box_bps
        lfus = box_bps[0]
        lfds = box_bps[1]
        lbus = box_bps[2]
        lbds = box_bps[3]

        rfus = box_bps[4]
        rfds = box_bps[5]
        rbus = box_bps[6]
        rbds = box_bps[7]

        tar_lfus = convert_static_point_to_world(lfus, tar_pos, tar_rot)
        tar_lfds = convert_static_point_to_world(lfds, tar_pos, tar_rot)
        tar_lbus = convert_static_point_to_world(lbus, tar_pos, tar_rot)
        tar_lbds = convert_static_point_to_world(lbds, tar_pos, tar_rot)
        tar_rfus = convert_static_point_to_world(rfus, tar_pos, tar_rot)
        tar_rfds = convert_static_point_to_world(rfds, tar_pos, tar_rot)
        tar_rbus = convert_static_point_to_world(rbus, tar_pos, tar_rot)
        tar_rbds = convert_static_point_to_world(rbds, tar_pos, tar_rot)

        verts1 = torch.cat([tar_lfus, tar_lfds], dim=-1).cpu().numpy()
        verts2 = torch.cat([tar_lbus, tar_lbds], dim=-1).cpu().numpy()
        verts3 = torch.cat([tar_rfus, tar_rfds], dim=-1).cpu().numpy()
        verts4 = torch.cat([tar_rbus, tar_rbds], dim=-1).cpu().numpy()
        verts5 = torch.cat([tar_lfus, tar_lbus], dim=-1).cpu().numpy()
        verts6 = torch.cat([tar_lbus, tar_rbus], dim=-1).cpu().numpy()
        verts7 = torch.cat([tar_rbus, tar_rfus], dim=-1).cpu().numpy()
        verts8 = torch.cat([tar_rfus, tar_lfus], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)

        return

### =========================jit functions=========================###
@torch.jit.script
def convert_static_point_to_local_observation(point_pos, root_states, central_pos, central_rot):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    point_states = torch.zeros_like(root_states[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_point_pos = quat_rotate(heading_rot, target_point_staets - root_pos)
    return local_point_pos

@torch.jit.script
def convert_static_point_to_world(point_pos, central_pos, central_rot):
    point_states = torch.zeros_like(central_pos[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    return target_point_staets

@torch.jit.script
def compute_carrybox_observations(root_states, box_states, tar_pos, tar_rot, box_bps, box_standing_points, tar_standing_points, density):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)  # (num_envs, 4)
    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]
    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)
    box_standing_points_xy = box_standing_points[:, 0:3]
    box_standing_points_xy[:, 2] = 0.0
    local_box_standing_points_pos = box_standing_points_xy - root_pos
    local_box_standing_points_pos = quat_rotate(heading_rot, local_box_standing_points_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)
    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos_obs = quat_rotate(heading_rot, local_tar_pos)
    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    tar_standing_points_xy = tar_standing_points[:, 0:3]
    tar_standing_points_xy[:, 2] = 0.0
    local_tar_standing_points_pos = tar_standing_points_xy - root_pos
    local_tar_standing_points_pos = quat_rotate(
        heading_rot, local_tar_standing_points_pos)

    lfus = box_bps[0]
    lfds = box_bps[1]
    lbus = box_bps[2]
    lbds = box_bps[3]
    rfus = box_bps[4]
    rfds = box_bps[5]
    rbus = box_bps[6]
    rbds = box_bps[7]

    box_local_lfus_pos = convert_static_point_to_local_observation(lfus, root_states, box_pos, box_rot)
    box_local_lfds_pos = convert_static_point_to_local_observation(lfds, root_states, box_pos, box_rot)
    box_local_lbus_pos = convert_static_point_to_local_observation(lbus, root_states, box_pos, box_rot)
    box_local_lbds_pos = convert_static_point_to_local_observation(lbds, root_states, box_pos, box_rot)

    box_local_rfus_pos = convert_static_point_to_local_observation(rfus, root_states, box_pos, box_rot)
    box_local_rfds_pos = convert_static_point_to_local_observation(rfds, root_states, box_pos, box_rot)
    box_local_rbus_pos = convert_static_point_to_local_observation(rbus, root_states, box_pos, box_rot)
    box_local_rbds_pos = convert_static_point_to_local_observation(rbds, root_states, box_pos, box_rot)

    tar_local_lfus_pos = convert_static_point_to_local_observation(lfus, root_states, tar_pos, tar_rot)
    tar_local_lfds_pos = convert_static_point_to_local_observation(lfds, root_states, tar_pos, tar_rot)
    tar_local_lbus_pos = convert_static_point_to_local_observation(lbus, root_states, tar_pos, tar_rot)
    tar_local_lbds_pos = convert_static_point_to_local_observation(lbds, root_states, tar_pos, tar_rot)

    tar_local_rfus_pos = convert_static_point_to_local_observation(rfus, root_states, tar_pos, tar_rot)
    tar_local_rfds_pos = convert_static_point_to_local_observation(rfds, root_states, tar_pos, tar_rot)
    tar_local_rbus_pos = convert_static_point_to_local_observation(rbus, root_states, tar_pos, tar_rot)
    tar_local_rbds_pos = convert_static_point_to_local_observation(rbds, root_states, tar_pos, tar_rot)

    obs = torch.cat([local_box_pos, local_box_rot_obs, local_box_vel, local_box_ang_vel], dim=-1)
    obs = torch.cat([box_local_lfus_pos, box_local_lfds_pos, box_local_lbus_pos, box_local_lbds_pos,
                    box_local_rfus_pos, box_local_rfds_pos, box_local_rbus_pos, box_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([local_box_standing_points_pos, obs], dim=-1)
    obs = torch.cat([local_tar_pos_obs, local_tar_rot_obs, obs], dim=-1)
    obs = torch.cat([tar_local_lfus_pos, tar_local_lfds_pos, tar_local_lbus_pos, tar_local_lbds_pos,
                    tar_local_rfus_pos, tar_local_rfds_pos, tar_local_rbus_pos, tar_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([torch.unsqueeze(density, -1), obs], dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids,
                           rigid_body_pos, box_contact_forces, lift_body_ids,
                           max_episode_length, enable_early_termination,
                           termination_heights,
                           box_pos, tar_pos, prev_box_pos, dt_tensor, hand_positions):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0
    box_vel_threshold = 1.0
    box_height_threshold = 0.3
    success_threshold = 0.05

    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        fall_masked_contact_buf = contact_buf.clone()
        fall_masked_contact_buf[:, contact_body_ids, :] = 0

        fall_contact = torch.any(
            torch.abs(fall_masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        box_to_target_distance = torch.norm(box_pos - tar_pos, dim=-1)
        box_in_target = box_to_target_distance < success_threshold

        box_height = box_pos[..., 2]
        delta_box_pos = box_pos - prev_box_pos
        box_vel = delta_box_pos / dt_tensor
        box_vel_xy = box_vel[..., 0:2]
        box_vel_xy_norm = torch.norm(box_vel_xy, dim=-1)
        box_has_velocity_horizontal = box_vel_xy_norm > box_vel_threshold
        box_low = box_height < box_height_threshold
        mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
        hand_high = mean_hand_positions[..., 2] > 0.5

        box_kicked = torch.logical_and(box_has_velocity_horizontal, box_low)
        box_kicked_with_hands_high = torch.logical_and(box_kicked, hand_high)

        has_failed = torch.logical_or(has_fallen, box_kicked_with_hands_high)

        has_failed *= (progress_buf > 1)
        terminated = torch.where(
            has_failed, torch.ones_like(reset_buf), terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_walk_reward(root_pos, root_rot, prev_root_pos, box_standing_pos, dt):
    near_threshold = 0.04
    target_speed = 1.0  # target speed in m/s
    pos_err_scale = 2.0
    vel_err_scale = 2.0

    box_standing_points_pos = box_standing_pos[..., 0:2]
    box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
    box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
    box_pos_reward = torch.exp(-pos_err_scale * box_pos_err)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1)
    box_dir_speed = torch.sum(box_dir * root_vel[..., :2], dim=-1)
    box_vel_err = target_speed - box_dir_speed
    box_vel_err = torch.clamp_min(box_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (box_vel_err * box_vel_err))
    speed_mask = box_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat(root_rot)

    facing_dir = torch.zeros_like(root_pos[..., 0:3])
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)

    facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    near_mask = box_pos_err <= near_threshold
    box_pos_reward[near_mask] = 1.0
    vel_reward[near_mask] = 1.0
    facing_reward[near_mask] = 1.0

    return box_pos_reward, vel_reward, facing_reward

@torch.jit.script
def compute_obs_reward(root_pos, root_rot, prev_root_pos, box_standing_pos, dt):
    near_threshold = 0.04
    target_speed = 1.0  # target speed in m/s
    pos_err_scale = 2.0
    vel_err_scale = 2.0

    box_standing_points_pos = box_standing_pos[..., 0:2]
    box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
    box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
    box_pos_reward = torch.exp(-pos_err_scale * box_pos_err)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1)
    box_dir_speed = torch.sum(box_dir * root_vel[..., :2], dim=-1)
    box_vel_err = target_speed - box_dir_speed
    box_vel_err = torch.clamp_min(box_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (box_vel_err * box_vel_err))
    speed_mask = box_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat(root_rot)

    facing_dir = torch.zeros_like(root_pos[..., 0:3])
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)

    facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    near_mask = box_pos_err <= near_threshold
    box_pos_reward[near_mask] = 1.0
    vel_reward[near_mask] = 1.0
    facing_reward[near_mask] = 1.0

    return box_pos_reward, vel_reward, facing_reward

@torch.jit.script
def compute_contact_reward(hand_positions, box_held_points, root_pos, box_standing_pos, box_pos, tar_pos):
    box_near_threshold = 0.09
    carry_dist_threshold = 0.04
    box_height_threshold = 0.4
    held_pos_err_scale = 5.0
    mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
    hand2box_diff = mean_hand_positions - box_held_points[..., 0:3]
    hands2box_pos_err = torch.sum(hand2box_diff * hand2box_diff, dim=-1)
    hands2box_reward = torch.exp(-held_pos_err_scale * hands2box_pos_err)
    box_height = box_held_points[..., 2]
    target_state_diff = tar_pos - box_pos  # xyz
    target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
    near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
    near_and_low_mask = torch.logical_and(
        near_mask, box_height < box_height_threshold)
    hands2box_reward[near_and_low_mask] = 1.0
    return hands2box_reward

@torch.jit.script
def compute_height_reward(held_point_height):
    target_height = 0.8
    height_err_scale = 10.0
    box_height_diff = target_height - held_point_height
    height_reward = torch.exp(
        -height_err_scale * box_height_diff * box_height_diff)
    return height_reward

@torch.jit.script
def compute_carry_reward(root_pos, root_rot, box_pos, box_rot, prev_box_pos, target_pos, target_rot, held_point_height, dt_tensor):
    target_speed = 1.0  # target speed in m/s
    carry_dist_threshold = 0.04
    height_threshold = 0.6
    tar_pos_err_far_scale = 0.5
    target_pos_err_near_scale = 10.0
    carry_vel_err_scale = 2.0

    x_axis = torch.zeros_like(root_pos[..., 0:3])
    x_axis[..., 0] = 1.0

    box_height = box_pos[..., 2]
    height_mask = box_height < height_threshold

    target_state_diff = target_pos - box_pos  # xyz
    target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
    near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
    target_pos_err_xyz = torch.sum(target_state_diff[..., 0:3] ** 2, dim=-1)
    target_pos_reward_far = torch.exp(-tar_pos_err_far_scale *
                                      target_pos_err_xy)
    target_pos_reward_near = torch.exp(-target_pos_err_near_scale *
                                       target_pos_err_xyz)

    far_and_low_mask = torch.logical_and(~near_mask, height_mask)
    target_pos_reward_far[far_and_low_mask] = 0.0
    target_pos_reward_near[far_and_low_mask] = 0.0
    target_pos_reward_far[near_mask] = 1.0

    tar_dir = target_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_reverse = box_pos[..., 0:2] - target_pos[..., 0:2]
    tar_dir_reverse = torch.nn.functional.normalize(tar_dir_reverse, dim=-1)
    root_heading_rot = torch_utils.calc_heading_quat(root_rot)
    root_facing_dir = quat_rotate(root_heading_rot, x_axis)

    front_mask = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1) > 0
    behind_mask = torch.sum(
        tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1) > 0
    facing_err = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1)
    facing_err[behind_mask] = torch.sum(
        tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1)[behind_mask]
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[height_mask] = 0.0
    facing_reward[near_mask] = 1.0

    delta_box_pos = box_pos - prev_box_pos
    box_vel = delta_box_pos / dt_tensor
    box_tar_dir_speed = torch.sum(tar_dir * box_vel[..., 0:2], dim=-1)
    tar_vel_err = target_speed - box_tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    tar_vel_reward = torch.exp(-carry_vel_err_scale *
                               (tar_vel_err * tar_vel_err))
    tar_speed_mask = box_tar_dir_speed <= 0
    tar_vel_reward[tar_speed_mask] = 0
    tar_vel_reward[height_mask] = 0.0

    box_facing_dir = quat_rotate(box_rot, x_axis)
    tar_facing_dir = quat_rotate(target_rot, x_axis)
    dir_err = torch.sum(
        box_facing_dir[..., 0:2] * tar_facing_dir[..., 0:2], dim=-1)  # xy;higher value indicating better alignment
    dir_reward = torch.clamp_min(dir_err, 0.0)
    dir_reward[~near_mask] = 0.0

    held_points_height = held_point_height - target_pos[..., 2]
    put_down_height_reward = torch.exp(
        -5.0 * held_points_height * held_points_height)
    put_down_height_reward[~near_mask] = 0
    return target_pos_reward_far, tar_vel_reward, target_pos_reward_near, facing_reward, dir_reward, put_down_height_reward

@torch.jit.script
def compute_task_finish(box_pos, tar_pos, success_threshold):
    pos_diff = tar_pos - box_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask

@torch.jit.script
def compute_box_raise_height(box_half_size, box_height):
    height_diff = box_height - box_half_size
    return height_diff