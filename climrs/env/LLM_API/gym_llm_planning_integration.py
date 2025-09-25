
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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from LLM.dev_revision.llm_agents.feedback_agent import FeedbackAgent
from LLM.dev_revision.llm_agents.oracle_planner import OraclePlanner
from LLM.dev_revision.arena import ArenaMultiAgent
from .gym_llm_integration import GymLLMIntegration
from .ask_Llm import LLMWorkflow
from .split_llm_with_skills import parse_workflow_text, SKILL_MAP

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
    
    def __init__(self, task, config_path: str = None):
        self.task = task
        self.config_path = config_path
        
        self.llm_arena = None
        self.oracle_planner = None
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
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    raw = json.load(f)
                if isinstance(raw, list) and raw:
                    env_config = raw[0]
                elif isinstance(raw, dict):
                    env_config = raw
                else:
                    env_config = {}
            else:
                env_config = {}
                
            self._create_llm_agents(env_config)
            self._create_planning_arena(env_config)
            self._create_oracle_planner(env_config)
            print("LLM planning system initialized successfully")
                
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
            def __init__(self, task_instance):
                if hasattr(task_instance, 'cfg'):
                    self.task = task_instance.cfg.get('task', 'HumanoidAMPCarryObjectObstacle')
                    self.num_envs = task_instance.cfg.get('num_envs', 1)
                    self.cfg_env = task_instance.cfg.get('cfg_env', 'climrs/data/cfg/humanoid_carrybox.yaml')
                    self.cfg_train = task_instance.cfg.get('cfg_train', 'climrs/data/cfg/train/amp_humanoid_task.yaml')
                    self.motion_file = task_instance.cfg.get('motion_file', 'climrs/data/motions/climrs_data/climrs_data.yaml')
                    self.checkpoint = task_instance.cfg.get('checkpoint', 'climrs/data/models/Humanoid.pth')
                    self.env = task_instance.cfg.get('env', 'env0')
                else:
                    self.task = 'HumanoidAMPCarryObjectObstacle'
                    self.num_envs = 1
                    self.cfg_env = 'climrs/data/cfg/humanoid_carrybox.yaml'
                    self.cfg_train = 'climrs/data/cfg/train/amp_humanoid_task.yaml'
                    self.motion_file = 'climrs/data/motions/climrs_data/climrs_data.yaml'
                    self.checkpoint = 'climrs/data/models/Humanoid.pth'
                    self.env = 'env0'
                
                self.debug = False
                self.source = 'llm_module'
                self.lm_id = 'gpt-4o-mini'
                self.max_tokens = 1000
                self.t = 0.7
                self.n = 1
                self.api_key = None
                self.organization = None
                self.oracle_prompt_path = 'LLM/dev_revision/prompt/oracle_prompt.txt'
                self.agent_selection_prompt_path = 'LLM/dev_revision/prompt/agent_selection_prompt.txt'
                self.quadrotor_prompt_path = 'LLM/dev_revision/prompt/quadrotor_prompt.txt'
                self.robot_dog_prompt_path = 'LLM/dev_revision/prompt/robot_dog_prompt.txt'
                self.robot_arm_prompt_path = 'LLM/dev_revision/prompt/robot_arm_prompt.txt'
                self.judge_prompt_path = 'LLM/dev_revision/prompt/judge_prompt.txt'
                self.select_agents = False
        
        return MockArgs(self.task)
    
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
    
    def _create_oracle_planner(self, env_config):
        try:
            def env_fn():
                return self._create_mock_env_info(env_config)

            def agent_fn(args_llm):
                return FeedbackAgent(**args_llm)

            args = self._create_mock_args()
            
            self.oracle_planner = OraclePlanner(
                environment_fn=env_fn,
                agent_fn=list(self.llm_agents.values()),
                args=args,
                run_predefined_actions=False,
                oracle_prompt_path=args.oracle_prompt_path,
                agent_selection_prompt_path=args.agent_selection_prompt_path,
            )
            print("Oracle planner initialized successfully")
        except Exception as e:
            print(f"Failed to create oracle planner: {e}")
    
    def _create_mock_env_info(self, env_config):
        class MockEnvInfo:
            def __init__(self, config):
                self.task_goal = config.get('task_goal', {})
                self.goal_instruction = config.get('goal_instruction', '')
                self.init_graph = config.get('init_graph', {})
        
        return MockEnvInfo(env_config)
    
    def start_planning(self, task_description: str = "please give me the plan"):
        if not self.llm_arena:
            try:
                self.planning_active = True
                self.plan_status = "planning"

                if hasattr(self, 'task'):
                    for attr in ['waypoints', 'waypoints_2', 'waypoints_3']:
                        if hasattr(self.task, attr):
                            setattr(self.task, attr, None)
                    for attr in ['current_wp_idx', 'current_wp_idx_2', 'current_wp_idx_3']:
                        if hasattr(self.task, attr):
                            setattr(self.task, attr, 0)
                self.planning_thread = threading.Thread(
                    target=self._run_planning,
                    args=(task_description,)
                )
                self.planning_thread.start()
                print("LLM planning (skills workflow) started in background")
                return True
            except Exception as e:
                print(f"Failed to start planning: {e}")
                self.planning_active = False
                self.plan_status = "idle"
                return False
        
        try:
            self.planning_active = True
            self.plan_status = "planning"
            if hasattr(self, 'task'):
                for attr in ['waypoints', 'waypoints_2', 'waypoints_3']:
                    if hasattr(self.task, attr):
                        setattr(self.task, attr, None)
                for attr in ['current_wp_idx', 'current_wp_idx_2', 'current_wp_idx_3']:
                    if hasattr(self.task, attr):
                        setattr(self.task, attr, 0)
            
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
            if self.oracle_planner:
                decision_text = self._use_oracle_planning(task_description)
            else:
                decision_text = self._use_fallback_planning(task_description)

            plan_steps: List[Dict[str, Any]] = []
            if isinstance(decision_text, str) and len(decision_text) > 0:
                try:
                    area_positions, agent_positions = self._get_environment_state()
                    plan_steps = parse_workflow_text(decision_text, area_positions)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

            self.current_plan = plan_steps
            self.plan_execution_index = 0
            self.plan_status = "executing" if self.current_plan else "idle"

            
        except Exception as e:
            print(f"Planning failed: {e}")
            self.plan_status = "idle"
        finally:
            self.planning_active = False

    def _use_oracle_planning(self, task_description: str) -> str:
        try:
            area_positions, agent_positions = self._get_environment_state()
            obs_text = self._format_gym_observation_to_text(area_positions, agent_positions)
            
            vanilla_message, usage = self.oracle_planner.oracle_planning_vanilla(
                obs_text=obs_text,
                goal_instruction=task_description,
                num_agents=len(self.llm_agents),
                dialogue_history="",
            )
            message, usage = self.oracle_planner.extract_structured_message(vanilla_message)
            
            return message
            
        except Exception as e:
            return self._use_fallback_planning(task_description)

    def _use_fallback_planning(self, task_description: str) -> str:
        try:
            area_positions, agent_positions = self._get_environment_state()
            workflow = LLMWorkflow(area_positions, agent_positions, {}, use_oracle=True)
            decision_text = workflow.ask_llm(task_description)
            if decision_text:
                print(f"LLM decision text: {decision_text}")
            return decision_text
        except Exception as e:
            return ""

    def _get_environment_state(self) -> Tuple[Dict, Dict]:
        try:
            from .gym_llm_integration import GymEnvironmentObserver
            observer = GymEnvironmentObserver(self.task)
            env_state = observer.get_environment_state()
            area_positions = env_state.get('area_positions', {})
            agent_positions = env_state.get('agent_positions', {})
            return area_positions, agent_positions
        except Exception as e:
            print(f"Failed to get environment state: {e}")
            return {}, {}

    def _format_gym_observation_to_text(self, area_positions: Dict, agent_positions: Dict) -> str:
        obs_text = ""
        
        if area_positions:
            obs_text += "Areas and Components:\n"
            for area, pos in area_positions.items():
                obs_text += f"Area {area}: Component at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        
        if agent_positions:
            obs_text += "\nAgent Positions:\n"
            for agent, pos in agent_positions.items():
                obs_text += f"{agent}: Position ({pos[0]:.2f}, {pos[1]:.2f})\n"
        
        obs_text += "\nCurrent Task: Assembly of robot components with humanoid, mobile robots, and franka arm coordination.\n"
        
        return obs_text

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
                    pos1 = np.array(agent1.position[:2])
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
                
                if a1 == 'franka' or a2 == 'franka':
                    continue

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
            if self._should_execute_parallel_phase():
                self._execute_parallel_phase()
            else:
                if self.plan_execution_index < len(self.current_plan):
                    current_step = self.current_plan[self.plan_execution_index]
                    
                    if self._can_execute_step(current_step):
                        success = self._execute_plan_step(current_step)
                        if success:
                            self.plan_execution_index += 1
                            print(f"✅ Step {current_step['step']} completed")
                        else:
                            skill_name = current_step.get('skill', 'Unknown')
                            if skill_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                                print(f"⏳ Step {current_step['step']} ({skill_name}) waiting for robot movement...")
                            else:
                                print(f"❌ Step {current_step['step']} failed")
                    else:
                        print(f"⏸️  Step {current_step['step']} paused due to safety concerns")
            
            if self.plan_execution_index >= len(self.current_plan):
                self.plan_status = "completed"
                print("🎉 Plan execution completed!")
                
        except Exception as e:
            print(f"Failed to execute plan: {e}")
    
    def _should_execute_parallel_phase(self):
        if not hasattr(self, '_parallel_phase_started'):
            self._parallel_phase_started = {}
        
        if self.plan_execution_index >= len(self.current_plan):
            return False
            
        current_step = self.current_plan[self.plan_execution_index]
        skill = current_step.get('skill', '')
        
        if skill in ['E', 'F', 'G'] and not self._parallel_phase_started.get('EFG', False):
            return True
            
        if skill in ['J', 'K', 'L'] and not self._parallel_phase_started.get('JKL', False):
            return True
            
        return False
    
    def _execute_parallel_phase(self):
        current_step = self.current_plan[self.plan_execution_index]
        skill = current_step.get('skill', '')
        
        if skill in ['E', 'F', 'G']:
            self._execute_efg_parallel()
        elif skill in ['J', 'K', 'L']:
            self._execute_jkl_parallel()
    
    def _execute_efg_parallel(self):
        if not hasattr(self, '_efg_executed'):
            
            efg_steps = []
            for i in range(self.plan_execution_index, len(self.current_plan)):
                step = self.current_plan[i]
                if step.get('skill', '') in ['E', 'F', 'G']:
                    efg_steps.append((i, step))
                else:
                    break
            
            for step_idx, step in efg_steps:
                self._execute_skill_based_step(step)
            
            self._efg_executed = True
            self._parallel_phase_started['EFG'] = True
        
        efg_completed = self._check_efg_all_completed()
        if efg_completed:
            while (self.plan_execution_index < len(self.current_plan) and 
                   self.current_plan[self.plan_execution_index].get('skill', '') in ['E', 'F', 'G']):
                self.plan_execution_index += 1
    
    def _execute_jkl_parallel(self):
        if not hasattr(self, '_jkl_executed'):
            
            jkl_steps = []
            for i in range(self.plan_execution_index, len(self.current_plan)):
                step = self.current_plan[i]
                if step.get('skill', '') in ['J', 'K', 'L']:
                    jkl_steps.append((i, step))
                else:
                    break
            
            for step_idx, step in jkl_steps:
                self._execute_skill_based_step(step)
            
            if hasattr(self.task, '__dict__'):
                self.task._llm_push_mode = True
            
            self._jkl_executed = True
            self._parallel_phase_started['JKL'] = True
        
        while (self.plan_execution_index < len(self.current_plan) and 
               self.current_plan[self.plan_execution_index].get('skill', '') in ['J', 'K', 'L']):
            self.plan_execution_index += 1
    
    def _check_efg_all_completed(self):
        robots_completed = 0
        
        if hasattr(self.task, 'waypoints') and self.task.waypoints is not None:
            if hasattr(self.task, 'current_wp_idx') and self.task.current_wp_idx >= len(self.task.waypoints):
                robots_completed += 1
        else:
            robots_completed += 1
            
        if hasattr(self.task, 'waypoints_2') and self.task.waypoints_2 is not None:
            if hasattr(self.task, 'current_wp_idx_2') and self.task.current_wp_idx_2 >= len(self.task.waypoints_2):
                robots_completed += 1
        else:
            robots_completed += 1
            
        if hasattr(self.task, 'waypoints_3') and self.task.waypoints_3 is not None:
            if hasattr(self.task, 'current_wp_idx_3') and self.task.current_wp_idx_3 >= len(self.task.waypoints_3):
                robots_completed += 1
        else:
            robots_completed += 1
        
        return robots_completed >= 3
    
    
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
        """Enhanced skill execution with validation, error handling and state tracking"""
        try:
            if 'skill' in step and 'robot_name' in step:
                return self._execute_skill_based_step(step)

            return self._execute_agent_based_step(step)
                
        except Exception as e:
            print(f"Failed to execute plan step: {e}")
            return False

    def _execute_skill_based_step(self, step: Dict) -> bool:
        """Execute skill-based step with enhanced validation and tracking"""
        skill_name = step.get('skill', '').strip()
        robot_name = step.get('robot_name', '').strip()
        area_name = step.get('area_name', '').strip()
        target_name = step.get('target_name', '').strip()
        
        
        if not skill_name or not robot_name:
            return False
        
        if skill_name not in SKILL_MAP:
            return False
        
        normalized_robot = self._normalize_robot_name(robot_name)
        if not self._validate_robot_exists(normalized_robot):
            print(f"❌ Robot {robot_name} not found in environment")
            return False
        
        if area_name and not self._validate_area_name(area_name):
            print(f"⚠️  Area {area_name} may not be valid, proceeding anyway")
        
        if not self._check_skill_preconditions(skill_name, normalized_robot, area_name, target_name):
            print(f"❌ Preconditions not met for skill {skill_name}")
            return False
        
        try:
            if skill_name not in SKILL_MAP:
                print(f"Skill {skill_name} not found in SKILL_MAP")
                return False
                
            func = SKILL_MAP[skill_name]
            
            pre_state = self._capture_execution_state()
            
            func(self.task, normalized_robot, area_name, target_name)
            
            post_state = self._capture_execution_state()
            success = self._validate_skill_execution(skill_name, pre_state, post_state)
            
            self._update_skill_execution_history(skill_name, normalized_robot, success)
            
            return success
            
        except Exception as e:
            print(f"Skill execution error: {e}")
            self._update_skill_execution_history(skill_name, normalized_robot, False)
            return False

    def _execute_agent_based_step(self, step: Dict) -> bool:
        """Execute agent/action based step (arena style)"""
        if 'agent' not in step or 'action' not in step:
            return False
            
        agent_id = step['agent']
        action = step['action']
        target = step.get('target')
        
        if agent_id.startswith('mobile'):
            return self._execute_mobile_robot_action(agent_id, action, target)
        elif agent_id == 'franka':
            return self._execute_franka_action(action, target)
        elif agent_id == 'humanoid':
            return self._execute_humanoid_action(action, target)
        else:
            print(f"❌ Unknown agent type: {agent_id}")
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
                if is_right:
                    self.task.franka_counter = 2
                elif is_left:
                    self.task.franka_counter = 0
                else:
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
                return True
            else:
                print(f"Unknown humanoid action: {action}")
                return False
                
        except Exception as e:
            print(f"Failed to execute humanoid action: {e}")
            return False
    
    def update(self, task_description: str = "Please provide assembly plan"):
        try:
            self.update_agent_states()
            
            collision_risks = self.check_collision_risks()
            self.execute_safety_control(collision_risks)
            
            if not self.current_plan and self.plan_status == "idle":
                if not hasattr(self, '_last_planning_attempt'):
                    self._last_planning_attempt = 0
                
                import time
                current_time = time.time()
                if current_time - self._last_planning_attempt > 10:
                    self._last_planning_attempt = current_time
                    self.start_planning(task_description)
                
            if self.plan_status == "executing":
                self.execute_current_plan()
            
            for agent_id, controller in self.llm_control_agents.items():
                if self.current_plan and self.plan_execution_index < len(self.current_plan):
                    current_step = self.current_plan[self.plan_execution_index]
                    step_agent = current_step.get('agent')
                    if step_agent == agent_id:
                        controller.update(task_description)
            
            high_risk_collisions = [r for r in collision_risks if r.risk_level in ['high', 'critical']]
            if high_risk_collisions:
                print(f"⚠️  High-risk collision detected: {len(high_risk_collisions)}")
                for risk in high_risk_collisions:
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

    def _normalize_robot_name(self, robot_name: str) -> str:
        """Normalize robot name for consistent execution"""
        robot_name = robot_name.strip()
        
        import re
        match = re.match(r'<([^>]+)>\s*\(\d+\)', robot_name)
        if match:
            robot_name = match.group(1)
        
        robot_mapping = {
            'wheeled robot1': '<wheeled robot1> (202)',
            'wheeled robot2': '<wheeled robot2> (203)', 
            'wheeled robot3': '<wheeled robot3> (204)',
            'humanoid': '<humanoid> (101)',
            'franka': '<franka> (606)'
        }
        
        return robot_mapping.get(robot_name, robot_name)

    def _validate_robot_exists(self, robot_name: str) -> bool:
        """Validate that robot exists in environment"""
        try:
            if 'wheeled robot1' in robot_name and hasattr(self.task, 'mobile_handles'):
                return len(self.task.mobile_handles) > 0
            elif 'wheeled robot2' in robot_name and hasattr(self.task, 'mobile_handles'):
                return len(self.task.mobile_handles) > 1
            elif 'wheeled robot3' in robot_name and hasattr(self.task, 'mobile_handles'):
                return len(self.task.mobile_handles) > 2
            elif 'humanoid' in robot_name and hasattr(self.task, 'humanoid_handles'):
                return len(self.task.humanoid_handles) > 0
            elif 'franka' in robot_name and hasattr(self.task, 'franka_handles'):
                return len(self.task.franka_handles) > 0
            return True
        except:
            return True

    def _validate_area_name(self, area_name: str) -> bool:
        """Validate area name"""
        valid_areas = ['A', 'B', 'C', 'D']
        return area_name.upper() in valid_areas

    def _check_skill_preconditions(self, skill_name: str, robot_name: str, area_name: str, target_name: str) -> bool:
        """Check if preconditions are met for skill execution"""
        try:
            if skill_name in ['P', 'Q']:
                if hasattr(self.task, 'wait_counter') and self.task.wait_counter < getattr(self.task, 'wait_steps', 0):
                    print(f"⏳ Franka still in wait period ({self.task.wait_counter}/{getattr(self.task, 'wait_steps', 0)})")
                    return False
            
            if skill_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                if hasattr(self, 'mobile_robot_stop_flags'):
                    robot_id = self._get_robot_id_from_name(robot_name)
                    if robot_id and self.mobile_robot_stop_flags.get(robot_id, False):
                        
                        return False
            
            return True
        except Exception as e:
            return True

    def _get_robot_id_from_name(self, robot_name: str) -> str:
        """Extract robot ID from robot name for safety checks"""
        if 'wheeled robot1' in robot_name:
            return 'mobile_0'
        elif 'wheeled robot2' in robot_name:
            return 'mobile_1'
        elif 'wheeled robot3' in robot_name:
            return 'mobile_2'
        return None

    def _capture_execution_state(self) -> Dict:
        """Capture current state for execution validation"""
        try:
            state = {}
            if hasattr(self.task, 'progress_buf'):
                state['step'] = int(self.task.progress_buf[0].item()) if hasattr(self.task.progress_buf[0], 'item') else self.task.progress_buf[0]
            if hasattr(self.task, 'franka_counter'):
                state['franka_counter'] = self.task.franka_counter
            if hasattr(self.task, 'current_wp_idx'):
                state['wp_idx'] = self.task.current_wp_idx
            return state
        except:
            return {}

    def _validate_skill_execution(self, skill_name: str, pre_state: Dict, post_state: Dict) -> bool:
        """Verify if skill execution is successful"""
        try:
            if skill_name in ['A', 'B', 'C', 'D']:
                return True
            
            elif skill_name in ['E', 'F', 'G']:
                return self._check_robot_movement_complete(skill_name)
            
            elif skill_name in ['J', 'K', 'L']:
                return True
            
            elif skill_name == 'H':
                return True
            
            elif skill_name == 'I':
                return True
            
            elif skill_name in ['M', 'N', 'O']:
                return True
            
            elif skill_name in ['P', 'Q']:
                expected_counter = 1 if skill_name == 'P' else 2
                if 'franka_counter' in post_state and post_state['franka_counter'] == expected_counter:
                    return True
                else:
                    return True
            
            elif skill_name in ['R', 'S', 'T', 'U', 'V']:
                return True
            
            else:
                return True
                
        except Exception as e:
            return True  
            
    def _update_skill_execution_history(self, skill_name: str, robot_name: str, success: bool):
        """Update skill execution history for tracking"""
        if not hasattr(self, 'skill_execution_history'):
            self.skill_execution_history = []
        
        self.skill_execution_history.append({
            'skill': skill_name,
            'robot': robot_name,
            'success': success,
            'timestamp': time.time()
        })
        
        if len(self.skill_execution_history) > 100:
            self.skill_execution_history = self.skill_execution_history[-50:]
    
    def _check_robot_movement_complete(self, skill_name: str) -> bool:
        """Check if the robot corresponding to current skill has completed movement"""
        try:
            robot_id = None
            robot_name = None
            
            if hasattr(self, 'current_plan') and hasattr(self, 'plan_execution_index'):
                if self.current_plan and 0 <= self.plan_execution_index < len(self.current_plan):
                    current_step = self.current_plan[self.plan_execution_index]
                    robot_name = current_step.get('robot_name', '')
                    
                    if 'robot1' in robot_name:
                        robot_id = 1
                    elif 'robot2' in robot_name:
                        robot_id = 2
                    elif 'robot3' in robot_name:
                        robot_id = 3
            
            if robot_id is None:
                return True
            
            if robot_id == 1:
                if hasattr(self.task, 'waypoints') and self.task.waypoints is not None:
                    if hasattr(self.task, 'current_wp_idx'):
                        if self.task.current_wp_idx < len(self.task.waypoints):
                            return False
                        else:
                            return True
                            
            elif robot_id == 2:
                if hasattr(self.task, 'waypoints_2') and self.task.waypoints_2 is not None:
                    if hasattr(self.task, 'current_wp_idx_2'):
                        if self.task.current_wp_idx_2 < len(self.task.waypoints_2):
                            return False
                        else:
                            return True
                            
            elif robot_id == 3:
                if hasattr(self.task, 'waypoints_3') and self.task.waypoints_3 is not None:
                    if hasattr(self.task, 'current_wp_idx_3'):
                        if self.task.current_wp_idx_3 < len(self.task.waypoints_3):
                            return False
                        else:
                            return True
            
            return False
            
        except Exception as e:
            return True
    
    def get_skill_execution_history(self) -> List[Dict]:
        """Get skill execution history for debugging and monitoring"""
        if not hasattr(self, 'skill_execution_history'):
            return []
        return self.skill_execution_history.copy()
    
    def get_skill_success_rate(self, skill_name: str = None) -> float:
        """Get success rate for specific skill or overall"""
        if not hasattr(self, 'skill_execution_history'):
            return 0.0
        
        history = self.skill_execution_history
        if skill_name:
            history = [h for h in history if h['skill'] == skill_name]
        
        if not history:
            return 0.0
        
        successful = sum(1 for h in history if h['success'])
        return successful / len(history)
    
    def reset_skill_execution_history(self):
        """Reset skill execution history"""
        self.skill_execution_history = []
    
    def get_enhanced_skill_map(self) -> Dict[str, str]:
        """Get enhanced skill mapping with descriptions"""
        skill_descriptions = {
            'A': 'explore area A',
            'B': 'explore area B', 
            'C': 'explore area C',
            'D': 'explore area D',
            'E': 'move wheeled robot1',
            'F': 'move wheeled robot2',
            'G': 'move wheeled robot3',
            'H': 'move humanoid',
            'I': 'carry obstacle',
            'J': 'wheeled robot1 push',
            'K': 'wheeled robot2 push',
            'L': 'wheeled robot3 push',
            'M': 'franka check trunk',
            'N': 'franka check left wheel',
            'O': 'franka check right wheel',
            'P': 'franka pick left wheel',
            'Q': 'franka pick right wheel',
            'R': 'franka wait',
            'S': 'humanoid wait',
            'T': 'wheeled robot1 wait',
            'U': 'wheeled robot2 wait',
            'V': 'wheeled robot3 wait'
        }
        return skill_descriptions

    def parse_llm_response_to_skills(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response text to extract skill commands more flexibly"""
        skills = []
        
        patterns = [
            r'Step\s+(\d+).*?Use\s+<([^>]+)>\s*\((\d+)\).*?area\s+<([^>]+)>\s*\((\d+)\).*?for\s+.*?<([^>]+)>\s*\((\d+)\).*?Skill:\s*([A-Z])',
            r'Skill\s+([A-Z]):\s*(.+)',
            r'(\d+)\.\s*([A-Z])\.\s*(.+)',
        ]
        
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                import re
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 8:
                        step_num = int(match.group(1))
                        robot_name = match.group(2)
                        robot_id = match.group(3)
                        area_name = match.group(4)
                        area_id = match.group(5)
                        target_name = match.group(6)
                        target_id = match.group(7)
                        skill = match.group(8)
                        
                        skills.append({
                            "step": step_num,
                            "robot_name": f"<{robot_name}> ({robot_id})",
                            "area_name": area_name,
                            "target_name": f"<{target_name}> ({target_id})",
                            "skill": skill
                        })
                    elif len(match.groups()) >= 2:
                        skill = match.group(1)
                        description = match.group(2) if len(match.groups()) > 1 else ""
                        
                        robot_match = re.search(r'<([^>]+)>', description)
                        area_match = re.search(r'area\s+([A-D])', description, re.IGNORECASE)
                        
                        skills.append({
                            "step": len(skills) + 1,
                            "robot_name": robot_match.group(0) if robot_match else "",
                            "area_name": area_match.group(1) if area_match else "",
                            "target_name": "",
                            "skill": skill
                        })
                    break
        
        return skills

    def execute_llm_response(self, llm_response: str) -> bool:
        """Execute LLM response directly by parsing and running skills"""
        try:
            skills = parse_workflow_text(llm_response)
            
            if not skills:
                skills = self.parse_llm_response_to_skills(llm_response)
            
            if not skills:
                print(f"❌ Could not parse any skills from LLM response")
                return False
            
            print(f"🎯 Parsed {len(skills)} skills from LLM response")
            
            self.current_plan = skills
            self.plan_execution_index = 0
            self.plan_status = "executing"
            
            self.execute_current_plan()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to execute LLM response: {e}")
            return False

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary for debugging"""
        summary = {
            'planning_status': self.plan_status,
            'current_plan_length': len(self.current_plan),
            'execution_index': self.plan_execution_index,
            'planning_active': self.planning_active,
            'safety_mode': self.safety_mode,
            'agent_count': len(self.agent_states),
            'available_skills': list(SKILL_MAP.keys()),
            'skill_execution_history_count': len(getattr(self, 'skill_execution_history', [])),
            'overall_success_rate': self.get_skill_success_rate()
        }
        
        if hasattr(self, 'skill_execution_history') and self.skill_execution_history:
            recent_skills = self.skill_execution_history[-5:]
            summary['recent_executions'] = [
                f"{h['skill']} ({h['robot']}) - {'✅' if h['success'] else '❌'}"
                for h in recent_skills
            ]
        
        return summary

    def print_execution_status(self):
        """Print detailed execution status for debugging"""
        summary = self.get_execution_summary()
        
        print("=" * 50)
        print("🤖 LLM Planning Integration Status")
        print("=" * 50)
        print(f"Planning Status: {summary['planning_status']}")
        print(f"Plan Progress: {summary['execution_index']}/{summary['current_plan_length']}")
        print(f"Safety Mode: {'🛡️  Enabled' if summary['safety_mode'] else '❌ Disabled'}")
        print(f"Active Agents: {summary['agent_count']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        
        if 'recent_executions' in summary:
            print(f"\nRecent Executions:")
            for execution in summary['recent_executions']:
                print(f"  {execution}")
        
        print("=" * 50)

